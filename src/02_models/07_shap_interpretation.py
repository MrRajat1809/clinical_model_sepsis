"""
Consensus SHAP attribution across 50 model seeds.

A single model's SHAP values reflect one particular fit as much as the data, so
this refits the primary configuration 50 times, each on a bootstrap resample of
the training rows under its own seed, and aggregates. Exact tree SHAP values are
computed on the fixed held-out test set for each fit.

Two aggregations are produced:
    patient level  attributions averaged across seeds, for beeswarm plots
    global         mean absolute SHAP per seed, then the mean and 2.5th-97.5th
                   percentile across fits, so feature importance carries an
                   interval reflecting both initialisation and sampling
                   variance

Reads:
    outputs/models/mimic_champion_xgboost.joblib for the configuration
    outputs/features/mimic_champion_features.json
Writes:
    outputs/features/mimic_consensus_feature_importance.csv
    outputs/features/mimic_shap_values_exact_test.npy
    outputs/features/mimic_top_20_consensus_features.json
"""

import os
import time
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

warnings.filterwarnings("ignore")

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"
OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_FEATS = BASE_DIR / "outputs" / "features"

# Input Models & Features
CHAMPION_MODEL_FILE = OUT_MODELS / "mimic_champion_xgboost.joblib"
FEAT_NAMES_FILE = OUT_FEATS / "mimic_champion_features.json"

# Make sure features dir exists for outputs
OUT_FEATS.mkdir(parents=True, exist_ok=True)

N_MODELS = 50
BASE_SEED = 42

def set_seed(seed):
    """Ensures absolute reproducibility."""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- Main Execution ------------------------------------------------------
def main():
    set_seed(BASE_SEED)
    print(f"[*] Initiating Phase 8: True {N_MODELS}-Model Consensus SHAP Interpretation...")
    start_time = time.time()
    
    # --- Load Champion Model & Metadata ----------------------------------
    if not CHAMPION_MODEL_FILE.exists() or not FEAT_NAMES_FILE.exists():
        print(f"[ERROR] Required files not found. Check: {CHAMPION_MODEL_FILE}")
        return
        
    print("    -> Loading locked Champion XGBoost model to extract hyperparameters...")
    base_champion = joblib.load(CHAMPION_MODEL_FILE)
    
    with open(FEAT_NAMES_FILE, "r") as f:
        feature_names = json.load(f)

    # --- Reconstruct Train & Test Feature Spaces -------------------------
    print("    -> Reconstructing and standardizing the exact MIMIC train/test sets...")
    X_imputed = np.load(PROCESSED_DIR / "mimic_sepsis_imputed_tensor.npy")
    stay_ids = np.load(PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy")
    
    df_cohort = pd.read_parquet(PROCESSED_DIR / "mimic_final_sepsis3_cohort.parquet")
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    
    idx_train = np.load(OUT_MODELS / "mimic_train_indices.npy")
    idx_test = np.load(OUT_MODELS / "mimic_test_set_indices.npy")
    
    y = df_cohort["hospital_expire_flag"].values
    y_train = y[idx_train]

    # Static Features
    static_cols = ["age", "baseline_sofa"]
    df_static = df_cohort[[c for c in static_cols if c in df_cohort.columns]].copy()
    X_static = df_static.fillna(0).values

    # Temporal Aggregations
    X_mean = np.mean(X_imputed, axis=1)
    X_min = np.min(X_imputed, axis=1)
    X_max = np.max(X_imputed, axis=1)
    X_std = np.std(X_imputed, axis=1)
    
    X_fused = np.concatenate([X_static, X_mean, X_min, X_max, X_std], axis=1)

    # Standardize strictly using the Training set distribution
    scaler = StandardScaler().fit(X_fused[idx_train])
    X_train_scaled = scaler.transform(X_fused[idx_train])
    X_test_scaled = scaler.transform(X_fused[idx_test])
    
    df_test_features = pd.DataFrame(X_test_scaled, columns=feature_names)

    # --- 50-model Consensus Shap Loop ------------------------------------
    print(f"    -> Training {N_MODELS} independent XGBoost models for Consensus SHAP...")
    
    test_size = len(X_test_scaled)
    n_features = len(feature_names)
    
    # To store raw SHAP values from all 50 models to compute CIs and Beeswarm data
    all_shap_values = np.zeros((N_MODELS, test_size, n_features))
    
    for i in range(N_MODELS):
        if (i + 1) % 10 == 0:
            print(f"        - Completed {i + 1}/{N_MODELS} model runs...")
            
        current_seed = BASE_SEED + i
        
        # Each fit gets a bootstrap of the training rows as well as its own seed.
        # Varying the seed alone measured initialisation scatter and reported it
        # as a confidence interval, which left the bands far too narrow: a 0.03%
        # change in cohort membership moved lactate_Mean from 0.259 to 0.108,
        # outside an interval of 0.229-0.290. Resampling the training data puts
        # sampling variance -- the dominant term, and the one a reader assumes is
        # there -- inside the interval. The test set stays fixed so the
        # attributions remain comparable across fits.
        rng = np.random.default_rng(current_seed)
        boot = rng.integers(0, len(X_train_scaled), len(X_train_scaled))
        
        # Clone model to preserve optimal hyperparams but set a new seed
        model_clone = clone(base_champion)
        model_clone.set_params(random_state=current_seed)
        
        # Suppress XGBoost output during the loop
        model_clone.fit(X_train_scaled[boot], np.asarray(y_train)[boot], verbose=False)
        
        # Compute exact SHAP values for the test set
        explainer = shap.TreeExplainer(model_clone)
        shap_values = explainer.shap_values(X_test_scaled)
        
        all_shap_values[i] = shap_values

    # --- Aggregate & Export ----------------------------------------------
    print("    -> Aggregating Consensus metrics and exporting...")
    
    # 1. Consensus Exact SHAP (Mean of all 50 runs for patient-level beeswarm plots)
    consensus_shap_exact = np.mean(all_shap_values, axis=0)
    np.save(OUT_FEATS / "mimic_shap_values_exact_test.npy", consensus_shap_exact)
    df_test_features.to_csv(OUT_FEATS / "mimic_test_features_scaled.csv", index=False)
    
    # 2. Global Feature Importance (Mean Absolute SHAP)
    # Get mean absolute SHAP for each feature per model, shape: (50, n_features)
    model_mean_abs_shap = np.mean(np.abs(all_shap_values), axis=1)
    
    # Global mean and 95% interval across the 50 bootstrap fits
    global_mean_importance = np.mean(model_mean_abs_shap, axis=0)
    lower_ci = np.percentile(model_mean_abs_shap, 2.5, axis=0)
    upper_ci = np.percentile(model_mean_abs_shap, 97.5, axis=0)
    
    df_consensus = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": global_mean_importance,
        "Lower_95CI": lower_ci,
        "Upper_95CI": upper_ci
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    df_consensus.to_csv(OUT_FEATS / "mimic_consensus_feature_importance.csv", index=False)
    
    with open(OUT_FEATS / "mimic_top_20_consensus_features.json", "w") as f:
        json.dump(df_consensus.head(20).to_dict(orient="records"), f, indent=4)

    # --- Console Report --------------------------------------------------
    print("\n=========================================================================")
    print(f" TOP 10 CLINICAL DRIVERS OF SEPSIS MORTALITY ({N_MODELS}-MODEL CONSENSUS SHAP)")
    print("=========================================================================")
    for i, row in df_consensus.head(10).iterrows():
        print(f" {i+1:>2}. {row['Feature']:<30} | {row['Mean_Abs_SHAP']:.4f} [95% CI: {row['Lower_95CI']:.4f} - {row['Upper_95CI']:.4f}]")
    print("=========================================================================")

    print(f"[*] SHAP interpretation completed in {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()
