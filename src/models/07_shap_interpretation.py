"""
07_shap_interpretation.py

Phase 8: Model Interpretation (Consensus SHAP)
Explains the predictions of the Champion XGBoost model using SHAP.
- Computes exact SHAP values on the hold-out test set for local explanations (Beeswarm).
- Runs 100-iteration Consensus SHAP (bootstrapping the test set) to generate 
  robust 95% Confidence Intervals for global feature importance.
- Exports all raw arrays and summary tables for downstream visualization scripts.
"""

import time
import json
import joblib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import shap
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

TENSOR_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors"
COHORT_DIR = BASE_DIR / "data" / "processed" / "mimiciv"
SPLITS_DIR = BASE_DIR / "outputs" / "baselines" / "train_test_split"

# Inputs
CHAMPION_MODEL_FILE = BASE_DIR / "outputs" / "champion" / "models" / "champion_xgboost.joblib"
FEAT_NAMES_FILE = BASE_DIR / "outputs" / "champion" / "feature_names" / "champion_features.json"

# Outputs
OUT_SHAP_DIR = BASE_DIR / "outputs" / "shap"
OUT_SHAP_DATA = OUT_SHAP_DIR / "data"
OUT_SHAP_DATA.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAPS = 100

def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    set_seed(RANDOM_STATE)
    print("[*] Initiating Phase 8: Consensus SHAP Interpretation...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD CHAMPION MODEL & FEATURE NAMES
    # ---------------------------------------------------------
    if not CHAMPION_MODEL_FILE.exists() or not FEAT_NAMES_FILE.exists():
        print("[ERROR] Champion model or feature names not found. Run Phase 4 first.")
        return
        
    print("    -> Loading locked Champion XGBoost model...")
    champion_xgb = joblib.load(CHAMPION_MODEL_FILE)
    
    with open(FEAT_NAMES_FILE, "r") as f:
        feature_names = json.load(f)

    # ---------------------------------------------------------
    # 2. RECONSTRUCT TEST FEATURE SPACE
    # ---------------------------------------------------------
    print("    -> Reconstructing the exact test feature space...")
    X_imputed = np.load(TENSOR_DIR / "sepsis_imputed_tensor.npy")
    stay_ids = np.load(TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    
    df_cohort = pl.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    
    idx_train_val = np.load(SPLITS_DIR / "train_indices.npy")
    idx_test = np.load(SPLITS_DIR / "test_indices.npy")

    # Static Features
    static_cols = [col for col in ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"] if col in df_cohort.columns]
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
        
    scaler_static = StandardScaler().fit(df_static.fillna(0).values[idx_train_val])
    X_static = scaler_static.transform(df_static.fillna(0).values)

    # Aggregated Features
    X_mean, X_min = np.mean(X_imputed, axis=1), np.min(X_imputed, axis=1)
    X_max, X_std = np.max(X_imputed, axis=1), np.std(X_imputed, axis=1)
    
    scaler_agg = StandardScaler().fit(np.concatenate([X_mean, X_min, X_max, X_std], axis=1)[idx_train_val])
    X_temporal_agg = scaler_agg.transform(np.concatenate([X_mean, X_min, X_max, X_std], axis=1))

    X_fused = np.concatenate([X_static, X_temporal_agg], axis=1)
    X_test = X_fused[idx_test]
    df_test_features = pd.DataFrame(X_test, columns=feature_names)

    # ---------------------------------------------------------
    # 3. EXACT SHAP (FOR BEESWARM / LOCAL EXPLANATIONS)
    # ---------------------------------------------------------
    print("    -> Initializing TreeExplainer and computing Exact SHAP values...")
    explainer = shap.TreeExplainer(champion_xgb)
    
    # These exact values are what you will use for the Beeswarm plot
    shap_values_exact = explainer.shap_values(df_test_features)
    
    np.save(OUT_SHAP_DATA / "shap_values_exact_test.npy", shap_values_exact)
    df_test_features.to_csv(OUT_SHAP_DATA / "test_features_scaled.csv", index=False)

    # ---------------------------------------------------------
    # 4. CONSENSUS SHAP (100-ITERATION BOOTSTRAP)
    # ---------------------------------------------------------
    print(f"    -> Running Consensus SHAP ({N_BOOTSTRAPS} iterations) for Confidence Intervals...")
    rng = np.random.default_rng(RANDOM_STATE)
    test_size = len(X_test)
    
    # Store the Mean Absolute SHAP for each feature across all iterations
    # Shape: (N_BOOTSTRAPS, N_FEATURES)
    bootstrap_global_importance = np.zeros((N_BOOTSTRAPS, len(feature_names)))

    for i in range(N_BOOTSTRAPS):
        if (i + 1) % 10 == 0:
            print(f"       - Completed {i + 1}/{N_BOOTSTRAPS} iterations...")
            
        idx = rng.choice(test_size, size=test_size, replace=True)
        X_boot = X_test[idx]
        
        # TreeExplainer is extremely fast, allowing us to rapidly recalculate
        shap_boot = explainer.shap_values(X_boot)
        bootstrap_global_importance[i, :] = np.abs(shap_boot).mean(axis=0)

    # ---------------------------------------------------------
    # 5. AGGREGATE & EXPORT
    # ---------------------------------------------------------
    print("    -> Aggregating Consensus metrics and exporting...")
    
    mean_importance = np.mean(bootstrap_global_importance, axis=0)
    lower_ci = np.percentile(bootstrap_global_importance, 2.5, axis=0)
    upper_ci = np.percentile(bootstrap_global_importance, 97.5, axis=0)
    
    df_consensus = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_importance,
        "Lower_95CI": lower_ci,
        "Upper_95CI": upper_ci
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    df_consensus.to_csv(OUT_SHAP_DATA / "consensus_feature_importance.csv", index=False)
    
    # Export top 20 to JSON for easy manuscript text generation
    with open(OUT_SHAP_DATA / "top_20_consensus_features.json", "w") as f:
        json.dump(df_consensus.head(20).to_dict(orient="records"), f, indent=4)

    # ---------------------------------------------------------
    # 6. CONSOLE REPORT
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" TOP 10 CLINICAL DRIVERS OF SEPSIS MORTALITY (CONSENSUS SHAP)")
    print("="*75)
    for i, row in df_consensus.head(10).iterrows():
        print(f" {i+1:>2}. {row['Feature']:<25} | {row['Mean_Abs_SHAP']:.4f} [95% CI: {row['Lower_95CI']:.4f} - {row['Upper_95CI']:.4f}]")
    print("="*75)

    elapsed = time.time() - start_time
    print(f"[*] SHAP interpretation completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()