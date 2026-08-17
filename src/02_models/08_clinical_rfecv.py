"""
08_clinical_rfecv.py

Phase 7: 100-Iteration Recursive Feature Elimination (RFECV)
Identifies the optimal, minimal subset of physiological features required 
to predict sepsis mortality using a 100-iteration stability analysis.
"""

import time
import json
import warnings
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Input Tensors & Cohort (MIMIC Training Set only)
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

# Flattened Global Outputs structure
OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_FEATS = BASE_DIR / "outputs" / "features"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

for d in [OUT_MODELS, OUT_FEATS, OUT_METRICS, OUT_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# [FIX]: Synchronized to the new explicit prefix 
FEAT_NAMES_FILE = OUT_FEATS / "mimic_champion_features.json"

BASE_RANDOM_STATE = 42
N_ITERATIONS = 100

# ==========================================
# CUSTOM SCORER TO BYPASS API CLASH
# ==========================================
def custom_auc_scorer(estimator, X, y):
    """
    Forces the calculation of AUROC using predict_proba.
    This entirely bypasses scikit-learn's buggy internal estimator checks.
    """
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred_proba)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(f"[*] Initiating Phase 9: {N_ITERATIONS}-Iteration Clinical RFECV...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD AND RECONSTRUCT TRAINING DATA
    # ---------------------------------------------------------
    print("    -> Reconstructing the MIMIC-IV training feature space...")
    X_imputed = np.load(PROCESSED_DIR / "mimic_sepsis_imputed_tensor.npy")
    stay_ids = np.load(PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy")
    
    df_cohort = pd.read_parquet(PROCESSED_DIR / "mimic_final_sepsis3_cohort.parquet")
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    
    idx_train = np.load(OUT_MODELS / "mimic_train_indices.npy")
    
    # Force 1D Integer Array
    y_train = df_cohort["hospital_expire_flag"].fillna(0).astype(int).values[idx_train]
    y_train = np.ravel(y_train)

    try:
        with open(FEAT_NAMES_FILE, "r") as f:
            feature_names = np.array(json.load(f))
    except Exception as e:
        print(f"[ERROR] Failed to load {FEAT_NAMES_FILE}. Error: {e}")
        return

    # Flatten the 3D tensor
    static_cols = ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"]
    df_static = df_cohort[[c for c in static_cols if c in df_cohort.columns]].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"].astype(str).str.upper() == "M").astype(int)
    X_static = df_static.fillna(0).values

    X_mean = np.mean(X_imputed, axis=1)
    X_min = np.min(X_imputed, axis=1)
    X_max = np.max(X_imputed, axis=1)
    X_std = np.std(X_imputed, axis=1)
    
    X_fused = np.concatenate([X_static, X_mean, X_min, X_max, X_std], axis=1)
    
    # Isolate training data and standardize
    X_train_raw = X_fused[idx_train]
    X_train_scaled = StandardScaler().fit_transform(X_train_raw)

    # ---------------------------------------------------------
    # 2. CONFIGURE COST-SENSITIVE XGBOOST
    # ---------------------------------------------------------
    n_survivors = np.sum(y_train == 0)
    n_nonsurvivors = np.sum(y_train == 1)
    scale_pos_weight = n_survivors / n_nonsurvivors
    
    print(f"    -> Class Weights (Survivors: {n_survivors}, Non-Survivors: {n_nonsurvivors})")
    
    # ---------------------------------------------------------
    # 3. RUN 100-ITERATION RECURSIVE FEATURE ELIMINATION
    # ---------------------------------------------------------
    print(f"    -> Beginning {N_ITERATIONS}-iteration stability analysis...")
    print("       (Note: Bypassing scikit-learn API bugs with custom scorer)")
    
    feature_selection_counts = np.zeros(len(feature_names))
    
    for i in tqdm(range(N_ITERATIONS), desc="RFECV Iterations"):
        current_seed = BASE_RANDOM_STATE + i
        
        xgb_estimator = XGBClassifier(
            objective="binary:logistic",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            random_state=current_seed,
            n_jobs=-1  
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=current_seed)
        
        rfecv = RFECV(
            estimator=xgb_estimator,
            step=1, 
            cv=cv,
            scoring=custom_auc_scorer,  # USING THE CUSTOM SCORER
            min_features_to_select=10,
            n_jobs=1 
        )
        
        rfecv.fit(X_train_scaled, y_train)
        
        feature_selection_counts += rfecv.support_.astype(int)
        
        # Free memory to prevent RAM overflow over 100 iterations
        del rfecv, cv, xgb_estimator
        gc.collect()

    # ---------------------------------------------------------
    # 4. EXTRACT & EXPORT STABILITY RESULTS
    # ---------------------------------------------------------
    selection_frequencies = (feature_selection_counts / N_ITERATIONS) * 100
    
    df_stability = pd.DataFrame({
        "Feature": feature_names,
        "Selection_Frequency_Pct": selection_frequencies
    }).sort_values(by="Selection_Frequency_Pct", ascending=False).reset_index(drop=True)
    
    # [FIX]: Use Explicit mimic_ prefixes for generated files
    df_stability.to_csv(OUT_METRICS / "mimic_rfecv_100_iteration_stability.csv", index=False)
    
    # Isolate highly stable features (e.g., >80% retention)
    stable_features = df_stability[df_stability["Selection_Frequency_Pct"] >= 80.0]["Feature"].tolist()
    
    with open(OUT_FEATS / "mimic_stable_optimal_features.json", "w") as f:
        json.dump(stable_features, f, indent=4)

    # ---------------------------------------------------------
    # 5. GENERATE SELECTION FREQUENCY PLOT
    # ---------------------------------------------------------
    print("\n    -> Generating RFECV Stability Plot...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 10))
    
    # Plot top 50 features for readability
    plot_data = df_stability.head(50)
    sns.barplot(x="Selection_Frequency_Pct", y="Feature", data=plot_data, palette="viridis")
    
    plt.axvline(x=80, color="firebrick", linestyle="--", linewidth=2, label="80% Stability Threshold")
    
    plt.title(f"Clinical Feature Selection Stability ({N_ITERATIONS} RFECV Iterations)", fontsize=14)
    plt.xlabel("Selection Frequency (%)", fontsize=12)
    plt.ylabel("Clinical Feature", fontsize=12)
    plt.legend(loc="lower right")
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "mimic_rfecv_stability_plot.png", dpi=300)
    plt.close()
    
    print("\n" + "="*70)
    print(" RFECV STABILITY ANALYSIS COMPLETE")
    print("="*70)
    print(f" Total Features Evaluated : {len(feature_names)}")
    print(f" Highly Stable Features   : {len(stable_features)} (Retained in >= 80% of runs)")
    print("="*70)
    
    elapsed = time.time() - start_time
    print(f"[*] Process completed in {elapsed / 60:.1f} minutes. Outputs saved to global directories.")

if __name__ == "__main__":
    main()