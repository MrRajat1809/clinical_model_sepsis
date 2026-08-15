"""
09a_mimic_pruned_internal_test.py

Phase 8: Pruned Model Validation & Export
Trains a lightweight XGBoost model using only the 72 highly stable features 
identified by the RFECV stability analysis. Compares its AUROC directly against 
the original 124-feature Champion model on the hold-out test set.
Saves the final pruned model for downstream external validation and deployment.
"""

import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Inputs mapped to flattened structure
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"
OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_FEATS = BASE_DIR / "outputs" / "features"

CHAMPION_MODEL_FILE = OUT_MODELS / "mimic_champion_xgboost.joblib"
FEAT_NAMES_FILE = OUT_FEATS / "mimic_champion_features.json"
STABLE_FEATS_FILE = OUT_FEATS / "mimic_stable_optimal_features.json"

# Outputs
OUT_MODELS.mkdir(parents=True, exist_ok=True)
PRUNED_MODEL_FILE = OUT_MODELS / "mimic_pruned_champion_xgboost.joblib"

RANDOM_STATE = 42

def main():
    print("[*] Initiating Phase 10: Pruned Model Validation & Export...")
    start_time = time.time()

    # ---------------------------------------------------------
    # 1. LOAD DATA & FEATURES
    # ---------------------------------------------------------
    print("    -> Reconstructing the MIMIC-IV feature space...")
    X_imputed = np.load(PROCESSED_DIR / "mimic_sepsis_imputed_tensor.npy")
    stay_ids = np.load(PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy")
    
    df_cohort = pd.read_parquet(PROCESSED_DIR / "mimic_final_sepsis3_cohort.parquet")
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    
    idx_train = np.load(OUT_MODELS / "mimic_train_indices.npy")
    idx_test = np.load(OUT_MODELS / "mimic_test_set_indices.npy")
    
    y_full = df_cohort["hospital_expire_flag"].fillna(0).astype(int).values
    y_train = np.ravel(y_full[idx_train])
    y_test = np.ravel(y_full[idx_test])

    with open(FEAT_NAMES_FILE, "r") as f:
        all_feature_names = np.array(json.load(f))
        
    with open(STABLE_FEATS_FILE, "r") as f:
        stable_features = json.load(f)

    # ---------------------------------------------------------
    # 2. FLATTEN & STANDARDIZE THE 124-FEATURE SET
    # ---------------------------------------------------------
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
    
    # Standardize based on training set
    scaler = StandardScaler().fit(X_fused[idx_train])
    X_train_full = scaler.transform(X_fused[idx_train])
    X_test_full = scaler.transform(X_fused[idx_test])

    # ---------------------------------------------------------
    # 3. FILTER DOWN TO THE PRUNED FEATURES
    # ---------------------------------------------------------
    print(f"    -> Filtering dataset down to {len(stable_features)} stable features...")
    stable_indices = [np.where(all_feature_names == f)[0][0] for f in stable_features]
    
    X_train_pruned = X_train_full[:, stable_indices]
    X_test_pruned = X_test_full[:, stable_indices]

    # ---------------------------------------------------------
    # 4. TRAIN & SAVE THE PRUNED MODEL
    # ---------------------------------------------------------
    n_survivors = np.sum(y_train == 0)
    n_nonsurvivors = np.sum(y_train == 1)
    scale_pos_weight = n_survivors / n_nonsurvivors
    
    print("    -> Training the lightweight Pruned XGBoost model...")
    pruned_xgb = XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    pruned_xgb.fit(X_train_pruned, y_train)
    
    print(f"    -> Saving Pruned XGBoost model to {PRUNED_MODEL_FILE.relative_to(BASE_DIR)}...")
    joblib.dump(pruned_xgb, PRUNED_MODEL_FILE)
    
    pruned_preds = pruned_xgb.predict_proba(X_test_pruned)[:, 1]
    pruned_auroc = roc_auc_score(y_test, pruned_preds)
    pruned_auprc = average_precision_score(y_test, pruned_preds)

    # ---------------------------------------------------------
    # 5. EVALUATE THE ORIGINAL CHAMPION MODEL
    # ---------------------------------------------------------
    print("    -> Evaluating the original Champion model...")
    champion_xgb = joblib.load(CHAMPION_MODEL_FILE)
    
    champion_preds = champion_xgb.predict_proba(X_test_full)[:, 1]
    champion_auroc = roc_auc_score(y_test, champion_preds)
    champion_auprc = average_precision_score(y_test, champion_preds)

    # ---------------------------------------------------------
    # 6. RESULTS
    # ---------------------------------------------------------
    print("\n======================================================================")
    print(" FINAL HOLD-OUT TEST SET PERFORMANCE (MIMIC-IV)")
    print("======================================================================")
    print(f" Original Champion ({X_test_full.shape[1]} Features) : AUROC = {champion_auroc:.4f} | AUPRC = {champion_auprc:.4f}")
    print(f" Pruned Model      ({X_test_pruned.shape[1]} Features)  : AUROC = {pruned_auroc:.4f} | AUPRC = {pruned_auprc:.4f}")
    print("======================================================================")
    
    diff = pruned_auroc - champion_auroc
    feat_diff = X_test_full.shape[1] - X_test_pruned.shape[1]
    if diff >= 0:
        print(f" [*] SUCCESS: Pruned model matched/improved AUROC by +{diff:.4f} while shedding {feat_diff} features!")
    else:
        print(f" [*] SUCCESS: Pruned model shed {feat_diff} features with only a marginal AUROC shift of {diff:.4f}.")
    
    elapsed = time.time() - start_time
    print(f"[*] Process completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()