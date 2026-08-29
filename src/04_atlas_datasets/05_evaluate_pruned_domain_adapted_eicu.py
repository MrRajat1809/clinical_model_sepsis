"""
Evaluate the reduced-feature model on the transport-aligned eICU cohort.

Asks whether the RFECV-stable features resist the geometric distortion that
transport introduces. If the reduced model degrades less than the full one, the
stable subset is carrying signal that survives alignment; if it degrades more,
the subset was more dependent on the exact distribution it was selected in.

Applies the same column reordering and MIMIC-IV standardisation as the full-model
evaluation before selecting the stable subset by name. The subset size is read
from the feature list at run time.

The scaled, correctly ordered feature matrix is exported for the portability map
in the demographics notebook, which compares per-feature signal retention with
and without transport.

Reads:
    the atlas features and metadata
    outputs/models/mimic_pruned_champion_xgboost.joblib
    outputs/features/mimic_stable_optimal_features.json
Writes:
    outputs/features/eicu_ot_pruned_{tensor, labels}.npy
    outputs/metrics/eicu_pruned_ot_metrics.json
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore")

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

def main():
    print("[*] Initiating Pruned Domain-Adapted External Validation...")
    start_time = time.time()
    
    # Paths
    atlas_npy = BASE_DIR / "data" / "processed" / "atlas" / "atlas_sepsis_features.npy"
    atlas_meta = BASE_DIR / "data" / "processed" / "atlas" / "atlas_metadata.parquet"
    
    full_feats_file = BASE_DIR / "outputs" / "features" / "mimic_champion_features.json"
    pruned_feats_file = BASE_DIR / "outputs" / "features" / "mimic_stable_optimal_features.json"
    model_file = BASE_DIR / "outputs" / "models" / "mimic_pruned_champion_xgboost.joblib"
    
    # Paths for Scaler Reconstruction (Fix P1-6)
    mimic_tensor = BASE_DIR / "data" / "processed" / "mimiciv" / "mimic_sepsis_imputed_tensor.npy"
    mimic_cohort = BASE_DIR / "data" / "processed" / "mimiciv" / "mimic_final_sepsis3_cohort.parquet"
    mimic_train_idx_file = BASE_DIR / "outputs" / "models" / "mimic_train_indices.npy"
    mimic_stay_ids_file = BASE_DIR / "data" / "processed" / "mimiciv" / "mimic_sepsis_tensor_stay_ids.npy"
    
    # New Artifact Outputs for Portability Map
    out_pruned_ot_tensor = BASE_DIR / "outputs" / "features" / "eicu_ot_pruned_tensor.npy"
    out_pruned_ot_labels = BASE_DIR / "outputs" / "features" / "eicu_ot_pruned_labels.npy"
    out_pruned_metrics = BASE_DIR / "outputs" / "metrics" / "eicu_pruned_ot_metrics.json"
    
    if not atlas_npy.exists() or not model_file.exists():
        print("[ERROR] Required Atlas or Model files are missing.")
        return

    # 1. Reconstruct the MIMIC Scalers (Fix P1-6 & Leakage Fix P2-1)
    print("    -> Reconstructing original scalers from MIMIC-IV to prevent feature drift...")
    X_mimic_tensor = np.load(mimic_tensor)
    mimic_train_idx = np.load(mimic_train_idx_file)
    mimic_stay_ids = np.load(mimic_stay_ids_file)
    
    df_mimic = pl.read_parquet(mimic_cohort).to_pandas()
    df_mimic = pd.DataFrame({"stay_id": mimic_stay_ids}).merge(df_mimic, on="stay_id", how="left")
    
    df_mimic_static = df_mimic[["age", "baseline_sofa"]].copy()
    
    scaler_static = StandardScaler().fit(df_mimic_static.fillna(0).values[mimic_train_idx])

    mimic_temporal_raw = np.concatenate([
        np.mean(X_mimic_tensor, axis=1), np.min(X_mimic_tensor, axis=1),
        np.max(X_mimic_tensor, axis=1), np.std(X_mimic_tensor, axis=1)
    ], axis=1)
    scaler_temporal = StandardScaler().fit(mimic_temporal_raw[mimic_train_idx])

    # 2. Load OT Atlas and isolate eICU
    print("    -> Extracting OT-Harmonized eICU data from Atlas...")
    X_atlas = np.load(atlas_npy)
    df_meta = pd.read_parquet(atlas_meta)
    
    eicu_mask = df_meta['cohort_source'].astype(str).str.contains("eICU", case=False, na=False).values
    y_eicu = df_meta.loc[eicu_mask, 'hospital_expire_flag'].values
    X_eicu_ot_raw = X_atlas[eicu_mask]
    
    # 3. Reorder and Scale the Atlas Data (Fix P1-5)
    print("    -> Reordering and scaling features into MIMIC statistical space...")
    ATLAS_SRC = X_eicu_ot_raw

    # Atlas layout, written by 04_atlas_datasets/01a: temporal block first, then
    # the static block in MODEL_STATICS order. Deriving the split from the names
    # means a change to the static set fails loudly rather than silently
    # selecting the wrong columns.
    MODEL_STATICS = ["age", "baseline_sofa"]
    N_TEMPORAL = ATLAS_SRC.shape[1] - len(MODEL_STATICS)
    assert N_TEMPORAL == 120, f"expected 120 temporal columns, got {N_TEMPORAL}"

    X_ot_temporal = X_eicu_ot_raw[:, :N_TEMPORAL]
    X_ot_static = X_eicu_ot_raw[:, N_TEMPORAL:]

    static_perm = [MODEL_STATICS.index(c) for c in ["age", "baseline_sofa"]]
    X_ot_static_reordered = X_ot_static[:, static_perm]
    
    X_ot_static_scaled = scaler_static.transform(X_ot_static_reordered)
    X_ot_temporal_scaled = scaler_temporal.transform(X_ot_temporal)
    
    # Concatenate to form the full 122-D correctly ordered and scaled array
    X_eicu_scaled = np.concatenate([X_ot_static_scaled, X_ot_temporal_scaled], axis=1)

    # 4. Subset to the Pruned Features
    print("    -> Isolating the stable features from the harmonized space...")
    with open(full_feats_file, "r") as f:
        full_feats = json.load(f)
    with open(pruned_feats_file, "r") as f:
        pruned_feats = json.load(f)
        
    feat_indices = [full_feats.index(feat) for feat in pruned_feats]
    X_eicu_pruned = X_eicu_scaled[:, feat_indices]
    
    print(f"       - OT eICU Cohort Size: {len(y_eicu):,}")
    print(f"       - Final Feature Vector Shape: {X_eicu_pruned.shape}")

    # 5. Save Artifacts for Notebook Analysis
    print("    -> Saving artifacts for the Pruned Portability Map...")
    np.save(out_pruned_ot_tensor, X_eicu_pruned)
    np.save(out_pruned_ot_labels, y_eicu)
    
    # 6. Predict
    print("    -> Loading locked Pruned XGBoost model (Inference Mode)...")
    model = joblib.load(model_file)
    
    X_df = pd.DataFrame(X_eicu_pruned, columns=pruned_feats)
    y_prob = model.predict_proba(X_df)[:, 1]
    
    # 7. Evaluate with Bootstrap
    print("    -> Running Evaluation & 1000-Iteration Bootstrap...")
    aucs, aprs, briers = [], [], []
    for _ in range(1000):
        idx = resample(np.arange(len(y_eicu)))
        if len(np.unique(y_eicu[idx])) < 2: 
            continue
        
        aucs.append(roc_auc_score(y_eicu[idx], y_prob[idx]))
        aprs.append(average_precision_score(y_eicu[idx], y_prob[idx]))
        briers.append(brier_score_loss(y_eicu[idx], y_prob[idx]))
        
    auc_m, auc_l, auc_u = np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)
    apr_m, apr_l, apr_u = np.mean(aprs), np.percentile(aprs, 2.5), np.percentile(aprs, 97.5)
    brier_m, brier_l, brier_u = np.mean(briers), np.percentile(briers, 2.5), np.percentile(briers, 97.5)
        
    print("\n============================================================")
    print(" OT-ADAPTED PRUNED EXTERNAL VALIDATION (eICU)")
    print("============================================================")
    print(f"    AUROC : {auc_m:.4f}  [95% CI: {auc_l:.4f} - {auc_u:.4f}]")
    print(f"    AUPRC : {apr_m:.4f}  [95% CI: {apr_l:.4f} - {apr_u:.4f}]")
    print(f"    Brier : {brier_m:.4f}  [95% CI: {brier_l:.4f} - {brier_u:.4f}]")
    print("============================================================\n")
    
    with open(out_pruned_metrics, "w") as f:
        json.dump({"AUROC": auc_m, "AUPRC": apr_m, "Brier": brier_m}, f)
    
    print(f"[+] Success! Pruned domain-adapted validation completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
