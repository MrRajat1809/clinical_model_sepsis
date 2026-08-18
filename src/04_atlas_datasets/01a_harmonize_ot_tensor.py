"""
01a_harmonize_ot_tensor.py

Fuses the MIMIC-IV and eICU-CRD Sepsis-3 cohorts into a unified ~20,000-patient Atlas.

Methodological Notes:
- Computes the exact 124-feature prognostic representation (Mean, Min, Max, Std + 4 Statics) 
  BEFORE running Optimal Transport. This prevents the "smoothing artifact" that destroys 
  temporal variance, and ensures the static features (Age, Gender, CCI, SOFA) are explicitly batch-corrected.
- Uses Sinkhorn Optimal Transport (POT library) to map eICU's 124D feature distributions 
  into MIMIC-IV's latent space, aligning the domains prior to Atlas projection.
- StandardScaler is applied to the 124D vectors to prevent geometric distortion in OT.
- The learned scaler is saved alongside the OT mapper to ensure reproducibility.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import joblib
import ot  # Python Optimal Transport
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Input Directories
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"

# Output Directories (Flattened Taxonomy)
PROCESSED_DIR_ATLAS = BASE_DIR / "data" / "processed" / "atlas"
OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"

for d in [PROCESSED_DIR_ATLAS, OUT_MODELS, OUT_METRICS]:
    d.mkdir(parents=True, exist_ok=True)

# Outputs
ATLAS_FEATURES_FILE = PROCESSED_DIR_ATLAS / "atlas_sepsis_features_124.npy"
ATLAS_IDS_FILE = PROCESSED_DIR_ATLAS / "atlas_stay_ids.npy"
ATLAS_META_FILE = PROCESSED_DIR_ATLAS / "atlas_metadata.parquet"

OT_MAPPER_FILE = OUT_MODELS / "atlas_ot_sinkhorn_mapper.joblib"
OT_SCALER_FILE = OUT_MODELS / "atlas_ot_scaler.joblib"
OT_METRICS_FILE = OUT_METRICS / "atlas_ot_harmonization_metrics.json"

def main():
    print("[*] Initiating Phase 4: Prognostic Representation Harmonization (Sinkhorn OT)...")
    start_time = time.time()

    # ---------------------------------------------------------
    # 1. LOAD DATASETS
    # ---------------------------------------------------------
    print("    -> Loading MIMIC-IV (Source) and eICU (Target) Imputed Tensors...")
    try:
        X_mimic_3d = np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_imputed_tensor.npy")
        ids_mimic = np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_stay_ids.npy", allow_pickle=True)
        X_static_mimic = np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_static.npy", allow_pickle=True)
        df_mimic = pl.read_parquet(PROCESSED_DIR_MIMIC / "mimic_final_sepsis3_cohort.parquet").to_pandas()

        X_eicu_3d = np.load(PROCESSED_DIR_EICU / "eicu_sepsis_imputed_tensor.npy")
        ids_eicu = np.load(PROCESSED_DIR_EICU / "eicu_sepsis_tensor_stay_ids.npy", allow_pickle=True)
        X_static_eicu = np.load(PROCESSED_DIR_EICU / "eicu_sepsis_tensor_static.npy", allow_pickle=True)
        df_eicu = pl.read_parquet(PROCESSED_DIR_EICU / "eicu_final_sepsis3_cohort.parquet").to_pandas()
    except Exception as e:
        print(f"[ERROR] Failed to load processed data arrays. Error: {e}")
        return

    n_mimic = X_mimic_3d.shape[0]
    n_eicu = X_eicu_3d.shape[0]
    
    print(f"       - MIMIC Cohort : {n_mimic:,} patients")
    print(f"       - eICU Cohort  : {n_eicu:,} patients")

    # ---------------------------------------------------------
    # 2. FEATURE ENGINEERING (EXTRACT 124D REPRESENTATION FIRST)
    # ---------------------------------------------------------
    print("\n    -> Flattening Temporal Aggregates and Appending Static Features (124D)...")
    
    # MIMIC Representation
    X_mimic_temporal = np.concatenate([
        np.mean(X_mimic_3d, axis=1), np.min(X_mimic_3d, axis=1),
        np.max(X_mimic_3d, axis=1), np.std(X_mimic_3d, axis=1)
    ], axis=1)
    X_mimic_static_4 = X_static_mimic[:, [0, 1, 5, 6]].astype(np.float32)
    X_mimic_124 = np.concatenate([X_mimic_temporal, X_mimic_static_4], axis=1)

    # eICU Representation
    X_eicu_temporal = np.concatenate([
        np.mean(X_eicu_3d, axis=1), np.min(X_eicu_3d, axis=1),
        np.max(X_eicu_3d, axis=1), np.std(X_eicu_3d, axis=1)
    ], axis=1)
    X_eicu_static_4 = X_static_eicu[:, [0, 1, 5, 6]].astype(np.float32)
    X_eicu_124 = np.concatenate([X_eicu_temporal, X_eicu_static_4], axis=1)

    print(f"       - MIMIC 124D Shape : {X_mimic_124.shape}")
    print(f"       - eICU 124D Shape  : {X_eicu_124.shape}")

    # ---------------------------------------------------------
    # 3. OPTIMAL TRANSPORT HARMONIZATION
    # ---------------------------------------------------------
    print("\n    -> Standardizing 124D features to stabilize OT geometry...")
    ot_scaler = StandardScaler()
    X_mimic_scaled = ot_scaler.fit_transform(X_mimic_124)
    X_eicu_scaled = ot_scaler.transform(X_eicu_124)

    print("    -> Fitting Sinkhorn Transport to project eICU distribution into MIMIC latent space...")
    ot_mapping = ot.da.SinkhornTransport(
        reg_e=0.1, 
        max_iter=1000, 
        norm="median", 
        verbose=True
    )
    
    ot_mapping.fit(Xs=X_eicu_scaled, Xt=X_mimic_scaled)
    
    print("\n    -> Transformation learned! Applying mapping to eICU 124D representation...")
    X_eicu_mapped_scaled = ot_mapping.transform(Xs=X_eicu_scaled)
    X_eicu_mapped_124 = ot_scaler.inverse_transform(X_eicu_mapped_scaled)

    # Calculate centroid alignment metrics
    pre_ot_mse = np.mean((X_eicu_124.mean(axis=0) - X_mimic_124.mean(axis=0))**2)
    post_ot_mse = np.mean((X_eicu_mapped_124.mean(axis=0) - X_mimic_124.mean(axis=0))**2)

    print(f"       - Pre-OT Centroid MSE : {pre_ot_mse:.4f}")
    print(f"       - Post-OT Centroid MSE: {post_ot_mse:.4f} ({(1 - post_ot_mse/pre_ot_mse)*100:.1f}% improvement)")

    # ---------------------------------------------------------
    # 4. CONCATENATE & NORMALIZE UNIFIED ATLAS
    # ---------------------------------------------------------
    print("\n    -> Fusing harmonized 124-feature tensors...")
    X_atlas_124 = np.concatenate([X_mimic_124, X_eicu_mapped_124], axis=0)
    
    mimic_ids_prefixed = np.array([f"MIMIC_{i}" for i in ids_mimic])
    eicu_ids_prefixed = np.array([f"eICU_{i}" for i in ids_eicu])
    atlas_stay_ids = np.concatenate([mimic_ids_prefixed, eicu_ids_prefixed], axis=0)

    print("    -> Aligning cohort metadata dictionaries...")
    shared_cols = [
        "stay_id", "age", "gender", "charlson_comorbidity_index", 
        "baseline_sofa", "baseline_pf_ratio", "hospital_expire_flag"
    ]
    
    df_mimic_align = df_mimic[shared_cols].copy()
    df_mimic_align["cohort_source"] = "MIMIC-IV"
    df_mimic_align["atlas_id"] = mimic_ids_prefixed
    
    df_eicu_align = df_eicu[shared_cols].copy()
    df_eicu_align["cohort_source"] = "eICU-CRD"
    df_eicu_align["atlas_id"] = eicu_ids_prefixed
    
    df_atlas_meta = pd.concat([df_mimic_align, df_eicu_align], axis=0).reset_index(drop=True)
    
    # Clean PyArrow strict-typing violations
    df_atlas_meta["age"] = pd.to_numeric(
        df_atlas_meta["age"].astype(str).str.replace('>', '').str.strip(), errors='coerce'
    ).astype(np.float32)
    
    df_atlas_meta["gender"] = df_atlas_meta["gender"].astype(str).str.upper()
    df_atlas_meta["cohort_source"] = df_atlas_meta["cohort_source"].astype(str)
    df_atlas_meta["atlas_id"] = df_atlas_meta["atlas_id"].astype(str)

    print(f"\n    [ATLAS SUMMARY]")
    print(f"       - Total Patients Formatted : {len(df_atlas_meta):,}")
    print(f"       - Sepsis Mortality Rate    : {df_atlas_meta['hospital_expire_flag'].mean()*100:.2f}%")
    print(f"       - Final 124D Matrix Shape  : {X_atlas_124.shape}")

    # ---------------------------------------------------------
    # 5. EXPORT ARTIFACTS
    # ---------------------------------------------------------
    print("\n    -> Exporting Atlas artifacts to centralized storage...")
    
    np.save(ATLAS_FEATURES_FILE, X_atlas_124)
    np.save(ATLAS_IDS_FILE, atlas_stay_ids)
    df_atlas_meta.to_parquet(ATLAS_META_FILE, index=False)
    
    joblib.dump(ot_mapping, OT_MAPPER_FILE)
    joblib.dump(ot_scaler, OT_SCALER_FILE)
    
    metrics_report = {
        "Total_Patients": int(len(df_atlas_meta)),
        "OT_Method": "Sinkhorn Domain Adaptation (reg_e=0.1, Scaled, 124D)",
        "Pre_OT_Centroid_MSE": float(pre_ot_mse),
        "Post_OT_Centroid_MSE": float(post_ot_mse),
        "Centroid_Alignment_Improvement_Pct": float((1 - post_ot_mse/pre_ot_mse)*100)
    }
    with open(OT_METRICS_FILE, "w") as f:
        json.dump(metrics_report, f, indent=4)

    elapsed = time.time() - start_time
    print(f"\n[+] Success! 124D Sepsis Atlas Harmonized in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()