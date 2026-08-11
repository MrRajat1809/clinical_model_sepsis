"""
01_build_atlas_tensor.py

Builds the Unified Multi-Center Atlas (Raw).
1. Loads the strictly audited and imputed 3D tensors from MIMIC-IV and eICU.
2. Concatenates them along the patient axis (Axis 0).
3. Merges the cohort metadata, adding a 'source_db' tracker.
4. Saves the final Raw Atlas Tensor and Atlas Cohort for downstream analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# MIMIC Data
MIMIC_TENSOR = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors" / "sepsis_imputed_tensor.npy"
MIMIC_IDS = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors" / "sepsis_tensor_stay_ids.npy"
MIMIC_COHORT = BASE_DIR / "data" / "processed" / "mimiciv" / "final_sepsis3_cohort.parquet"
MIMIC_FEATS = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors" / "sepsis_tensor_features.npy"

# eICU Data
EICU_TENSOR = BASE_DIR / "data" / "processed" / "eicu" / "tensors" / "eicu_sepsis_imputed_tensor.npy"
EICU_IDS = BASE_DIR / "data" / "processed" / "eicu" / "tensors" / "eicu_sepsis_tensor_stay_ids.npy"
EICU_COHORT = BASE_DIR / "data" / "processed" / "eicu" / "eicu_final_sepsis3_cohort.parquet"
EICU_FEATS = BASE_DIR / "data" / "processed" / "eicu" / "tensors" / "eicu_sepsis_tensor_features.npy"

# Output Atlas Data
ATLAS_DIR = BASE_DIR / "data" / "processed" / "atlas"
ATLAS_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_TENSOR_OUT = ATLAS_DIR / "atlas_imputed_tensor.npy"
ATLAS_IDS_OUT = ATLAS_DIR / "atlas_tensor_stay_ids.npy"
ATLAS_COHORT_OUT = ATLAS_DIR / "atlas_final_cohort.parquet"
ATLAS_FEATS_OUT = ATLAS_DIR / "atlas_tensor_features.npy"

def main():
    print("[*] Initiating Multi-Center Atlas Construction...")
    
    # 1. Feature Parity Check
    print("    -> Verifying absolute feature parity...")
    m_feats = list(np.load(MIMIC_FEATS))
    e_feats = list(np.load(EICU_FEATS))
    
    if m_feats != e_feats:
        print("[ERROR] Feature sets do not match perfectly. Cannot build Atlas.")
        return
        
    np.save(ATLAS_FEATS_OUT, m_feats)
    
    # 2. Load Tensors & IDs
    print("    -> Loading 3D Imputed Tensors...")
    X_mimic = np.load(MIMIC_TENSOR)
    ids_mimic = np.load(MIMIC_IDS)
    
    X_eicu = np.load(EICU_TENSOR)
    ids_eicu = np.load(EICU_IDS)
    
    print(f"       - MIMIC Shape: {X_mimic.shape}")
    print(f"       - eICU Shape : {X_eicu.shape}")
    
    # 3. Concatenate Tensors
    print("    -> Fusing Tensors into Global Atlas...")
    X_atlas = np.concatenate([X_mimic, X_eicu], axis=0)
    ids_atlas = np.concatenate([ids_mimic, ids_eicu], axis=0)
    
    print(f"       - ATLAS SHAPE: {X_atlas.shape}")
    
    # 4. Merge Cohort Metadata
    print("    -> Merging Cohort Metadata...")
    df_mimic = pd.read_parquet(MIMIC_COHORT)
    df_mimic["source_db"] = "MIMIC-IV"
    
    df_eicu = pd.read_parquet(EICU_COHORT)
    df_eicu["source_db"] = "eICU"
    
    # Ensure shared columns for the merge
    shared_cols = ["stay_id", "hospital_expire_flag", "age", "gender", "baseline_sofa", "charlson_comorbidity_index", "source_db"]
    df_mimic_sub = df_mimic[[c for c in shared_cols if c in df_mimic.columns]]
    df_eicu_sub = df_eicu[[c for c in shared_cols if c in df_eicu.columns]]
    
    df_atlas = pd.concat([df_mimic_sub, df_eicu_sub], axis=0, ignore_index=True)
    
    # Verify alignment
    assert len(df_atlas) == len(ids_atlas), "Mismatch between Tensor patients and Cohort patients!"
    
    for col in ["age", "baseline_sofa", "charlson_comorbidity_index"]:
        if col in df_atlas.columns:
            df_atlas[col] = pd.to_numeric(df_atlas[col], errors="coerce").astype(float)
            
    # 5. Save Atlas
    print(f"    -> Saving Raw Atlas Tensor to {ATLAS_DIR.relative_to(BASE_DIR)}...")
    np.save(ATLAS_TENSOR_OUT, X_atlas)
    np.save(ATLAS_IDS_OUT, ids_atlas)
    df_atlas.to_parquet(ATLAS_COHORT_OUT, index=False)
    
    atlas_mortality = df_atlas["hospital_expire_flag"].mean() * 100
    print("\n============================================================")
    print(" MULTI-CENTER ATLAS SUCCESSFULLY BUILT (RAW)")
    print("============================================================")
    print(f" Total Patients : {len(df_atlas):,}")
    print(f" Total Features : {X_atlas.shape[2]}")
    print(f" Time Steps     : {X_atlas.shape[1]} Hours")
    print(f" Atlas Mortality: {atlas_mortality:.1f}%")
    print(f" MIMIC Cohort   : {len(df_mimic):,}")
    print(f" eICU Cohort    : {len(df_eicu):,}")
    print("============================================================")

if __name__ == "__main__":
    main()