"""
03_atlas_batch_effect_pca.py

Generates an "Atlas Matrix" for visualization purposes only.
1. Loads pre-imputed tensors for both MIMIC-IV and eICU.
2. Extracts the 124-feature representation (Static + Temporal Aggregations).
3. Merges the cohorts and applies Principal Component Analysis (PCA).
4. Plots the cohorts to visualize batch effects and physiological overlap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# eICU Data
EICU_TENSOR = BASE_DIR / "data" / "processed" / "eicu" / "tensors" / "eicu_sepsis_imputed_tensor.npy"
EICU_IDS = BASE_DIR / "data" / "processed" / "eicu" / "tensors" / "eicu_sepsis_tensor_stay_ids.npy"
EICU_COHORT = BASE_DIR / "data" / "processed" / "eicu" / "eicu_final_sepsis3_cohort.parquet"

OUT_PLOT = BASE_DIR / "outputs" / "visualization" / "atlas_pca_batch_effects.png"
OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

def extract_features(tensor_path, ids_path, cohort_path):
    """Loads 3D tensor and flattens it into the 124-feature XGBoost format."""
    X_3d = np.load(tensor_path)
    ids = np.load(ids_path)
    df_cohort = pd.read_parquet(cohort_path)
    
    # Align clinical data
    df_aligned = pd.DataFrame({"stay_id": ids}).merge(df_cohort, on="stay_id", how="left")
    
    # 1. Static Features
    statics = ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"]
    df_static = df_aligned[statics].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"].astype(str).str.upper() == "M").astype(int)
    X_static = df_static.fillna(0).values
    
    # 2. Temporal Features
    X_mean = np.mean(X_3d, axis=1)
    X_min = np.min(X_3d, axis=1)
    X_max = np.max(X_3d, axis=1)
    X_std = np.std(X_3d, axis=1)
    X_temporal = np.concatenate([X_mean, X_min, X_max, X_std], axis=1)
    
    # 3. Fuse
    X_fused = np.concatenate([X_static, X_temporal], axis=1)
    return X_fused

def main():
    print("[*] Building the Merged Atlas Matrix for Visualization...")
    
    # 1. Extract standard feature sets
    print("    -> Extracting MIMIC-IV features...")
    X_mimic = extract_features(MIMIC_TENSOR, MIMIC_IDS, MIMIC_COHORT)
    
    print("    -> Extracting eICU features...")
    X_eicu = extract_features(EICU_TENSOR, EICU_IDS, EICU_COHORT)
    
    # 2. Merge into Atlas
    print("    -> Merging into global Atlas and standardizing...")
    X_atlas = np.vstack([X_mimic, X_eicu])
    y_hospital = np.array(["MIMIC-IV (Beth Israel)"] * len(X_mimic) + ["eICU (Multi-Center)"] * len(X_eicu))
    
    # Standardize across the entire merged atlas for visualization
    X_atlas_scaled = StandardScaler().fit_transform(X_atlas)
    
    # 3. Apply PCA
    print("    -> Running Principal Component Analysis (PCA)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_atlas_scaled)
    
    var_explained = pca.explained_variance_ratio_ * 100
    
    # 4. Plot
    print(f"    -> Saving Batch Effect Visualization to {OUT_PLOT.relative_to(BASE_DIR)}")
    plt.figure(figsize=(10, 8))
    
    # Plot eICU first (background) then MIMIC (foreground) or vice versa
    mask_mimic = y_hospital == "MIMIC-IV (Beth Israel)"
    mask_eicu = y_hospital == "eICU (Multi-Center)"
    
    plt.scatter(X_pca[mask_eicu, 0], X_pca[mask_eicu, 1], alpha=0.4, s=15, color="coral", label="eICU (Multi-Center)", edgecolors='none')
    plt.scatter(X_pca[mask_mimic, 0], X_pca[mask_mimic, 1], alpha=0.4, s=15, color="steelblue", label="MIMIC-IV (Beth Israel)", edgecolors='none')
    
    plt.title("Atlas Matrix: Cross-Cohort Physiological Distribution\nVisualizing Hospital-Level Batch Effects", fontsize=14)
    plt.xlabel(f"Principal Component 1 ({var_explained[0]:.1f}% Variance)", fontsize=12)
    plt.ylabel(f"Principal Component 2 ({var_explained[1]:.1f}% Variance)", fontsize=12)
    
    plt.legend(loc="upper right", markerscale=3, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    print("[*] Visualization complete.")

if __name__ == "__main__":
    main()