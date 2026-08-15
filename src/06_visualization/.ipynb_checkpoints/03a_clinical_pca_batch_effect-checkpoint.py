"""
03a_clinical_pca_batch_effect.py

Generates Panel A: Comparative PCA Visualization (Clinical Atlas).
Visualizes the integrated 124-feature clinical expression tensor to 
demonstrate the baseline shift (batch effect) across the independent 
MIMIC-IV and eICU hospital systems.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
ATLAS_DIR = BASE_DIR / "data" / "processed" / "atlas"
FIG_OUT = BASE_DIR / "outputs" / "figures"

ATLAS_TENSOR = ATLAS_DIR / "atlas_imputed_tensor.npy"
ATLAS_COHORT = ATLAS_DIR / "atlas_final_cohort.parquet"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Generating Figure 2 Panel A: Clinical Batch Effect PCA...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD & FLATTEN ATLAS TENSOR
    # ---------------------------------------------------------
    print("    -> Loading 3D Atlas tensor and metadata...")
    X_3d = np.load(ATLAS_TENSOR)
    df_cohort = pd.read_parquet(ATLAS_COHORT)

    print("    -> Flattening temporal dimensions (Mean, Min, Max, Std)...")
    statics = ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"]
    df_static = df_cohort[statics].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"].astype(str).str.upper() == "M").astype(int)
    X_static = df_static.fillna(0).values

    X_mean = np.mean(X_3d, axis=1)
    X_min = np.min(X_3d, axis=1)
    X_max = np.max(X_3d, axis=1)
    X_std = np.std(X_3d, axis=1)
    
    X_fused = np.concatenate([X_static, X_mean, X_min, X_max, X_std], axis=1)
    
    print("    -> Standardizing features...")
    X_scaled = StandardScaler().fit_transform(X_fused)

    # ---------------------------------------------------------
    # 2. COMPUTE PRINCIPAL COMPONENTS
    # ---------------------------------------------------------
    print("    -> Computing Principal Components...")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(X_scaled)

    df_meta = pd.DataFrame({
        'PC1': pca_results[:, 0],
        'PC2': pca_results[:, 1],
        'Hospital_System': df_cohort['source_db']
    })

    var_explained = pca.explained_variance_ratio_ * 100

    # ---------------------------------------------------------
    # 3. RENDER VISUALIZATION
    # ---------------------------------------------------------
    print("    -> Generating PCA visualization...")
    sns.set_theme(style="white")
    
    plt.figure(figsize=(9, 7))
    palette = {"MIMIC-IV": "steelblue", "eICU": "coral"}

    sns.scatterplot(
        data=df_meta, x='PC1', y='PC2', hue='Hospital_System', 
        alpha=0.5, s=20, palette=palette, edgecolor=None
    )
    
    plt.title('A. Baseline Variance (124-Feature Clinical Space)', fontsize=14)
    plt.xlabel(f'Principal Component 1 ({var_explained[0]:.1f}% Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({var_explained[1]:.1f}% Variance)', fontsize=12)
    plt.legend(title='Database Cohort', loc='upper right', frameon=True, shadow=True)
    sns.despine()

    plt.tight_layout()
    
    # ---------------------------------------------------------
    # 4. EXPORT
    # ---------------------------------------------------------
    out_path = FIG_OUT / "Fig2A_Clinical_Batch_Effect.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] SUCCESS! Panel A saved to: {out_path.name}")

if __name__ == "__main__":
    main()