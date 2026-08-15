"""
03b_clinical_pca_biology_preservation.py

Generates Panel B: Biological Signal vs. Technical Shift.
Calculates the Silhouette Score to mathematically quantify dataset shift, 
and generates a dual-panel PCA figure to contrast the technical hospital 
batch effect against the biological mortality signal.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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
    print("[*] Generating Figure 2 Panel B: Quality Control & Signal Diagnostics...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD & FLATTEN ATLAS TENSOR
    # ---------------------------------------------------------
    print("    -> Loading and standardizing multi-center tensors...")
    X_3d = np.load(ATLAS_TENSOR)
    df_cohort = pd.read_parquet(ATLAS_COHORT)

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
    X_scaled = StandardScaler().fit_transform(X_fused)

    # ---------------------------------------------------------
    # 2. COMPUTE PRINCIPAL COMPONENTS
    # ---------------------------------------------------------
    print("    -> Calculating Principal Components...")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame({
        'PC1': pca_results[:, 0],
        'PC2': pca_results[:, 1],
        'Hospital_System': df_cohort['source_db'],
        'Mortality': df_cohort['hospital_expire_flag'].map({0: 'Survivor', 1: 'Non-Survivor'})
    })

    # ---------------------------------------------------------
    # 3. MATHEMATICAL BATCH MIXING QUANTIFICATION
    # ---------------------------------------------------------
    print("    -> Computing Silhouette Score for dataset shift...")
    # Subsampling for silhouette score speed if dataset is massive
    sub_idx = np.random.choice(len(pca_results), min(len(pca_results), 10000), replace=False)
    sil_score = silhouette_score(pca_results[sub_idx], pca_df['Hospital_System'].iloc[sub_idx])
    print(f"       [METRIC] Global Silhouette Score: {sil_score:.4f}")

    # ---------------------------------------------------------
    # 4. RENDER VISUALIZATION
    # ---------------------------------------------------------
    print("    -> Rendering PCA grids...")
    sns.set_theme(style="white")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel C (Left): Cohort Shift & Silhouette Annotation
    cohort_palette = {"MIMIC-IV": "steelblue", "eICU": "coral"}
    sns.scatterplot(
        x='PC1', y='PC2', hue='Hospital_System', data=pca_df, 
        palette=cohort_palette, alpha=0.5, s=20, edgecolor=None, ax=axes[0]
    )
    
    axes[0].set_title('C. Technical Shift by Hospital System', fontsize=14)
    axes[0].set_xlabel('Principal Component 1', fontsize=12)
    axes[0].set_ylabel('Principal Component 2', fontsize=12)
    axes[0].legend(title='Database Cohort', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(ax=axes[0])
    
    axes[0].text(
        0.05, 0.95, f'Batch Silhouette Score: {sil_score:.3f}', 
        transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='square,pad=0.6', facecolor='#f9f9f9', edgecolor='#dddddd', alpha=0.9)
    )

    # Panel D (Right): Biological Signal Preservation
    sns.scatterplot(
        x='PC1', y='PC2', hue='Mortality', data=pca_df, 
        palette={'Survivor': '#4a6fe3', 'Non-Survivor': '#db4325'}, 
        alpha=0.5, s=20, edgecolor=None, ax=axes[1]
    )
    
    axes[1].set_title('D. Covariate Retention Check (Sepsis Mortality)', fontsize=14)
    axes[1].set_xlabel('Principal Component 1', fontsize=12)
    axes[1].set_ylabel('Principal Component 2', fontsize=12)
    axes[1].legend(title='Clinical Outcome', loc='upper right', frameon=True, shadow=True)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    
    # ---------------------------------------------------------
    # 5. EXPORT
    # ---------------------------------------------------------
    out_path = FIG_OUT / "Fig2B_Clinical_Biology_Preservation.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SUCCESS! Panel B saved to: {out_path.name}")

if __name__ == "__main__":
    main()