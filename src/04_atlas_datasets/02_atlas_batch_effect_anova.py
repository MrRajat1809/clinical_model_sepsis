"""
02_atlas_batch_effect_anova.py

Statistical Batch Effect Validation Module (Clinical Atlas).
Runs a comparative one-way ANOVA on the Raw tensor (Globally Standardized),
the Raw tensor (Within-Cohort Standardized), and the ComBat-corrected tensor
to mathematically evaluate technical batch effects and biological signal preservation.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
ATLAS_DIR = BASE_DIR / "data" / "processed" / "atlas"

ATLAS_TENSOR_RAW = ATLAS_DIR / "atlas_imputed_tensor.npy"
ATLAS_TENSOR_COMBAT = ATLAS_DIR / "atlas_combat_tensor.npy"
ATLAS_COHORT = ATLAS_DIR / "atlas_final_cohort.parquet"

def evaluate_tensor(tensor_path, df_cohort, name, scaling_method='global'):
    print(f"\n{'='*70}")
    print(f" EVALUATING: {name}")
    print(f"{'='*70}")
    
    # 1. Load & Flatten
    X_3d = np.load(tensor_path)
    
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
    
    # 1.5 Apply Scaling Method
    if scaling_method == 'within_cohort':
        X_scaled = np.zeros_like(X_fused, dtype=float)
        # Apply StandardScaler independently to each hospital cohort
        for cohort_name in df_cohort['source_db'].unique():
            idx = df_cohort['source_db'] == cohort_name
            X_scaled[idx] = StandardScaler().fit_transform(X_fused[idx])
    else:
        # Standardize the entire merged dataset globally
        X_scaled = StandardScaler().fit_transform(X_fused)

    # 2. Compute PCA
    pca = PCA(n_components=3, random_state=42)
    pca_results = pca.fit_transform(X_scaled)
    
    df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2', 'PC3'])
    df['Hospital_System'] = df_cohort['source_db']
    df['Mortality'] = df_cohort['hospital_expire_flag']

    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f" -> PCA computed. Top 3 components explain {explained_var:.1f}% of variance.\n")

    # 3. ANOVA: Technical Batch Effect
    cohorts = df['Hospital_System'].unique()
    print("-" * 65)
    print(" ANALYSIS I: TECHNICAL BATCH EFFECT (HOSPITAL SYSTEM)")
    print(" Target: Fail to reject H0 (p > 0.05) -> Variance neutralized.")
    print("-" * 65)
    
    for pc in ['PC1', 'PC2', 'PC3']:
        cohort_groups = [df[df['Hospital_System'] == c][pc].values for c in cohorts]
        stat, p_val = f_oneway(*cohort_groups)
        status = "Neutralized (p > 0.05)" if p_val > 0.05 else "Artifact Detected (p <= 0.05)"
        print(f"  [{pc}] F-statistic: {stat:>8.3f} | p-value: {p_val:>8.4e} -> {status}")

    # 4. ANOVA: Biological Signal
    print("\n" + "-" * 65)
    print(" ANALYSIS II: BIOLOGICAL SIGNAL (MORTALITY)")
    print(" Target: Reject H0 (p < 0.05) -> Biological signal preserved.")
    print("-" * 65)
    
    for pc in ['PC1', 'PC2', 'PC3']:
        survivors = df[df['Mortality'] == 0][pc].values
        nonsurvivors = df[df['Mortality'] == 1][pc].values
        stat, p_val = f_oneway(survivors, nonsurvivors)
        status = "Preserved (p < 0.05)" if p_val < 0.05 else "Attenuated (p >= 0.05)"
        print(f"  [{pc}] F-statistic: {stat:>8.3f} | p-value: {p_val:>8.4e} -> {status}")

def main():
    print("[*] Initiating clinical statistical batch effect validation (ANOVA)...")
    
    if not ATLAS_TENSOR_RAW.exists() or not ATLAS_COHORT.exists():
        print("[ERROR] Required Atlas files not found.")
        return

    df_cohort = pd.read_parquet(ATLAS_COHORT)
    
    # Run comparison
    evaluate_tensor(ATLAS_TENSOR_RAW, df_cohort, "RAW ATLAS (Globally Standardized)", scaling_method='global')
    evaluate_tensor(ATLAS_TENSOR_RAW, df_cohort, "RAW ATLAS (Within-Cohort Standardized)", scaling_method='within_cohort')
    
    if ATLAS_TENSOR_COMBAT.exists():
        evaluate_tensor(ATLAS_TENSOR_COMBAT, df_cohort, "COMBAT ATLAS (Empirical Bayes)", scaling_method='global')
    
    print("\n[*] Statistical validation complete.")

if __name__ == "__main__":
    main()