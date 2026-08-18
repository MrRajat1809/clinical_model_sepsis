"""
01b_verify_atlas_harmonization.py

Runs rigorous Quality Control (QC) on the harmonized Sepsis Atlas before PHATE manifold projection.
Verifies:
1. No NaN/Inf corruption in the 124D feature matrix.
2. Perfect shape and ID alignment across tensor, metadata, and ID arrays.
3. Distribution and variance preservation (checking Mean HR and Mean MAP for implausibilities).
4. Cohort mixing vs. clinical structure (PCA + Silhouette scores).
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Inputs
PROCESSED_DIR_ATLAS = BASE_DIR / "data" / "processed" / "atlas"
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"

# Outputs
OUT_FIGURES = BASE_DIR / "outputs" / "figures"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

QC_PLOT_FILE = OUT_FIGURES / "atlas_QC_Harmonization_Checks.png"

def main():
    print("[*] Initiating Atlas Quality Control & Harmonization Verification (124D)...")
    start_time = time.time()

    # ---------------------------------------------------------
    # 0. LOAD DATA
    # ---------------------------------------------------------
    print("    -> Loading 124D Atlas artifacts and pre-OT eICU tensor for comparison...")
    X_atlas_124 = np.load(PROCESSED_DIR_ATLAS / "atlas_sepsis_features_124.npy")
    ids_atlas = np.load(PROCESSED_DIR_ATLAS / "atlas_stay_ids.npy", allow_pickle=True)
    df_meta = pd.read_parquet(PROCESSED_DIR_ATLAS / "atlas_metadata.parquet")
    
    # Load feature names (using MIMIC as the reference)
    features = list(np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_features.npy", allow_pickle=True))
    
    # Load original pre-OT eICU tensor to calculate its raw mean for comparison
    X_eicu_raw_3d = np.load(PROCESSED_DIR_EICU / "eicu_sepsis_imputed_tensor.npy")
    X_eicu_raw_mean = np.mean(X_eicu_raw_3d, axis=1)

    # ---------------------------------------------------------
    # CHECK 1: NaN and Infinity Check
    # ---------------------------------------------------------
    print("\n[QC CHECK 1]: Array Integrity (NaN / Inf)")
    nan_count = np.isnan(X_atlas_124).sum()
    inf_count = np.isinf(X_atlas_124).sum()
    
    if nan_count == 0 and inf_count == 0:
        print("    [PASS] 124D Matrix is perfectly clean (0 NaNs, 0 Infs).")
    else:
        print(f"    [FAIL] Matrix corruption detected! NaNs: {nan_count:,}, Infs: {inf_count:,}")

    # ---------------------------------------------------------
    # CHECK 2: Shape and ID Alignment
    # ---------------------------------------------------------
    print("\n[QC CHECK 2]: Dimensionality and ID Alignment")
    n_tensor = X_atlas_124.shape[0]
    n_ids = len(ids_atlas)
    n_meta = len(df_meta)
    
    print(f"    - Matrix Rows:   {n_tensor:,}")
    print(f"    - ID Array Rows: {n_ids:,}")
    print(f"    - Metadata Rows: {n_meta:,}")
    
    if n_tensor == n_ids == n_meta:
        print("    [PASS] Dimensions align perfectly.")
    else:
        print("    [FAIL] Row counts do not match!")

    if (df_meta["atlas_id"].values == ids_atlas).all():
        print("    [PASS] Metadata exactly matches Tensor ID sequence.")
    else:
        print("    [FAIL] Metadata order does not match Tensor IDs!")

    # ---------------------------------------------------------
    # CHECK 3: Distribution & Variance Preservation
    # ---------------------------------------------------------
    print("\n[QC CHECK 3]: Biological Variance Preservation")
    # In the 124D array, the first 30 columns are the MEANS of the temporal variables.
    hr_idx = features.index("hr")
    map_idx = features.index("map")
    
    eicu_mask = df_meta["cohort_source"] == "eICU-CRD"
    X_eicu_ot_124 = X_atlas_124[eicu_mask]
    
    # Compute bounds for Mean HR and Mean MAP
    hr_min_pre, hr_max_pre = np.min(X_eicu_raw_mean[:, hr_idx]), np.max(X_eicu_raw_mean[:, hr_idx])
    hr_min_post, hr_max_post = np.min(X_eicu_ot_124[:, hr_idx]), np.max(X_eicu_ot_124[:, hr_idx])
    
    map_min_pre, map_max_pre = np.min(X_eicu_raw_mean[:, map_idx]), np.max(X_eicu_raw_mean[:, map_idx])
    map_min_post, map_max_post = np.min(X_eicu_ot_124[:, map_idx]), np.max(X_eicu_ot_124[:, map_idx])

    print("    - eICU Mean Heart Rate Bounds:")
    print(f"        Pre-OT:  [{hr_min_pre:.1f}, {hr_max_pre:.1f}]")
    print(f"        Post-OT: [{hr_min_post:.1f}, {hr_max_post:.1f}]")
    
    print("    - eICU Mean MAP Bounds:")
    print(f"        Pre-OT:  [{map_min_pre:.1f}, {map_max_pre:.1f}]")
    print(f"        Post-OT: [{map_min_post:.1f}, {map_max_post:.1f}]")
    
    if hr_min_post >= 0 and map_min_post >= 0:
        print("    [PASS] OT transformation produced no physiologically impossible negative vitals.")
    else:
        print("    [WARNING] Negative physiological values detected post-OT!")

    # ---------------------------------------------------------
    # CHECK 4: Cohort Mixing vs Clinical Structure (PCA)
    # ---------------------------------------------------------
    print("\n[QC CHECK 4]: Cohort Mixing vs. Severity Preservation")
    print("    -> Running fast PCA on 124D Atlas to test linear manifold structure...")
    
    # Random subsample of 5000 patients for fast Silhouette calculation
    np.random.seed(42)
    sample_idx = np.random.choice(X_atlas_124.shape[0], 5000, replace=False)
    X_sample = X_atlas_124[sample_idx]
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)
    
    cohort_labels = df_meta.iloc[sample_idx]["cohort_source"].values
    mortality_labels = df_meta.iloc[sample_idx]["hospital_expire_flag"].values
    
    sil_cohort = silhouette_score(X_pca, cohort_labels)
    sil_mortality = silhouette_score(X_pca, mortality_labels)
    
    print(f"    - Silhouette Score (Cohort Source) : {sil_cohort:.4f} (Closer to 0 is better mixing)")
    print(f"    - Silhouette Score (Mortality)     : {sil_mortality:.4f} (Negative/Low implies continuous gradient)")
    
    if sil_cohort < 0.05:
        print("    [PASS] Cohorts are highly mixed. Batch effect is structurally mitigated.")
    else:
        print("    [WARNING] Cohorts may still contain residual geometric separation.")

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    print(f"\n    -> Generating 4-Panel QC Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: HR Distribution (Pre vs Post)
    sns.kdeplot(X_eicu_raw_mean[:, hr_idx], label="eICU (Pre-OT)", ax=axes[0, 0], color="grey", linestyle="--")
    sns.kdeplot(X_eicu_ot_124[:, hr_idx], label="eICU (Post-OT)", ax=axes[0, 0], color="#C44E52")
    axes[0, 0].set_title("A) Mean Heart Rate Geometry Preservation")
    axes[0, 0].legend()
    
    # Panel B: MAP Distribution (Pre vs Post)
    sns.kdeplot(X_eicu_raw_mean[:, map_idx], label="eICU (Pre-OT)", ax=axes[0, 1], color="grey", linestyle="--")
    sns.kdeplot(X_eicu_ot_124[:, map_idx], label="eICU (Post-OT)", ax=axes[0, 1], color="#C44E52")
    axes[0, 1].set_title("B) Mean MAP Geometry Preservation")
    axes[0, 1].legend()

    # Panel C: Cohort Mixing
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cohort_labels, alpha=0.5, s=15, ax=axes[1, 0], palette=["#4C72B0", "#C44E52"])
    axes[1, 0].set_title(f"C) Cohort Mixing (Silhouette: {sil_cohort:.3f})")
    
    # Panel D: Mortality Structure
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=mortality_labels, alpha=0.5, s=15, ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title(f"D) Severity Structure (Mortality Silhouette: {sil_mortality:.3f})")
    
    plt.tight_layout()
    plt.savefig(QC_PLOT_FILE, dpi=300)
    plt.close()
    
    elapsed = time.time() - start_time
    print(f"\n[+] QC complete in {elapsed:.2f} seconds.")
    print(f"    -> Plot saved to: {QC_PLOT_FILE.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()