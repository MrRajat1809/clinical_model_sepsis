"""
01b_verify_atlas_harmonization.py

Runs rigorous Quality Control (QC) on the harmonized Sepsis Atlas.
Addresses advanced methodological critiques by verifying:
1. Array Integrity (NaN/Inf).
2. Dimensionality and ID Alignment.
3. Biological Variance & Univariate Prognostic Signal across ALL 124 features.
4. Cohort mixing (PCA + Silhouette).
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR_ATLAS = BASE_DIR / "data" / "processed" / "atlas"
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"

OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"
OUT_METRICS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

QC_PLOT_FILE = OUT_FIGURES / "atlas_QC_Harmonization_Checks.png"
UNIVARIATE_CSV_FILE = OUT_METRICS / "atlas_univariate_auroc_shifts.csv"

def get_feature_index(feature_names, search_term):
    for i, name in enumerate(feature_names):
        if search_term.lower() in name.lower():
            return i
    return None

def main():
    print("[*] Initiating Advanced Atlas Quality Control (124D)...")
    start_time = time.time()

    # ---------------------------------------------------------
    # 0. LOAD DATA
    # ---------------------------------------------------------
    print("    -> Loading 124D Atlas artifacts and pre-OT eICU tensors...")
    X_atlas_124 = np.load(PROCESSED_DIR_ATLAS / "atlas_sepsis_features_124.npy")
    ids_atlas = np.load(PROCESSED_DIR_ATLAS / "atlas_stay_ids.npy", allow_pickle=True)
    df_meta = pd.read_parquet(PROCESSED_DIR_ATLAS / "atlas_metadata.parquet")
    
    features = list(np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_features.npy", allow_pickle=True))
    
    X_eicu_raw_3d = np.load(PROCESSED_DIR_EICU / "eicu_sepsis_imputed_tensor.npy")
    X_eicu_raw_static = np.load(PROCESSED_DIR_EICU / "eicu_sepsis_tensor_static.npy", allow_pickle=True)
    
    eicu_mask = df_meta["cohort_source"] == "eICU-CRD"
    X_eicu_ot_124 = X_atlas_124[eicu_mask]
    y_eicu = df_meta[eicu_mask]["hospital_expire_flag"].values

    # ---------------------------------------------------------
    # CHECK 1 & 2: Integrity and Alignment
    # ---------------------------------------------------------
    print("\n[QC CHECK 1 & 2]: Array Integrity & Alignment")
    if np.isnan(X_atlas_124).sum() == 0 and np.isinf(X_atlas_124).sum() == 0:
        print("    [PASS] 124D Matrix is clean (0 NaNs, 0 Infs).")
    else:
        print("    [FAIL] Matrix corruption detected!")

    if X_atlas_124.shape[0] == len(ids_atlas) == len(df_meta):
        print("    [PASS] Dimensions align perfectly across arrays.")
    else:
        print("    [FAIL] Row counts do not match!")
    
    if (df_meta["atlas_id"].values == ids_atlas).all():
        print("    [PASS] Metadata exactly matches Tensor ID sequence.")
    else:
        print("    [FAIL] Metadata order does not match Tensor IDs!")

    # ---------------------------------------------------------
    # CHECK 3 & 4: Variance Retention & Univariate AUROC (ALL 124 FEATURES)
    # ---------------------------------------------------------
    print("\n[QC CHECK 3 & 4]: Prognostic Signal & Variance Preservation (All 124 Features)")
    
    # Reconstruct raw 124D representation for direct column-to-column comparison
    X_eicu_temporal_raw = np.concatenate([
        np.mean(X_eicu_raw_3d, axis=1), np.min(X_eicu_raw_3d, axis=1),
        np.max(X_eicu_raw_3d, axis=1), np.std(X_eicu_raw_3d, axis=1)
    ], axis=1)
    X_eicu_static_raw = X_eicu_raw_static[:, [0, 1, 5, 6]].astype(np.float32)
    X_eicu_124_raw = np.concatenate([X_eicu_temporal_raw, X_eicu_static_raw], axis=1)

    # Generate feature names for the 124 dimensions
    feature_names_124 = (
        [f"{f}_mean" for f in features] +
        [f"{f}_min" for f in features] +
        [f"{f}_max" for f in features] +
        [f"{f}_std" for f in features] +
        ["age", "gender", "cci", "baseline_sofa"]
    )

    results = []
    for i in range(124):
        raw_vals = X_eicu_124_raw[:, i]
        ot_vals = X_eicu_ot_124[:, i]
        
        # Calculate standard deviation ratio (handling division by zero)
        std_raw = np.std(raw_vals)
        std_ot = np.std(ot_vals)
        var_ratio = (std_ot / std_raw) if std_raw > 0 else 1.0
        
        # Calculate AUROC (using absolute correlation direction to handle inverted risks like GCS/MAP)
        # roc_auc_score requires directionality. We take the max of standard or inverted.
        try:
            auc_standard_raw = roc_auc_score(y_eicu, raw_vals)
            auc_invert_raw = roc_auc_score(y_eicu, -raw_vals)
            auc_raw = max(auc_standard_raw, auc_invert_raw)
            
            auc_standard_ot = roc_auc_score(y_eicu, ot_vals)
            auc_invert_ot = roc_auc_score(y_eicu, -ot_vals)
            auc_ot = max(auc_standard_ot, auc_invert_ot)
        except ValueError:
            auc_raw, auc_ot = 0.5, 0.5 # Fallback for flat arrays
            
        results.append({
            "Feature": feature_names_124[i],
            "Var_Ratio": var_ratio,
            "Pre_OT_AUROC": auc_raw,
            "Post_OT_AUROC": auc_ot,
            "AUROC_Diff": auc_ot - auc_raw
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(UNIVARIATE_CSV_FILE, index=False)
    
    avg_diff = df_results["AUROC_Diff"].mean()
    print(f"    -> Average AUROC shift across all 124 features: {avg_diff:+.4f}")
    
    print("\n    -> Top 5 Most Improved Features:")
    print(df_results.sort_values("AUROC_Diff", ascending=False).head(5)[["Feature", "Pre_OT_AUROC", "Post_OT_AUROC", "AUROC_Diff"]].to_string(index=False))

    # ---------------------------------------------------------
    # CHECK 5: Cohort Mixing vs Clinical Structure (PCA)
    # ---------------------------------------------------------
    print("\n[QC CHECK 5]: Geometric Cohort Mixing")
    np.random.seed(42)
    sample_idx = np.random.choice(X_atlas_124.shape[0], 5000, replace=False)
    X_sample = X_atlas_124[sample_idx]
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)
    
    cohort_labels = df_meta.iloc[sample_idx]["cohort_source"].values
    sil_cohort = silhouette_score(X_pca, cohort_labels)
    
    print(f"    - Silhouette Score (Cohort Source) : {sil_cohort:.4f} (Closer to 0 is perfect mixing)")
    if sil_cohort < 0.05:
        print("    [PASS] Cohorts are highly mixed. Batch effect is structurally mitigated.")

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    print(f"\n    -> Generating QC Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # We use Lactate and SOFA just for the visual panels
    idx_lactate = get_feature_index(features, "lactate")
    sns.kdeplot(X_eicu_124_raw[:, idx_lactate], label="eICU (Pre-OT)", ax=axes[0, 0], color="grey", linestyle="--")
    sns.kdeplot(X_eicu_ot_124[:, idx_lactate], label="eICU (Post-OT)", ax=axes[0, 0], color="#C44E52")
    axes[0, 0].set_title("A) Mean Lactate Geometry Preservation")
    axes[0, 0].legend()
    
    sns.kdeplot(X_eicu_124_raw[:, 123], label="eICU (Pre-OT)", ax=axes[0, 1], color="grey", linestyle="--")
    sns.kdeplot(X_eicu_ot_124[:, 123], label="eICU (Post-OT)", ax=axes[0, 1], color="#C44E52")
    axes[0, 1].set_title("B) Baseline SOFA Geometry Preservation")
    axes[0, 1].legend()

    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cohort_labels, alpha=0.5, s=15, ax=axes[1, 0], palette=["#4C72B0", "#C44E52"])
    axes[1, 0].set_title(f"C) Cohort Mixing (Silhouette: {sil_cohort:.3f})")
    
    mortality_labels = df_meta.iloc[sample_idx]["hospital_expire_flag"].values
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=mortality_labels, alpha=0.5, s=15, ax=axes[1, 1], palette="viridis")
    axes[1, 1].set_title("D) Severity Structure in PCA Space")
    
    plt.tight_layout()
    plt.savefig(QC_PLOT_FILE, dpi=300)
    plt.close()
    
    print(f"\n[+] QC complete in {time.time() - start_time:.2f} seconds.")
    print(f"    -> Full Univariate results saved to: {UNIVARIATE_CSV_FILE.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()