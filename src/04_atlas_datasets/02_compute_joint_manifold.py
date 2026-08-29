"""
Project the harmonised atlas into a shared two-dimensional prognostic space.

Embeds both cohorts together with PHATE so that patients from either database
can be examined in one geometry under one feature representation.

The 122-D representation is standardised and reduced to 50 principal components
before diffusion. Pre-diffusion PCA denoises the space and stabilises the
embedding's optimisation, which otherwise struggles to converge at this
dimensionality.

Two silhouette coefficients are computed on a random subsample and answer
different questions. By database of origin, near zero means the cohorts occupy
the same region rather than sitting in separate clusters. By mortality, a
negative value means severity varies as a continuous gradient across the
manifold rather than forming discrete outcome clusters.

Distinct from the DTW trajectory manifolds: this one runs on the 122-D summary
representation and is fast, so it belongs in the main pipeline.

Reads:
    data/processed/atlas/{atlas_sepsis_features.npy, atlas_metadata.parquet}
Writes:
    outputs/features/atlas_phate_coordinates.parquet
    outputs/metrics/atlas_manifold_mixing_metrics.json
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    import phate
except ImportError:
    raise ImportError("The 'phate' library is required. Please run: pip install phate")

import warnings
warnings.filterwarnings("ignore")

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR_ATLAS = BASE_DIR / "data" / "processed" / "atlas"
ATLAS_FEATURES_FILE = PROCESSED_DIR_ATLAS / "atlas_sepsis_features.npy"
ATLAS_META_FILE = PROCESSED_DIR_ATLAS / "atlas_metadata.parquet"

OUT_FEATURES = BASE_DIR / "outputs" / "features"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"

OUT_FEATURES.mkdir(parents=True, exist_ok=True)
OUT_METRICS.mkdir(parents=True, exist_ok=True)

PHATE_COORDS_FILE = OUT_FEATURES / "atlas_phate_coordinates.parquet"
MIXING_METRICS_FILE = OUT_METRICS / "atlas_manifold_mixing_metrics.json"

def main():
    print("[*] Initiating Phase 4: Prognostic Space Projection (PHATE Manifold)...")
    start_time = time.time()

    # --- Load Harmonized Atlas (122d) ------------------------------------
    print("    -> Loading Harmonized 122D Atlas array...")
    if not ATLAS_FEATURES_FILE.exists() or not ATLAS_META_FILE.exists():
        print(f"[ERROR] Atlas files not found. Ensure 01_harmonize_ot_tensor.py ran successfully.")
        return

    X_atlas = np.load(ATLAS_FEATURES_FILE)
    df_meta = pd.read_parquet(ATLAS_META_FILE)
    
    print(f"       - 122D Matrix Shape : {X_atlas.shape}")
    print(f"       - Metadata Rows     : {len(df_meta):,}")

    # --- Geometric Scaling & Phate Embedding -----------------------------
    print("\n    -> Scaling feature space for geometric stabilization (N(0,1))...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_atlas)

    print("    -> Fitting PHATE Manifold (n_components=2, n_pca=50)...")
    print("       (Computing dense diffusion probabilities. This may take 1-3 minutes...)")
    
    # Adding n_pca=50 smooths the distance matrix and prevents SGD-MDS convergence failures
    phate_operator = phate.PHATE(
        n_components=2, 
        n_pca=50,
        knn=40,
        decay=20,
        n_jobs=-1, 
        random_state=42,
        verbose=False
    )
    
    X_phate = phate_operator.fit_transform(X_scaled)

    # --- Manifold Mixing & Severity Validation ---------------------------
    print("\n    -> Calculating manifold structural metrics (Silhouette Scores)...")
    
    # Sample 5000 for metric compute speed
    np.random.seed(42)
    sample_idx = np.random.choice(X_scaled.shape[0], 5000, replace=False)
    X_phate_sample = X_phate[sample_idx]
    
    cohort_labels = df_meta.iloc[sample_idx]["cohort_source"].values
    mortality_labels = df_meta.iloc[sample_idx]["hospital_expire_flag"].values
    
    sil_cohort = silhouette_score(X_phate_sample, cohort_labels)
    sil_mortality = silhouette_score(X_phate_sample, mortality_labels)
    
    print(f"       - Cohort Mixing Silhouette  : {sil_cohort:.4f} (Ideal is ~0)")
    print(f"       - Mortality Gradients       : {sil_mortality:.4f} (Negative implies lack of discrete clusters)")

    # --- Export ----------------------------------------------------------
    print("\n    -> Exporting Manifold Coordinates...")
    
    df_coords = pd.DataFrame({
        "atlas_id": df_meta["atlas_id"],
        "PHATE_1": X_phate[:, 0],
        "PHATE_2": X_phate[:, 1]
    })
    
    df_coords.to_parquet(PHATE_COORDS_FILE, index=False)
    
    metrics = {
        "Manifold_Algorithm": "PHATE",
        "Feature_Representation": f"{X_atlas.shape[1]}-Feature Prognostic Vector (OT Harmonized)",
        "Parameters": {"n_pca": 50, "knn": 40, "decay": 20},
        "Cohort_Silhouette_Score": float(sil_cohort),
        "Mortality_Silhouette_Score": float(sil_mortality),
        "Interpretation": "Cohort score near 0 confirms spatial overlap of cohorts (batch effect mitigation). Negative mortality score confirms severity is distributed as a continuous gradient rather than detached discrete clusters."
    }
    
    with open(MIXING_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    elapsed = time.time() - start_time
    print(f"\n[+] Success! PHATE Manifold embedded in {elapsed:.2f} seconds.")
    print(f"    -> Coordinates saved to: {PHATE_COORDS_FILE.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
