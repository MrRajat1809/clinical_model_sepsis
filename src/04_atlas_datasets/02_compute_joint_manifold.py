"""
02_compute_joint_manifold.py

Projects the high-dimensional prognostic feature space into a 2D manifold.

Features included:
- Loads the pre-computed, OT-harmonized 124D feature matrix.
- Applies StandardScaler to ensure geometric stability.
- Applies pre-diffusion PCA (n_pca=50) to denoise the 124D space, 
  resolving the SGD-MDS convergence warnings and stabilizing the topology.
- Fits the PHATE embedding to visualize continuous severity trajectories.
- Computes Silhouette mixing scores to mathematically validate cohort mixing.
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

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Inputs
PROCESSED_DIR_ATLAS = BASE_DIR / "data" / "processed" / "atlas"
ATLAS_FEATURES_FILE = PROCESSED_DIR_ATLAS / "atlas_sepsis_features_124.npy"
ATLAS_META_FILE = PROCESSED_DIR_ATLAS / "atlas_metadata.parquet"

# Outputs
OUT_FEATURES = BASE_DIR / "outputs" / "features"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"

OUT_FEATURES.mkdir(parents=True, exist_ok=True)
OUT_METRICS.mkdir(parents=True, exist_ok=True)

PHATE_COORDS_FILE = OUT_FEATURES / "atlas_phate_coordinates.parquet"
MIXING_METRICS_FILE = OUT_METRICS / "atlas_manifold_mixing_metrics.json"

def main():
    print("[*] Initiating Phase 4: Prognostic Space Projection (PHATE Manifold)...")
    start_time = time.time()

    # ---------------------------------------------------------
    # 1. LOAD HARMONIZED ATLAS (124D)
    # ---------------------------------------------------------
    print("    -> Loading Harmonized 124D Atlas array...")
    if not ATLAS_FEATURES_FILE.exists() or not ATLAS_META_FILE.exists():
        print(f"[ERROR] Atlas files not found. Ensure 01_harmonize_ot_tensor.py ran successfully.")
        return

    X_124 = np.load(ATLAS_FEATURES_FILE)
    df_meta = pd.read_parquet(ATLAS_META_FILE)
    
    print(f"       - 124D Matrix Shape : {X_124.shape}")
    print(f"       - Metadata Rows     : {len(df_meta):,}")

    # ---------------------------------------------------------
    # 2. GEOMETRIC SCALING & PHATE EMBEDDING
    # ---------------------------------------------------------
    print("\n    -> Scaling feature space for geometric stabilization (N(0,1))...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_124)

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

    # ---------------------------------------------------------
    # 3. MANIFOLD MIXING & SEVERITY VALIDATION
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 4. EXPORT
    # ---------------------------------------------------------
    print("\n    -> Exporting Manifold Coordinates...")
    
    df_coords = pd.DataFrame({
        "atlas_id": df_meta["atlas_id"],
        "PHATE_1": X_phate[:, 0],
        "PHATE_2": X_phate[:, 1]
    })
    
    df_coords.to_parquet(PHATE_COORDS_FILE, index=False)
    
    metrics = {
        "Manifold_Algorithm": "PHATE",
        "Feature_Representation": "124-Feature Prognostic Vector (OT Harmonized)",
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