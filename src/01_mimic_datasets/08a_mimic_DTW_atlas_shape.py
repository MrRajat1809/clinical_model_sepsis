"""
08a_mimic_DTW_atlas_shape.py

Computes the pairwise Dynamic Time Warping (DTW) distance matrix for the fully imputed 
3D time-series tensor to map the patient trajectory manifold.

Features included:
- Uses TimeSeriesScalerMeanVariance(). Scaling each patient independently forces 
  the DTW algorithm to evaluate trajectories based on the *morphology* (shape) 
  of their physiological changes rather than absolute severity.
- Replaced discrete K-Means clustering with pairwise distance matrix computation 
  (cdist_dtw) to feed manifold learning algorithms (e.g., PHATE).
- Implemented smart upper-triangular batching wrapped in a tqdm progress bar 
  to track the O(N^2) computation without losing tslearn's self-similarity optimization.
- Filenames explicitly designated with `_shape_` to prevent clashes with the Severity manifold.
"""

import time
from pathlib import Path
import numpy as np
import joblib
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Flattened Data Inputs
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

# Flattened Global Outputs
OUT_FEATS = BASE_DIR / "outputs" / "features"
OUT_MODELS = BASE_DIR / "outputs" / "models"

def run_tensor_atlas():
    print("[*] Initializing Tensor Atlas & Pairwise DTW Distance Pipeline...")
    start_time = time.time()
    
    OUT_FEATS.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    
    tensor_file = PROCESSED_DIR / "mimic_sepsis_imputed_tensor.npy"
    id_file = PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy"
    
    # Atlas Outputs explicitly routed and named with "shape"
    distance_matrix_file = OUT_FEATS / "mimic_dtw_shape_pairwise_distance_matrix.npy"
    atlas_ids_file = OUT_FEATS / "mimic_shape_atlas_stay_ids.npy"
    scaler_file = OUT_MODELS / "mimic_dtw_shape_scaler.joblib"
    
    if not tensor_file.exists() or not id_file.exists():
        print(f"[ERROR] Required tensor files not found in {PROCESSED_DIR}")
        return

    # 1. Load the imputed clinical tensor and patient IDs
    print("    -> Loading 3D Tensor and Patient IDs...")
    X_clinical = np.load(tensor_file)
    stay_ids = np.load(id_file)
    
    num_patients, n_steps, n_features = X_clinical.shape
    print(f"    -> Cohort size: {num_patients} patients | Steps: {n_steps} | Features: {n_features}")

    # 2. Re-scale data along the time axis for shape/morphology isolation
    print("    -> Standardizing morphological features across the temporal axis...")
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X_clinical)
    
    # Save the fitted scaler for reproducibility
    joblib.dump(scaler, scaler_file)

    # 3. Compute All-Pairs DTW Distance Matrix
    print("    -> Computing pairwise DTW distance matrix...")
    
    # Pre-allocate the distance matrix
    dtw_matrix = np.zeros((num_patients, num_patients), dtype=np.float32)
    chunk_size = 200  
    
    # We iterate in chunks, only computing the upper triangle to save 50% of the compute time
    for i in tqdm(range(0, num_patients, chunk_size), desc="DTW Matrix Progress", unit="chunk"):
        end_i = min(i + chunk_size, num_patients)
        
        # Compare the current chunk to itself and all subsequent trajectories
        chunk_dist = cdist_dtw(X_scaled[i:end_i], X_scaled[i:], n_jobs=-1)
        
        # Assign to the upper triangle block
        dtw_matrix[i:end_i, i:] = chunk_dist
        
        # Mirror across the diagonal for the lower triangle block
        dtw_matrix[i:, i:end_i] = chunk_dist.T

    print(f"\n        - Distance Matrix Shape: {dtw_matrix.shape}")

    # 4. Serialization
    print("    -> Serializing pairwise distance matrix and patient mapping...")
    np.save(distance_matrix_file, dtw_matrix)
    np.save(atlas_ids_file, stay_ids)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Tensor Atlas distance matrix computed.")
    print(f"    -> Distance Matrix saved to: {distance_matrix_file.relative_to(BASE_DIR)}")
    print(f"    -> Atlas IDs saved to: {atlas_ids_file.relative_to(BASE_DIR)}")
    print(f"    -> DTW Scaler saved to: {scaler_file.relative_to(BASE_DIR)}")
    print(f"    -> Total Execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    run_tensor_atlas()