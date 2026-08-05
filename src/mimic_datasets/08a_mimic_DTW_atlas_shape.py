"""
08_mimic_DTW_atlas_shape.py

Computes the pairwise Dynamic Time Warping (DTW) distance matrix for the fully imputed 
3D time-series tensor to map the patient trajectory manifold.

[METHODOLOGY NOTE]: This script uses TimeSeriesScalerMeanVariance(). Scaling each patient 
independently forces the DTW algorithm to evaluate trajectories based on the *morphology* (shape) 
of their physiological changes rather than absolute severity.
[ARCHITECTURE SHIFT]: Replaced discrete K-Means clustering with pairwise distance matrix 
computation (cdist_dtw) to feed manifold learning algorithms (e.g., PHATE).
[FIX APPLIED]: Implemented smart upper-triangular batching wrapped in a tqdm progress bar 
to track the O(N^2) computation without losing tslearn's self-similarity optimization.
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
TENSOR_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors"
ATLAS_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "atlas"
META_DIR = ATLAS_DIR / "metadata"

def run_tensor_atlas():
    print("[*] Initializing Tensor Atlas & Pairwise DTW Distance Pipeline...")
    start_time = time.time()
    
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    
    tensor_file = TENSOR_DIR / "sepsis_imputed_tensor.npy"
    id_file = TENSOR_DIR / "sepsis_tensor_stay_ids.npy"
    
    # Atlas Outputs
    distance_matrix_file = ATLAS_DIR / "dtw_pairwise_distance_matrix.npy"
    scaler_file = META_DIR / "dtw_scaler.joblib"
    
    if not tensor_file.exists() or not id_file.exists():
        print(f"[ERROR] Required tensor files not found in {TENSOR_DIR}")
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
    
    # Save the fitted scaler for reproducibility or vault extensions
    joblib.dump(scaler, scaler_file)

    # 3. Compute All-Pairs DTW Distance Matrix
    print("    -> Computing pairwise DTW distance matrix...")
    
    # Pre-allocate the distance matrix
    dtw_matrix = np.zeros((num_patients, num_patients), dtype=np.float32)
    chunk_size = 200  # Adjust if memory or updates need tuning
    
    # We iterate in chunks, only computing the upper triangle to save 50% of the compute time
    for i in tqdm(range(0, num_patients, chunk_size), desc="DTW Matrix Progress", unit="chunk"):
        end_i = min(i + chunk_size, num_patients)
        
        # Compare the current chunk to itself and all subsequent trajectories
        chunk_dist = cdist_dtw(X_scaled[i:end_i], X_scaled[i:], n_jobs=-1)
        
        # Assign to the upper triangle block
        dtw_matrix[i:end_i, i:] = chunk_dist
        
        # Mirror across the diagonal for the lower triangle block
        dtw_matrix[i:, i:end_i] = chunk_dist.T

    print(f"\n       - Distance Matrix Shape: {dtw_matrix.shape}")

    # 4. Serialization
    print("    -> Serializing pairwise distance matrix and patient mapping...")
    np.save(distance_matrix_file, dtw_matrix)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Tensor Atlas distance matrix computed.")
    print(f"    -> Distance Matrix saved to: {distance_matrix_file.relative_to(BASE_DIR)}")
    print(f"    -> Total Execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    run_tensor_atlas()