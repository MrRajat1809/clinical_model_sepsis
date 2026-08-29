"""
Pairwise DTW distance matrix preserving absolute severity, eICU cohort.

Severity counterpart to 08a. A single global scaler across all patients and
hours equalises the contribution of each variable while preserving how sick each
patient actually is, so proximity requires both similar shape and similar
magnitude.

Same O(N^2) cost; runs from run_dtw_phate_atlas.sh.

Reads:
    eicu_sepsis_imputed_tensor.npy, eicu_sepsis_tensor_stay_ids.npy
Writes:
    outputs/features/eicu_dtw_severity_pairwise_distance_matrix.npy
    outputs/features/eicu_severity_atlas_stay_ids.npy
"""

import time
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import cdist_dtw
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

# Flattened Inputs
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

# Flattened Outputs
OUT_FEATS = BASE_DIR / "outputs" / "features"
OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"

def run_clinical_atlas():
    print("[*] Initializing eICU Clinical Tensor Atlas & Pairwise DTW Pipeline...")
    start_time = time.time()
    
    for d in [OUT_FEATS, OUT_MODELS, OUT_METRICS]:
        d.mkdir(parents=True, exist_ok=True)
    
    tensor_file = PROCESSED_DIR / "eicu_sepsis_imputed_tensor.npy"
    id_file = PROCESSED_DIR / "eicu_sepsis_tensor_stay_ids.npy"
    
    # Atlas Outputs - Explicitly named with "severity" to avoid clashing with shape matrices
    distance_matrix_file = OUT_FEATS / "eicu_dtw_severity_pairwise_distance_matrix.npy"
    atlas_ids_file = OUT_FEATS / "eicu_severity_atlas_stay_ids.npy"
    scaler_file = OUT_MODELS / "eicu_global_severity_scaler.joblib"
    metadata_file = OUT_METRICS / "eicu_severity_atlas_metadata.json"
    
    if not tensor_file.exists() or not id_file.exists():
        print(f"[ERROR] Required tensor files not found in {PROCESSED_DIR}")
        return

    # 1. Load the imputed clinical tensor and patient IDs
    print("    -> Loading 3D Tensor and Patient IDs...")
    try:
        X_clinical = np.load(tensor_file)
        stay_ids = np.load(id_file)
    except Exception as e:
        print(f"[ERROR] Failed to load arrays. Error: {e}")
        return
    
    num_patients, n_steps, n_features = X_clinical.shape
    print(f"    -> eICU Cohort size: {num_patients:,} patients | Steps: {n_steps} | Features: {n_features}")

    # 2. Global Standardization (Preserving Magnitude/Severity)
    print("    -> Applying global standardization to balance features while preserving absolute magnitude...")
    scaler = StandardScaler()
    
    # Flatten to 2D (Patients * Steps, Features), scale, and reshape back to 3D
    X_flat = X_clinical.reshape(-1, n_features)
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(num_patients, n_steps, n_features)
    
    # Save the fitted scaler and atlas IDs
    joblib.dump(scaler, scaler_file)
    np.save(atlas_ids_file, stay_ids)
    
    # Export Atlas Metadata
    with open(metadata_file, "w") as f:
        json.dump({
            "distance_metric": "DTW",
            "scaling": "Global StandardScaler",
            "time_steps": int(n_steps),
            "features": int(n_features),
            "patients": int(num_patients)
        }, f, indent=4)

    # 3. Compute All-Pairs DTW Distance Matrix
    print("    -> Computing pairwise DTW distance matrix (Severity)...")
    
    # Pre-allocate the distance matrix as float32 to conserve memory
    dtw_matrix = np.zeros((num_patients, num_patients), dtype=np.float32)
    chunk_size = 200  
    
    # Iterate in chunks, computing the upper triangle to save compute time
    for i in tqdm(range(0, num_patients, chunk_size), desc="DTW Matrix Progress", unit="chunk"):
        end_i = min(i + chunk_size, num_patients)
        
        # Compare current chunk to itself and all subsequent trajectories
        chunk_dist = cdist_dtw(X_scaled[i:end_i], X_scaled[i:], n_jobs=-1)
        
        # Assign to upper triangle block
        dtw_matrix[i:end_i, i:] = chunk_dist
        
        # Mirror across diagonal for lower triangle block
        dtw_matrix[i:, i:end_i] = chunk_dist.T

    print(f"\n       - Distance Matrix Shape: {dtw_matrix.shape}")

    # 4. Serialization
    print("    -> Serializing clinical distance matrix...")
    np.save(distance_matrix_file, dtw_matrix)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! eICU Clinical Tensor Atlas distance matrix computed.")
    print(f"    -> Distance Matrix saved to: {distance_matrix_file.relative_to(BASE_DIR)}")
    print(f"    -> Total Execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    run_clinical_atlas()
