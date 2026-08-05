"""
07b_mimic_saits_imputation.py

Loads the raw 3D tensor containing NaNs. Applies log1p transformations to skewed features, 
scales the data, and trains a PyPOTS SAITS model.
Uses an 80/20 stratified validation split for early stopping.
Imputes missing variables, reverses transformations, clips to physiological bounds, 
and generates comprehensive publication-grade QC reports and distribution plots.
"""

import time
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pypots.imputation import SAITS

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & REPRODUCIBILITY
# ==========================================
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parents[2]
TENSOR_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors"
META_DIR = TENSOR_DIR / "metadata"

SKEWED_FEATURES = ["lactate", "bilirubin", "creatinine", "bun", "urine_output", "neq"]

CLIPPING_BOUNDS = {
    "hr": (0, 250), "map": (0, 200), "rr": (0, 80), "temp_c": (25, 45), "spo2": (0, 100),
    "gcs_eye": (1, 4), "gcs_verbal": (1, 5), "gcs_motor": (1, 6),
    "pao2": (20, 800), "fio2": (0.2, 1.0), "pf_ratio": (0, 1000), "paco2": (10, 200),
    "lactate": (0, 30), "creatinine": (0, 20), "bun": (0, 200), "bilirubin": (0, 50),
    "platelets": (0, 2000), "wbc": (0, 300), "hemoglobin": (0, 30),
    "ph": (6.5, 8.0), "pt": (0, 200), "aptt": (0, 300), "albumin": (0, 15),
    "potassium": (0, 15), "sodium": (90, 200), "glucose": (10, 2000), "chloride": (50, 160),
    "urine_output": (0, 5000), "neq": (0, 1000), "vent": (0, 1)
}

def plot_distributions(X_raw, X_imputed, features, target_vars, out_path):
    """Generates overlaid histograms for pre- and post-imputation distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(target_vars):
        if var in features:
            idx = np.where(features == var)[0][0]
            raw_vals = X_raw[:, :, idx][~np.isnan(X_raw[:, :, idx])]
            imp_vals = X_imputed[:, :, idx].flatten()
            
            axes[i].hist(imp_vals, bins=50, alpha=0.6, color='blue', label='Imputed', density=True)
            axes[i].hist(raw_vals, bins=50, alpha=0.6, color='red', label='Raw (Observed)', density=True)
            axes[i].set_title(f"{var.upper()} Distribution")
            axes[i].legend()
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def run_saits_model():
    print("[*] Initializing SAITS Deep Learning Imputer (Publication Grade)...")
    start_time = time.time()
    META_DIR.mkdir(parents=True, exist_ok=True)
    
    raw_file = TENSOR_DIR / "sepsis_tensor_raw.npy"
    feature_file = TENSOR_DIR / "sepsis_tensor_features.npy"
    label_file = TENSOR_DIR / "sepsis_tensor_labels.npy"
    out_file = TENSOR_DIR / "sepsis_imputed_tensor.npy"
    
    # Export paths
    scaler_file = META_DIR / "saits_scaler.joblib"
    mask_file = META_DIR / "saits_missingness_mask.npy"
    weights_file = META_DIR / "saits_model_weights.pypots"
    qc_file = META_DIR / "sepsis_tensor_qc_report.csv"
    config_file = META_DIR / "preprocessing_config.json"
    train_idx_file = META_DIR / "saits_train_indices.npy"
    val_idx_file = META_DIR / "saits_val_indices.npy"
    plot_file = META_DIR / "imputation_distributions.png"
    
    if not raw_file.exists():
        print(f"[ERROR] Raw tensor not found at {raw_file}")
        return

    # 1. Load Data
    X_3d_raw = np.load(raw_file)
    features = np.load(feature_file)
    y_labels = np.load(label_file)
    num_patients, n_steps, n_features = X_3d_raw.shape
    
    missingness_mask = np.isnan(X_3d_raw)
    np.save(mask_file, missingness_mask)
    
    # 2. Log1p Transformation
    print("    -> Applying Log1p transforms to highly skewed variables...")
    X_3d_transformed = X_3d_raw.copy()
    skewed_indices = [i for i, f in enumerate(features) if f in SKEWED_FEATURES]
    
    for idx in skewed_indices:
        valid_mask = ~np.isnan(X_3d_transformed[:, :, idx])
        X_3d_transformed[:, :, idx][valid_mask] = np.log1p(X_3d_transformed[:, :, idx][valid_mask])
    
    # 3. Standard Scaling
    print("    -> Scaling data for neural network stability...")
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_3d_transformed.reshape(-1, n_features))
    X_3d_scaled = X_flat_scaled.reshape(num_patients, n_steps, n_features)
    joblib.dump(scaler, scaler_file)

    # 4. Stratified Train/Validation Split
    print("    -> Splitting data (80/20) with mortality stratification...")
    indices = np.arange(num_patients)
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42, stratify=y_labels)
    
    np.save(train_idx_file, train_idx)
    np.save(val_idx_file, val_idx)
    
    dataset_for_training = {"X": X_3d_scaled[train_idx]}
    dataset_for_validating = {"X": X_3d_scaled[val_idx], "X_ori": X_3d_scaled[val_idx]} 
    dataset_for_imputation = {"X": X_3d_scaled}

    # 5. Model Initialization & Training
    print("    -> Initializing SAITS Architecture (epochs=100, patience=10)...")
    saits = SAITS(
        n_steps=n_steps, n_features=n_features, n_layers=2,
        d_model=256, d_ffn=128, n_heads=4, d_k=64, d_v=64,
        dropout=0.1, epochs=100, patience=10,
        saving_path=str(META_DIR), model_saving_strategy="best"
    )
    
    print("    -> Training SAITS model...")
    saits.fit(train_set=dataset_for_training, val_set=dataset_for_validating)
    saits.save(str(weights_file))
    
    # 6. Full Tensor Imputation
    print("    -> Imputing full dataset...")
    imputed_dataset = saits.impute(dataset_for_imputation)
    X_imputed_scaled = imputed_dataset["imputation"] if isinstance(imputed_dataset, dict) else imputed_dataset
    
    # 7. Inverse Transforms & Clipping
    print("    -> Inverse scaling back to clinical units...")
    X_imputed_flat = X_imputed_scaled.reshape(-1, n_features)
    X_imputed_transformed = scaler.inverse_transform(X_imputed_flat).reshape(num_patients, n_steps, n_features)
    
    for idx in skewed_indices:
        X_imputed_transformed[:, :, idx] = np.expm1(X_imputed_transformed[:, :, idx])
    
    print("    -> Clipping predictions to strict physiological boundaries...")
    for i, f in enumerate(features):
        if f in CLIPPING_BOUNDS:
            min_val, max_val = CLIPPING_BOUNDS[f]
            X_imputed_transformed[:, :, i] = np.clip(X_imputed_transformed[:, :, i], min_val, max_val)
    
    # 8. QC Report & Visualizations
    print("    -> Generating QC Report and Distribution Plots...")
    qc_data = []
    for i, f in enumerate(features):
        raw_missing = np.isnan(X_3d_raw[:, :, i]).mean() * 100
        qc_data.append({
            "Feature": f,
            "Missingness_Pre_Imputation_%": round(raw_missing, 2),
            "Imputed_Min": round(np.min(X_imputed_transformed[:, :, i]), 2),
            "Imputed_Median": round(np.median(X_imputed_transformed[:, :, i]), 2),
            "Imputed_Mean": round(np.mean(X_imputed_transformed[:, :, i]), 2),
            "Imputed_Std": round(np.std(X_imputed_transformed[:, :, i]), 2),
            "Imputed_99th_Pct": round(np.percentile(X_imputed_transformed[:, :, i], 99), 2),
            "Imputed_Max": round(np.max(X_imputed_transformed[:, :, i]), 2)
        })
        
    pd.DataFrame(qc_data).to_csv(qc_file, index=False)
    
    plot_distributions(X_3d_raw, X_imputed_transformed, features, 
                       ["lactate", "creatinine", "pf_ratio", "urine_output"], plot_file)
    
    # Save Preprocessing Config
    with open(config_file, "w") as f:
        json.dump({
            "log1p_features": SKEWED_FEATURES,
            "scaled": True,
            "clip_bounds": CLIPPING_BOUNDS,
            "split": {"train_size": len(train_idx), "val_size": len(val_idx), "stratified": True}
        }, f, indent=4)

    np.save(out_file, X_imputed_transformed)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Deep Learning Imputation complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_saits_model()