"""
Reconstruct the eICU tensor using the locked MIMIC-IV imputation model.

Inference only. The scaler and SAITS weights fitted on MIMIC-IV are loaded and
applied unchanged; nothing is refitted on eICU. This is what keeps the external
cohort a genuine target domain, and it means any difference between the two
imputed tensors reflects the data rather than two separately fitted models.

The same preparation as training is reapplied in the same order: log1p on the
six skewed variables, the locked scaler, inference, then inverse scaling,
expm1, and clipping to physiological ranges.

eICU is considerably sparser than MIMIC-IV, so a larger share of this tensor is
reconstructed. The QC report gives per-feature pre-imputation missingness
alongside the imputed distribution, and the feature parity audit in 07c
quantifies the gap directly.

Reads:
    eicu_sepsis_tensor_raw.npy and its feature names
    outputs/models/{mimic_saits_scaler.joblib, mimic_saits_model_weights.pypots}
Writes:
    eicu_sepsis_imputed_tensor.npy
    outputs/metrics/eicu_saits_qc_report.csv and distribution plots
"""

import time
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pypots.imputation import SAITS

warnings.filterwarnings("ignore")

# --- Configuration & Reproducibility -------------------------------------
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parents[2]

# eICU Paths (Flattened Data)
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"

# Global Output Paths (Models, Metrics, Figures)
OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

for d in [PROCESSED_DIR_EICU, OUT_METRICS, OUT_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

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
            axes[i].set_title(f"eICU {var.upper()} Distribution")
            axes[i].legend()
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def run_saits_inference():
    print("[*] Initiating Locked SAITS Inference Pipeline (No Retraining)...")
    start_time = time.time()
    
    # Inputs (eICU Data)
    raw_file = PROCESSED_DIR_EICU / "eicu_sepsis_tensor_raw.npy"
    feature_file = PROCESSED_DIR_EICU / "eicu_sepsis_tensor_features.npy"
    
    # Locked Artifacts (MIMIC-IV)
    locked_scaler_file = OUT_MODELS / "mimic_saits_scaler.joblib"
    locked_weights_file = OUT_MODELS / "mimic_saits_model_weights.pypots"
    
    # Outputs (eICU Imputed Data & QC)
    out_file = PROCESSED_DIR_EICU / "eicu_sepsis_imputed_tensor.npy"
    mask_file = PROCESSED_DIR_EICU / "eicu_saits_missingness_mask.npy"
    qc_file = OUT_METRICS / "eicu_saits_qc_report.csv"
    plot_file = OUT_FIGURES / "eicu_saits_imputation_distributions.png"
    
    if not raw_file.exists():
        print(f"[ERROR] Raw eICU tensor not found at {raw_file}")
        return
    if not locked_scaler_file.exists() or not locked_weights_file.exists():
        print(f"[ERROR] Locked MIMIC-IV SAITS artifacts not found in {OUT_MODELS}. Cannot proceed with external validation.")
        return

    # 1. Load Data
    X_3d_raw = np.load(raw_file)
    features = np.load(feature_file)
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
    
    # 3. Apply Locked MIMIC-IV Scaler
    print("    -> Applying locked MIMIC-IV StandardScaler...")
    scaler = joblib.load(locked_scaler_file)
    X_flat_scaled = scaler.transform(X_3d_transformed.reshape(-1, n_features))
    X_3d_scaled = X_flat_scaled.reshape(num_patients, n_steps, n_features)

    dataset_for_imputation = {"X": X_3d_scaled}

    # 4. Initialize SAITS and Load Locked Weights
    print("    -> Initializing SAITS architecture and injecting MIMIC-IV weights...")
    saits = SAITS(
        n_steps=n_steps, n_features=n_features, n_layers=2,
        d_model=256, d_ffn=128, n_heads=4, d_k=64, d_v=64,
        dropout=0.1, epochs=1 # Epochs are irrelevant for inference
    )
    
    # Load the learned temporal dynamics
    saits.load(str(locked_weights_file))
    
    # 5. Full Tensor Imputation
    print("    -> Imputing full eICU dataset (Inference Mode)...")
    imputed_dataset = saits.impute(dataset_for_imputation)
    X_imputed_scaled = imputed_dataset["imputation"] if isinstance(imputed_dataset, dict) else imputed_dataset
    
    # 6. Inverse Transforms & Clipping
    print("    -> Inverse scaling back to clinical units using MIMIC parameters...")
    X_imputed_flat = X_imputed_scaled.reshape(-1, n_features)
    X_imputed_transformed = scaler.inverse_transform(X_imputed_flat).reshape(num_patients, n_steps, n_features)
    
    for idx in skewed_indices:
        X_imputed_transformed[:, :, idx] = np.expm1(X_imputed_transformed[:, :, idx])
    
    print("    -> Clipping predictions to strict physiological boundaries...")
    for i, f in enumerate(features):
        if f in CLIPPING_BOUNDS:
            min_val, max_val = CLIPPING_BOUNDS[f]
            X_imputed_transformed[:, :, i] = np.clip(X_imputed_transformed[:, :, i], min_val, max_val)
    
    # 7. QC Report & Visualizations
    print("    -> Generating eICU QC Report and Distribution Plots...")
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
    
    np.save(out_file, X_imputed_transformed)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Locked Deep Learning Imputation complete in {elapsed:.2f} seconds.")
    print(f"    -> Imputed Tensor saved to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    run_saits_inference()
