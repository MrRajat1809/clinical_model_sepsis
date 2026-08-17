"""
10_temporal_early_warning.py

Clinical Actionability (Lead-Time Bias / Temporal Slicing Analysis)
Evaluates how early the Champion XGBoost model achieves high predictive power.

Features included:
- Slices the 3D tensor into cumulative time windows: 6h, 12h, 18h, 24h.
- Extracts static + temporal aggregations (Mean, Min, Max, Std) for each slice.
- Retrains and evaluates the model using the locked Champion hyperparameters.
- Generates a 'Time-to-Accuracy' curve to prove early clinical actionability.
"""

import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Flattened Data Directories
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"

# Flattened Global Outputs
OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

# Inputs
CHAMPION_METRICS = OUT_METRICS / "mimic_champion_metrics.json"

MIMIC_TENSOR = PROCESSED_DIR_MIMIC / "mimic_sepsis_imputed_tensor.npy"
MIMIC_IDS = PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_stay_ids.npy"
MIMIC_COHORT = PROCESSED_DIR_MIMIC / "mimic_final_sepsis3_cohort.parquet"

# [FIX]: Pointed to OUT_MODELS and correct test set filename
MIMIC_TRAIN_IDX = OUT_MODELS / "mimic_train_indices.npy"
MIMIC_TEST_IDX = OUT_MODELS / "mimic_test_set_indices.npy"

# Outputs
OUT_PLOT = OUT_FIGURES / "mimic_temporal_early_warning.png"
OUT_JSON = OUT_METRICS / "mimic_temporal_early_warning_metrics.json"

TIME_WINDOWS = [6, 12, 18, 24]
RANDOM_STATE = 42

def main():
    print("[*] Initiating Clinical Actionability Analysis (Temporal Slicing)...")
    start_time = time.time()
    
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    
    if not CHAMPION_METRICS.exists():
        print(f"[ERROR] Could not find Champion parameters at {CHAMPION_METRICS}")
        return

    # 1. Load Locked Champion Hyperparameters
    with open(CHAMPION_METRICS, "r") as f:
        champ_data = json.load(f)
    best_params = champ_data["hyperparameters"]
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1

    # 2. Load Data & Splits
    print("    -> Loading MIMIC-IV 3D Tensor and Cohort splits...")
    try:
        X_imputed = np.load(MIMIC_TENSOR)
        stay_ids = np.load(MIMIC_IDS)
        
        df_cohort = pd.read_parquet(MIMIC_COHORT)
        
        idx_train_val = np.load(MIMIC_TRAIN_IDX)
        idx_test = np.load(MIMIC_TEST_IDX)
    except Exception as e:
        print(f"[ERROR] Failed to load MIMIC-IV arrays. Error: {e}")
        return
        
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    # 3. Process Static Features (Constant across all time windows)
    potential_statics = ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"]
    static_cols = [col for col in potential_statics if col in df_cohort.columns]
    
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
        
    X_static_raw = df_static.fillna(0).values
    
    scaler_static = StandardScaler()
    scaler_static.fit(X_static_raw[idx_train_val])
    X_static = scaler_static.transform(X_static_raw)

    # 4. Iterate over Temporal Slices
    results = []
    print(f"\n    -> Running Temporal Ablation for {TIME_WINDOWS} hours post-onset...")
    
    for hours in TIME_WINDOWS:
        # Slice the tensor up to the current hour
        X_slice = X_imputed[:, :hours, :]
        
        # Aggregate based ONLY on data available up to 'hours'
        X_mean = np.mean(X_slice, axis=1)
        X_min = np.min(X_slice, axis=1)
        X_max = np.max(X_slice, axis=1)
        X_std = np.std(X_slice, axis=1)
        
        X_temporal_agg = StandardScaler().fit_transform(np.concatenate([X_mean, X_min, X_max, X_std], axis=1))
        X_fused = np.concatenate([X_static, X_temporal_agg], axis=1)
        
        X_train_val, y_train_val = X_fused[idx_train_val], y[idx_train_val]
        X_test, y_test = X_fused[idx_test], y[idx_test]
        
        # Add scale_pos_weight dynamically
        best_params["scale_pos_weight"] = float((len(y_train_val) - sum(y_train_val)) / sum(y_train_val))
        
        # Train and Evaluate
        model = XGBClassifier(**best_params)
        model.fit(X_train_val, y_train_val)
        
        preds = model.predict_proba(X_test)[:, 1]
        
        auroc = roc_auc_score(y_test, preds)
        auprc = average_precision_score(y_test, preds)
        brier = brier_score_loss(y_test, preds)
        
        results.append({"Hours": hours, "AUROC": auroc, "AUPRC": auprc, "Brier": brier})
        print(f"       - Window [0 to {hours}h] | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Brier: {brier:.4f}")

    # 5. Output Summary Table & Save JSON
    df_res = pd.DataFrame(results)
    
    with open(OUT_JSON, "w") as f:
        json.dump({"mimic_temporal_ablation": results}, f, indent=4)
    
    print("\n============================================================")
    print(" CLINICAL ACTIONABILITY: TIME-TO-ACCURACY METRICS")
    print("============================================================")
    print(df_res.to_string(index=False))
    print("============================================================")

    # 6. Generate Publication Plot
    print(f"\n    -> Generating Plot at {OUT_PLOT.relative_to(BASE_DIR)}...")
    
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Hours Post Sepsis-3 Onset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUROC', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(df_res['Hours'], df_res['AUROC'], marker='o', linewidth=2.5, markersize=8, color=color1, label='AUROC')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.70, 0.95)
    ax1.set_xticks(TIME_WINDOWS)

    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('AUPRC', color=color2, fontsize=12, fontweight='bold')  
    ax2.plot(df_res['Hours'], df_res['AUPRC'], marker='s', linewidth=2.5, markersize=8, color=color2, label='AUPRC')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.40, 0.75)

    plt.title('Time-to-Accuracy Analysis:\nPredictive Power by Cumulative Clinical Window', fontsize=14)
    fig.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    fig.legend(loc="lower right", bbox_to_anchor=(0.85, 0.15), fontsize=11)
    
    plt.savefig(OUT_PLOT, dpi=300)
    print(f"[*] Pipeline completed in {time.time() - start_time:.2f} seconds.")
    print(f"    -> Plot saved to: {OUT_PLOT.relative_to(BASE_DIR)}")
    print(f"    -> Metrics saved to: {OUT_JSON.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()