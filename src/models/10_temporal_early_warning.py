"""
09_temporal_early_warning.py

Phase 11: Clinical Actionability (Lead-Time Bias / Temporal Slicing Analysis)
Evaluates how early the Champion XGBoost model achieves high predictive power.
1. Slices the 3D tensor into cumulative time windows: 6h, 12h, 18h, 24h.
2. Extracts static + temporal aggregations (Mean, Min, Max, Std) for each slice.
3. Retrains and evaluates the model using the locked Champion hyperparameters.
4. Generates a 'Time-to-Accuracy' curve to prove early clinical actionability.
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

TENSOR_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors"
COHORT_DIR = BASE_DIR / "data" / "processed" / "mimiciv"
SPLITS_DIR = BASE_DIR / "outputs" / "baselines" / "train_test_split"

CHAMPION_METRICS = BASE_DIR / "outputs" / "champion" / "metrics" / "champion_metrics.json"
OUT_PLOT = BASE_DIR / "outputs" / "champion" / "metrics" / "temporal_early_warning.png"
OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

TIME_WINDOWS = [6, 12, 18, 24]
RANDOM_STATE = 42

def main():
    print("[*] Initiating Clinical Actionability Analysis (Temporal Slicing)...")
    start_time = time.time()
    
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
    X_imputed = np.load(TENSOR_DIR / "sepsis_imputed_tensor.npy")
    stay_ids = np.load(TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    
    df_cohort = pd.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet")
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    idx_train_val = np.load(SPLITS_DIR / "train_indices.npy")
    idx_test = np.load(SPLITS_DIR / "test_indices.npy")
    
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

    # 5. Output Summary Table
    df_res = pd.DataFrame(results)
    
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

if __name__ == "__main__":
    main()