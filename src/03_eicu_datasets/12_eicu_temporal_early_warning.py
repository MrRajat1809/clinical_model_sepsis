"""
How early the prognostic signal transports, measured externally.

External counterpart of 02_models/10. Rebuilds the feature representation from
cumulative 6, 12, 18 and 24 hour windows in both cohorts, trains on the MIMIC-IV
development partition for each window, and evaluates only on eICU.

Both cohorts are sliced and aggregated identically, and each window's scaler is
fitted on the MIMIC-IV development partition and applied to eICU, so the
external features stay in the source statistical space at every horizon.

Read against the internal curve, this separates how quickly the signal
accumulates from how much of it survives the move between systems.

Reads:
    outputs/metrics/mimic_champion_metrics.json for the hyperparameters
    both imputed tensors and cohort tables, the shared split indices
Writes:
    outputs/metrics/eicu_temporal_early_warning_metrics.json and a plot
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

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

# Flattened Data Directories
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"

OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

CHAMPION_METRICS = OUT_METRICS / "mimic_champion_metrics.json"

MIMIC_TENSOR = PROCESSED_DIR_MIMIC / "mimic_sepsis_imputed_tensor.npy"
MIMIC_IDS = PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_stay_ids.npy"
MIMIC_COHORT = PROCESSED_DIR_MIMIC / "mimic_final_sepsis3_cohort.parquet"
MIMIC_TRAIN_IDX = OUT_MODELS / "mimic_train_indices.npy"

EICU_TENSOR = PROCESSED_DIR_EICU / "eicu_sepsis_imputed_tensor.npy"
EICU_IDS = PROCESSED_DIR_EICU / "eicu_sepsis_tensor_stay_ids.npy"
EICU_COHORT = PROCESSED_DIR_EICU / "eicu_final_sepsis3_cohort.parquet"

OUT_PLOT = OUT_FIGURES / "eicu_temporal_early_warning.png"
OUT_JSON = OUT_METRICS / "eicu_temporal_early_warning_metrics.json"

TIME_WINDOWS = [6, 12, 18, 24]
RANDOM_STATE = 42
# Fixed rather than -1: thread count changes the order of floating-point
# accumulation, so "all cores" makes results depend on the machine.
N_JOBS = 8

def extract_static(df, ids):
    df_aligned = pd.DataFrame({"stay_id": ids}).merge(df, on="stay_id", how="left")
    y = df_aligned["hospital_expire_flag"].values
    
    statics = ["age", "baseline_sofa"]
    df_static = df_aligned[statics].copy()
        
    return df_static.fillna(0).values, y

def main():
    print("[*] Initiating External Clinical Actionability Analysis on eICU...")
    start_time = time.time()
    
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    try:
        with open(CHAMPION_METRICS, "r") as f:
            best_params = json.load(f)["hyperparameters"]
    except Exception as e:
        print(f"[ERROR] Could not load Champion hyperparameters at {CHAMPION_METRICS}. Error: {e}")
        return
        
    best_params.update({"random_state": RANDOM_STATE, "n_jobs": N_JOBS})

    # 1. Load Tensors & Cohorts
    print("    -> Loading MIMIC (Train) and eICU (Test) imputed tensors...")
    try:
        X_mimic_3d = np.load(MIMIC_TENSOR)
        ids_mimic = np.load(MIMIC_IDS)
        df_mimic = pd.read_parquet(MIMIC_COHORT)
        mimic_train_idx = np.load(MIMIC_TRAIN_IDX)

        X_eicu_3d = np.load(EICU_TENSOR)
        ids_eicu = np.load(EICU_IDS)
        df_eicu = pd.read_parquet(EICU_COHORT)
    except Exception as e:
        print(f"[ERROR] Failed to load processed data arrays. Error: {e}")
        return

    # 2. Extract Static Features
    X_stat_mimic_raw, y_mimic = extract_static(df_mimic, ids_mimic)
    X_stat_eicu_raw, y_eicu = extract_static(df_eicu, ids_eicu)

    # Fit scaler ONLY on train indices
    scaler_stat = StandardScaler()
    scaler_stat.fit(X_stat_mimic_raw[mimic_train_idx])
    
    X_stat_mimic = scaler_stat.transform(X_stat_mimic_raw)
    X_stat_eicu = scaler_stat.transform(X_stat_eicu_raw)

    # This was computed from the full cohort and then
    # overwritten inside the loop from the training subset. Dead assignment.

    # 3. Temporal Ablation Loop
    results = []
    print(f"\n    -> Running Temporal Ablation for {TIME_WINDOWS} hours post-onset...")
    
    for hours in TIME_WINDOWS:
        # Slice Tensors
        X_mimic_slice = X_mimic_3d[:, :hours, :]
        X_eicu_slice = X_eicu_3d[:, :hours, :]
        
        # Aggregate MIMIC
        X_mimic_agg = np.concatenate([
            np.mean(X_mimic_slice, axis=1), np.min(X_mimic_slice, axis=1),
            np.max(X_mimic_slice, axis=1), np.std(X_mimic_slice, axis=1)
        ], axis=1)
        
        # Aggregate eICU
        X_eicu_agg = np.concatenate([
            np.mean(X_eicu_slice, axis=1), np.min(X_eicu_slice, axis=1),
            np.max(X_eicu_slice, axis=1), np.std(X_eicu_slice, axis=1)
        ], axis=1)
        
        # Scale Fit on mimic_train_idx, transform both
        scaler_temp = StandardScaler()
        scaler_temp.fit(X_mimic_agg[mimic_train_idx])
        
        X_mimic_fused = np.concatenate([X_stat_mimic, scaler_temp.transform(X_mimic_agg)], axis=1)
        X_eicu_fused = np.concatenate([X_stat_eicu, scaler_temp.transform(X_eicu_agg)], axis=1)
        
        # Train on MIMIC, Predict on eICU
        # Train strictly on MIMIC train_val split to match internal methodology
        X_mimic_train_val = X_mimic_fused[mimic_train_idx]
        y_mimic_train_val = y_mimic[mimic_train_idx]
        
        # Re-calculate scale_pos_weight dynamically based ONLY on the train subset
        best_params["scale_pos_weight"] = float((len(y_mimic_train_val) - sum(y_mimic_train_val)) / sum(y_mimic_train_val))
        
        model = XGBClassifier(**best_params)
        model.fit(X_mimic_train_val, y_mimic_train_val)
        
        preds = model.predict_proba(X_eicu_fused)[:, 1]
        
        auroc = roc_auc_score(y_eicu, preds)
        auprc = average_precision_score(y_eicu, preds)
        brier = brier_score_loss(y_eicu, preds)
        
        results.append({"Hours": hours, "AUROC": auroc, "AUPRC": auprc, "Brier": brier})
        print(f"       - Window [0 to {hours}h] | External AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Brier: {brier:.4f}")

    # 4. Output Summary Table & Save JSON
    df_res = pd.DataFrame(results)
    
    with open(OUT_JSON, "w") as f:
        json.dump({"eicu_temporal_ablation": results}, f, indent=4)
    
    print("\n============================================================")
    print(" EXTERNAL CLINICAL ACTIONABILITY: eICU TIME-TO-ACCURACY")
    print("============================================================")
    print(df_res.to_string(index=False))
    print("============================================================")

    # 5. Generate Publication Plot
    print(f"\n    -> Generating Plot at {OUT_PLOT.relative_to(BASE_DIR)}...")
    
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Hours Post Sepsis-3 Onset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('External AUROC (eICU)', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(df_res['Hours'], df_res['AUROC'], marker='o', linewidth=2.5, markersize=8, color=color1, label='AUROC')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.65, 0.85) 
    ax1.set_xticks(TIME_WINDOWS)

    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('External AUPRC (eICU)', color=color2, fontsize=12, fontweight='bold')  
    ax2.plot(df_res['Hours'], df_res['AUPRC'], marker='s', linewidth=2.5, markersize=8, color=color2, label='AUPRC')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.40, 0.65) 

    plt.title('External Validation Time-to-Accuracy:\nPredictive Power on eICU by Cumulative Clinical Window', fontsize=14)
    fig.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    fig.legend(loc="lower right", bbox_to_anchor=(0.85, 0.15), fontsize=11)
    
    plt.savefig(OUT_PLOT, dpi=300)
    
    print(f"[*] Pipeline completed in {time.time() - start_time:.2f} seconds.")
    print(f"    -> Plot saved to: {OUT_PLOT.relative_to(BASE_DIR)}")
    print(f"    -> Metrics saved to: {OUT_JSON.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
