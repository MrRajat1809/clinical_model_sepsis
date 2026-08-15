"""
12_eicu_temporal_early_warning.py

External Validation of Clinical Actionability (Lead-Time Bias)
Evaluates how early the Champion model generalizes to the eICU hospital system.

Features included:
- Slices both MIMIC and eICU 3D tensors into cumulative windows: 6h, 12h, 18h, 24h.
- Fits the StandardScaler on MIMIC's time slice, applies it to eICU's time slice.
- Retrains the locked XGBoost architecture on MIMIC.
- Evaluates strictly on the eICU external validation dataset.
- Generates the External Time-to-Accuracy plot.
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
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"

# Flattened Global Outputs
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

# Inputs
CHAMPION_METRICS = OUT_METRICS / "champion_metrics.json"

MIMIC_TENSOR = PROCESSED_DIR_MIMIC / "mimic_sepsis_imputed_tensor.npy"
MIMIC_IDS = PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_stay_ids.npy"
MIMIC_COHORT = PROCESSED_DIR_MIMIC / "mimic_final_sepsis3_cohort.parquet"

EICU_TENSOR = PROCESSED_DIR_EICU / "eicu_sepsis_imputed_tensor.npy"
EICU_IDS = PROCESSED_DIR_EICU / "eicu_sepsis_tensor_stay_ids.npy"
EICU_COHORT = PROCESSED_DIR_EICU / "eicu_final_sepsis3_cohort.parquet"

# Outputs
OUT_PLOT = OUT_FIGURES / "eicu_temporal_early_warning.png"
OUT_JSON = OUT_METRICS / "eicu_temporal_early_warning_metrics.json"

TIME_WINDOWS = [6, 12, 18, 24]
RANDOM_STATE = 42

def extract_static(df, ids):
    df_aligned = pd.DataFrame({"stay_id": ids}).merge(df, on="stay_id", how="left")
    y = df_aligned["hospital_expire_flag"].values
    
    statics = ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"]
    df_static = df_aligned[statics].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"].astype(str).str.upper() == "M").astype(int)
        
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
        
    best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1})

    # 1. Load Tensors & Cohorts
    print("    -> Loading MIMIC (Train) and eICU (Test) imputed tensors...")
    try:
        X_mimic_3d = np.load(MIMIC_TENSOR)
        ids_mimic = np.load(MIMIC_IDS)
        df_mimic = pd.read_parquet(MIMIC_COHORT)

        X_eicu_3d = np.load(EICU_TENSOR)
        ids_eicu = np.load(EICU_IDS)
        df_eicu = pd.read_parquet(EICU_COHORT)
    except Exception as e:
        print(f"[ERROR] Failed to load processed data arrays. Error: {e}")
        return

    # 2. Extract Static Features
    X_stat_mimic_raw, y_mimic = extract_static(df_mimic, ids_mimic)
    X_stat_eicu_raw, y_eicu = extract_static(df_eicu, ids_eicu)

    scaler_stat = StandardScaler().fit(X_stat_mimic_raw)
    X_stat_mimic = scaler_stat.transform(X_stat_mimic_raw)
    X_stat_eicu = scaler_stat.transform(X_stat_eicu_raw)

    best_params["scale_pos_weight"] = float((len(y_mimic) - sum(y_mimic)) / sum(y_mimic))

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
        
        # Scale (Fit on MIMIC, transform both)
        scaler_temp = StandardScaler().fit(X_mimic_agg)
        X_mimic_fused = np.concatenate([X_stat_mimic, scaler_temp.transform(X_mimic_agg)], axis=1)
        X_eicu_fused = np.concatenate([X_stat_eicu, scaler_temp.transform(X_eicu_agg)], axis=1)
        
        # Train on MIMIC, Predict on eICU
        model = XGBClassifier(**best_params)
        model.fit(X_mimic_fused, y_mimic)
        
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
    ax1.set_ylim(0.65, 0.85) # Adjusted for eICU's ~0.79 max
    ax1.set_xticks(TIME_WINDOWS)

    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('External AUPRC (eICU)', color=color2, fontsize=12, fontweight='bold')  
    ax2.plot(df_res['Hours'], df_res['AUPRC'], marker='s', linewidth=2.5, markersize=8, color=color2, label='AUPRC')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.40, 0.65) # Adjusted for eICU's ~0.55 max

    plt.title('External Validation Time-to-Accuracy:\nPredictive Power on eICU by Cumulative Clinical Window', fontsize=14)
    fig.tight_layout()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    fig.legend(loc="lower right", bbox_to_anchor=(0.85, 0.15), fontsize=11)
    
    plt.savefig(OUT_PLOT, dpi=300)
    
    print(f"[*] Pipeline completed in {time.time() - start_time:.2f} seconds.")
    print(f"    -> Metrics saved to: {OUT_JSON.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()