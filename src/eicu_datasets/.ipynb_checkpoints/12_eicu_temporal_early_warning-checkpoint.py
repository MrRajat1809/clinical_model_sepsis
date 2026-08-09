"""
12_eicu_temporal_early_warning.py

External Validation of Clinical Actionability (Lead-Time Bias)
Evaluates how early the Champion model generalizes to the eICU hospital system.
1. Slices both MIMIC and eICU 3D tensors into cumulative windows: 6h, 12h, 18h, 24h.
2. Fits the StandardScaler on MIMIC's time slice, applies it to eICU's time slice.
3. Retrains the locked XGBoost architecture on MIMIC.
4. Evaluates strictly on the eICU external validation dataset.
5. Generates the External Time-to-Accuracy plot.
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

# MIMIC Data (For Training & Scaler Fitting)
MIMIC_TENSOR_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors"
MIMIC_COHORT_DIR = BASE_DIR / "data" / "processed" / "mimiciv"
CHAMPION_METRICS = BASE_DIR / "outputs" / "champion" / "metrics" / "champion_metrics.json"

# eICU Data (For External Testing)
EICU_TENSOR_DIR = BASE_DIR / "data" / "processed" / "eicu" / "tensors"
EICU_COHORT_DIR = BASE_DIR / "data" / "processed" / "eicu"

OUT_PLOT = BASE_DIR / "results" / "eicu_external_validation" / "eicu_temporal_early_warning.png"
OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

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

    with open(CHAMPION_METRICS, "r") as f:
        best_params = json.load(f)["hyperparameters"]
    best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1})

    # 1. Load Tensors & Cohorts
    print("    -> Loading MIMIC (Train) and eICU (Test) imputed tensors...")
    X_mimic_3d = np.load(MIMIC_TENSOR_DIR / "sepsis_imputed_tensor.npy")
    ids_mimic = np.load(MIMIC_TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    df_mimic = pd.read_parquet(MIMIC_COHORT_DIR / "final_sepsis3_cohort.parquet")

    X_eicu_3d = np.load(EICU_TENSOR_DIR / "eicu_sepsis_imputed_tensor.npy")
    ids_eicu = np.load(EICU_TENSOR_DIR / "eicu_sepsis_tensor_stay_ids.npy")
    df_eicu = pd.read_parquet(EICU_COHORT_DIR / "eicu_final_sepsis3_cohort.parquet")

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

    # 4. Output Summary Table
    df_res = pd.DataFrame(results)
    
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

if __name__ == "__main__":
    main()