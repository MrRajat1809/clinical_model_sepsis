"""
10_eicu_xgboost_external_validation.py

Loads the locked Champion XGBoost model trained on MIMIC-IV and evaluates it 
on the eICU external validation cohort.

Features included:
- Strictly enforces external validation standards by PREVENTING any retraining on eICU data.
- Dynamically reconstructs the MIMIC-IV StandardScalers to ensure the eICU data 
  is transformed into the exact same statistical distribution space the trees were trained on.
- Applies the exact Mean/Min/Max/Std aggregation logic used during the MIMIC-IV feature engineering.
- Calculates AUROC, AUPRC, and Brier Score with 1000-iteration Bootstrapped 95% CIs.
"""

import time
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, balanced_accuracy_score, precision_score,
    roc_curve, precision_recall_curve
)
from scipy.special import logit

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Flattened Data Directories
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"

# Flattened Output Directories
OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

# eICU Paths (Target Validation Data)
EICU_TENSOR_FILE = PROCESSED_DIR_EICU / "eicu_sepsis_imputed_tensor.npy"
EICU_STAY_ID_FILE = PROCESSED_DIR_EICU / "eicu_sepsis_tensor_stay_ids.npy"
EICU_COHORT_FILE = PROCESSED_DIR_EICU / "eicu_final_sepsis3_cohort.parquet"

# MIMIC-IV Paths (Source Data for Scaler Reconstruction)
MIMIC_TENSOR_FILE = PROCESSED_DIR_MIMIC / "mimic_sepsis_imputed_tensor.npy"
MIMIC_STAY_ID_FILE = PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_stay_ids.npy"
MIMIC_COHORT_FILE = PROCESSED_DIR_MIMIC / "mimic_final_sepsis3_cohort.parquet"

# [FIX]: Train indices are located in OUT_MODELS
MIMIC_TRAIN_IDX_FILE = OUT_MODELS / "mimic_train_indices.npy"

# [FIX]: Updated locked Champion Model Path
XGB_MODEL_FILE = OUT_MODELS / "mimic_champion_xgboost.joblib"

# Outputs
METRICS_FILE = OUT_METRICS / "eicu_champion_metrics.json"
PREDS_FILE = OUT_METRICS / "eicu_champion_predictions.csv"
ROC_PLOT_FILE = OUT_FIGURES / "eicu_Champion_ROC_PR.png"

RANDOM_STATE = 42
N_BOOTSTRAPS = 1000

# ==========================================
# EVALUATION HELPERS (Mirrored from Training)
# ==========================================
def compute_calibration_metrics(y_true, y_prob):
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    logits = logit(y_prob_clipped).reshape(-1, 1)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state=RANDOM_STATE)
    lr.fit(logits, y_true)

    return lr.coef_[0][0], lr.intercept_[0]

def evaluate_champion(y_true, y_prob, threshold=0.5, n_bootstraps=1000):
    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cal_slope, cal_intercept = compute_calibration_metrics(y_true, y_prob)

    rng = np.random.default_rng(RANDOM_STATE)
    boot_auroc, boot_auprc, boot_brier = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        boot_auroc.append(roc_auc_score(y_true[idx], y_prob[idx]))
        boot_auprc.append(average_precision_score(y_true[idx], y_prob[idx]))
        boot_brier.append(brier_score_loss(y_true[idx], y_prob[idx]))

    return {
        "AUROC": float(auroc),
        "AUROC_95CI": [float(np.percentile(boot_auroc, 2.5)), float(np.percentile(boot_auroc, 97.5))],
        "AUPRC": float(auprc),
        "AUPRC_95CI": [float(np.percentile(boot_auprc, 2.5)), float(np.percentile(boot_auprc, 97.5))],
        "Brier": float(brier),
        "Brier_95CI": [float(np.percentile(boot_brier, 2.5)), float(np.percentile(boot_brier, 97.5))],
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "PPV": float(ppv),
        "NPV": float(npv),
        "F1": float(f1),
        "Balanced_Accuracy": float(bal_acc),
        "Calibration_Slope": float(cal_slope),
        "Calibration_Intercept": float(cal_intercept)
    }

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Initiating External Validation (Champion XGBoost)...")
    start_time = time.time()

    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    if not XGB_MODEL_FILE.exists():
        print(f"[ERROR] Locked XGBoost model not found at {XGB_MODEL_FILE}")
        return

    # ---------------------------------------------------------
    # 1. RECONSTRUCT SOURCE SCALERS FROM MIMIC-IV
    # ---------------------------------------------------------
    print("    -> Reconstructing original scalers from MIMIC-IV to prevent feature drift...")
    try:
        mimic_tensor = np.load(MIMIC_TENSOR_FILE)
        mimic_stay_ids = np.load(MIMIC_STAY_ID_FILE)
        mimic_train_idx = np.load(MIMIC_TRAIN_IDX_FILE)
        df_mimic = pl.read_parquet(MIMIC_COHORT_FILE).to_pandas()
    except Exception as e:
        print(f"[ERROR] Failed to load MIMIC-IV reference data. Error: {e}")
        return

    df_mimic = pd.DataFrame({"stay_id": mimic_stay_ids}).merge(df_mimic, on="stay_id", how="left")

    static_cols = ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"]
    df_mimic_static = df_mimic[static_cols].copy()
    if "gender" in df_mimic_static.columns and df_mimic_static["gender"].dtype == 'O':
        df_mimic_static["gender"] = (df_mimic_static["gender"] == "M").astype(int)

    X_mimic_static_raw = df_mimic_static.fillna(0).values

    # Fit Static Scaler EXACTLY as in training script (on train_val only)
    scaler_static = StandardScaler()
    scaler_static.fit(X_mimic_static_raw[mimic_train_idx])

    # Fit Temporal Scaler EXACTLY as in training script (on full dataset)
    mimic_mean = np.mean(mimic_tensor, axis=1)
    mimic_min = np.min(mimic_tensor, axis=1)
    mimic_max = np.max(mimic_tensor, axis=1)
    mimic_std = np.std(mimic_tensor, axis=1)
    X_mimic_temporal_raw = np.concatenate([mimic_mean, mimic_min, mimic_max, mimic_std], axis=1)

    scaler_temporal = StandardScaler()
    scaler_temporal.fit(X_mimic_temporal_raw)

    # ---------------------------------------------------------
    # 2. EXTRACT AND SCALE TARGET DATA (eICU)
    # ---------------------------------------------------------
    print("    -> Extracting and scaling eICU external validation dataset...")
    try:
        eicu_tensor = np.load(EICU_TENSOR_FILE)
        eicu_stay_ids = np.load(EICU_STAY_ID_FILE)
        df_eicu = pl.read_parquet(EICU_COHORT_FILE).to_pandas()
    except Exception as e:
        print(f"[ERROR] Failed to load eICU validation data. Error: {e}")
        return

    df_eicu = pd.DataFrame({"stay_id": eicu_stay_ids}).merge(df_eicu, on="stay_id", how="left")
    y_test = df_eicu["hospital_expire_flag"].values

    # Static eICU
    df_eicu_static = df_eicu[static_cols].copy()
    if "gender" in df_eicu_static.columns and df_eicu_static["gender"].dtype == 'O':
        df_eicu_static["gender"] = (df_eicu_static["gender"] == "M").astype(int)

    X_eicu_static_raw = df_eicu_static.fillna(0).values
    X_eicu_static_scaled = scaler_static.transform(X_eicu_static_raw)

    # Temporal eICU
    eicu_mean = np.mean(eicu_tensor, axis=1)
    eicu_min = np.min(eicu_tensor, axis=1)
    eicu_max = np.max(eicu_tensor, axis=1)
    eicu_std = np.std(eicu_tensor, axis=1)
    X_eicu_temporal_raw = np.concatenate([eicu_mean, eicu_min, eicu_max, eicu_std], axis=1)
    X_eicu_temporal_scaled = scaler_temporal.transform(X_eicu_temporal_raw)

    # Final Fused eICU Tensor
    X_test = np.concatenate([X_eicu_static_scaled, X_eicu_temporal_scaled], axis=1)

    print(f"       - eICU Cohort Size: {len(y_test):,} patients")
    print(f"       - eICU Mortality Rate: {y_test.mean() * 100:.2f}%")
    print(f"       - Final Feature Vector Shape: {X_test.shape}")

    # ---------------------------------------------------------
    # 3. INFERENCE & EVALUATION
    # ---------------------------------------------------------
    print("    -> Loading locked MIMIC-IV XGBoost model (Inference Mode)...")
    champion_xgb = joblib.load(XGB_MODEL_FILE)

    print(f"    -> Running Evaluation & {N_BOOTSTRAPS}-Iteration Bootstrap...")
    preds = champion_xgb.predict_proba(X_test)[:, 1]

    # Save Predictions
    df_preds = pd.DataFrame({
        "stay_id": eicu_stay_ids,
        "true_label": y_test,
        "pred_probability": preds,
        "pred_label": (preds >= 0.5).astype(int)
    })
    df_preds.to_csv(PREDS_FILE, index=False)

    # Calculate Metrics
    metrics = evaluate_champion(y_test, preds, n_bootstraps=N_BOOTSTRAPS)

    with open(METRICS_FILE, "w") as f:
        json.dump({"model": "Champion_XGBoost_External_Validation", "metrics": metrics}, f, indent=4)

    print("\n" + "="*60)
    print(" EXTERNAL VALIDATION (eICU) PERFORMANCE")
    print("="*60)
    print(f"    AUROC : {metrics['AUROC']:.4f}  [95% CI: {metrics['AUROC_95CI'][0]:.4f} - {metrics['AUROC_95CI'][1]:.4f}]")
    print(f"    AUPRC : {metrics['AUPRC']:.4f}  [95% CI: {metrics['AUPRC_95CI'][0]:.4f} - {metrics['AUPRC_95CI'][1]:.4f}]")
    print(f"    Brier : {metrics['Brier']:.4f}  [95% CI: {metrics['Brier_95CI'][0]:.4f} - {metrics['Brier_95CI'][1]:.4f}]")
    print("="*60)

    # ---------------------------------------------------------
    # 4. PUBLICATION PLOTS
    # ---------------------------------------------------------
    print("\n    -> Generating Publication-Quality ROC and PR Curves...")
    fpr, tpr, _ = roc_curve(y_test, preds)
    precision, recall, _ = precision_recall_curve(y_test, preds)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AUROC Curve
    axes[0].plot(fpr, tpr, color='#C44E52', lw=2, label=f'eICU Validation (AUROC = {metrics["AUROC"]:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12, weight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=12, weight='bold')
    axes[0].set_title('A. Receiver Operating Characteristic (ROC)', fontsize=14, weight='bold')
    axes[0].legend(loc="lower right", frameon=True)

    # AUPRC Curve
    baseline_pr = y_test.mean()
    axes[1].plot(recall, precision, color='#4C72B0', lw=2, label=f'eICU Validation (AUPRC = {metrics["AUPRC"]:.3f})')
    axes[1].plot([0, 1], [baseline_pr, baseline_pr], color='gray', linestyle='--', label=f'Baseline ({baseline_pr:.2f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=12, weight='bold')
    axes[1].set_ylabel('Precision', fontsize=12, weight='bold')
    axes[1].set_title('B. Precision-Recall Curve (PRC)', fontsize=14, weight='bold')
    axes[1].legend(loc="upper right", frameon=True)

    sns.despine()
    plt.tight_layout()
    plt.savefig(ROC_PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n[+] Success! External validation completed in {elapsed:.2f} seconds.")
    print(f"    -> Metrics saved to: {METRICS_FILE.relative_to(BASE_DIR)}")
    print(f"    -> Predictions saved to: {PREDS_FILE.relative_to(BASE_DIR)}")
    print(f"    -> Figure saved to: {ROC_PLOT_FILE.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()