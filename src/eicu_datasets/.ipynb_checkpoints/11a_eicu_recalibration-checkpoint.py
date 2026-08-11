"""
11a_eicu_recalibration.py

Phase 10: External Recalibration & Deep Metric Analysis
[FIX]: Eliminates data leak by calculating Optimal Decision Thresholds (Youden's J) 
       strictly on the 80% Calibration Set and applying them to the 20% Test Set.
"""

import warnings
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, roc_curve
)
from sklearn.calibration import calibration_curve
from scipy.special import logit

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
PREDS_FILE = BASE_DIR / "results" / "eicu_external_validation" / "eicu_champion_predictions.csv"
OUT_PLOT = BASE_DIR / "results" / "eicu_external_validation" / "eicu_calibration_curve.png"
OUT_METRICS = BASE_DIR / "results" / "eicu_external_validation" / "eicu_calibration_metrics.json"

RANDOM_STATE = 42

def find_optimal_threshold(y_true, y_prob):
    """Calculates Youden's J statistic optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def evaluate_model_metrics(y_true, y_prob, threshold):
    """Evaluates metrics using a strictly pre-defined (locked) threshold."""
    # 1. Core overall metrics
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    # 2. Calculate Threshold-Dependent Metrics using the locked threshold
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Brier": float(brier),
        "Optimal_Threshold": float(threshold),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "PPV": float(ppv),
        "F1": float(f1)
    }

def main():
    print("[*] Initiating Strict External Recalibration (Locked Optimal Thresholds)...")
    
    if not PREDS_FILE.exists():
        print(f"[ERROR] Could not find predictions at {PREDS_FILE}")
        return

    df_preds = pd.read_csv(PREDS_FILE)
    y_true = df_preds["true_label"].values
    y_prob_raw = df_preds["pred_probability"].values
    
    print("    -> Splitting eICU cohort (80% Calibration / 20% Test)...")
    y_true_cal, y_true_test, y_prob_cal, y_prob_test = train_test_split(
        y_true, y_prob_raw, test_size=0.20, random_state=RANDOM_STATE, stratify=y_true
    )
    
    baseline_brier = np.mean(y_true_test) * (1 - np.mean(y_true_test))
    
    # 1. Fit Calibrators on 80% Calibration Set
    print("    -> Fitting Platt Scaler & Isotonic Regression on 80% split...")
    eps = 1e-15
    logits_cal = logit(np.clip(y_prob_cal, eps, 1 - eps)).reshape(-1, 1)
    
    platt = LogisticRegression(random_state=RANDOM_STATE)
    platt.fit(logits_cal, y_true_cal)
    
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(y_prob_cal, y_true_cal)
    
    # 2. Get predictions on the Calibration Set to find thresholds
    y_prob_cal_platt = platt.predict_proba(logits_cal)[:, 1]
    y_prob_cal_iso = isotonic.transform(y_prob_cal)
    
    # 3. Find Optimal Thresholds strictly on Calibration Data
    print("    -> Locking optimal clinical thresholds using Calibration Data...")
    thresh_uncal = find_optimal_threshold(y_true_cal, y_prob_cal)
    thresh_platt = find_optimal_threshold(y_true_cal, y_prob_cal_platt)
    thresh_iso = find_optimal_threshold(y_true_cal, y_prob_cal_iso)
    
    # 4. Predict on 20% Test Set
    logits_test = logit(np.clip(y_prob_test, eps, 1 - eps)).reshape(-1, 1)
    y_prob_platt = platt.predict_proba(logits_test)[:, 1]
    y_prob_iso = isotonic.transform(y_prob_test)
    
    # 5. Evaluate on 20% Test Set using the locked thresholds
    metrics_uncal = evaluate_model_metrics(y_true_test, y_prob_test, thresh_uncal)
    metrics_platt = evaluate_model_metrics(y_true_test, y_prob_platt, thresh_platt)
    metrics_iso = evaluate_model_metrics(y_true_test, y_prob_iso, thresh_iso)
    
    print("\n=========================================================================================")
    print(" RECALIBRATION IMPACT ANALYSIS (20% eICU TEST SET) - LOCKED THRESHOLDS")
    print("=========================================================================================")
    print(f"    Metric          | Uncalibrated  | Platt Scaling | Isotonic Regression")
    print("    ---------------------------------------------------------------------------------")
    print(f"    AUROC           | {metrics_uncal['AUROC']:.4f}        | {metrics_platt['AUROC']:.4f}        | {metrics_iso['AUROC']:.4f}")
    print(f"    AUPRC           | {metrics_uncal['AUPRC']:.4f}        | {metrics_platt['AUPRC']:.4f}        | {metrics_iso['AUPRC']:.4f}")
    print(f"    Brier Score     | {metrics_uncal['Brier']:.4f}        | {metrics_platt['Brier']:.4f}        | {metrics_iso['Brier']:.4f}")
    print("    ---------------------------------------------------------------------------------")
    print(f"    Optimal Cutoff  | {metrics_uncal['Optimal_Threshold']:.4f}        | {metrics_platt['Optimal_Threshold']:.4f}        | {metrics_iso['Optimal_Threshold']:.4f}")
    print(f"    Sensitivity     | {metrics_uncal['Sensitivity']:.4f}        | {metrics_platt['Sensitivity']:.4f}        | {metrics_iso['Sensitivity']:.4f}")
    print(f"    Specificity     | {metrics_uncal['Specificity']:.4f}        | {metrics_platt['Specificity']:.4f}        | {metrics_iso['Specificity']:.4f}")
    print(f"    PPV (Precision) | {metrics_uncal['PPV']:.4f}        | {metrics_platt['PPV']:.4f}        | {metrics_iso['PPV']:.4f}")
    print(f"    F1 Score        | {metrics_uncal['F1']:.4f}        | {metrics_platt['F1']:.4f}        | {metrics_iso['F1']:.4f}")
    print("    ---------------------------------------------------------------------------------")
    print(f"    Statistical Baseline Brier: {baseline_brier:.4f}")
    
    # Save Plot
    plt.figure(figsize=(9, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true_test, y_prob_test, n_bins=10, strategy='quantile')
    plt.plot(prob_pred_uncal, prob_true_uncal, "s-", color="red", alpha=0.7, label=f"Uncalibrated (Brier: {metrics_uncal['Brier']:.3f})")
    
    prob_true_platt, prob_pred_platt = calibration_curve(y_true_test, y_prob_platt, n_bins=10, strategy='quantile')
    plt.plot(prob_pred_platt, prob_true_platt, "o-", color="blue", alpha=0.8, label=f"Platt Scaling (Brier: {metrics_platt['Brier']:.3f})")
    
    prob_true_iso, prob_pred_iso = calibration_curve(y_true_test, y_prob_iso, n_bins=10, strategy='quantile')
    plt.plot(prob_pred_iso, prob_true_iso, "^-", color="green", alpha=0.8, label=f"Isotonic (Brier: {metrics_iso['Brier']:.3f})")
    
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives (Observed Rate)", fontsize=12)
    plt.title("Reliability Diagram: External Validation on eICU\nBefore and After Strict External Recalibration", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)
    print("\n[*] Calibration pipeline completed successfully.")

if __name__ == "__main__":
    main()