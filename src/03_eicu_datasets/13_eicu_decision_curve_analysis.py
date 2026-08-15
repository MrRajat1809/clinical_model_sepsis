"""
13_eicu_decision_curve_analysis.py

Decision Curve Analysis (DCA) on External Validation Cohort
Evaluates the true clinical utility of the model by calculating Net Benefit.

Features included:
- Recreates the exact 80/20 Calibration/Test split for eICU.
- Applies Platt Scaling and Isotonic Regression.
- Computes Net Benefit for Uncalibrated, Platt, Isotonic, Treat All, and Treat None.
- Generates a TRIPOD-compliant DCA plot, marking the optimal clinical thresholds.
"""

import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve
from scipy.special import logit

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Flattened Global Outputs
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

# Files
PREDS_FILE = OUT_METRICS / "eicu_champion_predictions.csv"
OUT_PLOT = OUT_FIGURES / "eicu_dca_plot.png"
OUT_CSV = OUT_METRICS / "eicu_dca_summary.csv"

# Thresholds to evaluate for the continuous curve
THRESHOLDS = np.linspace(0.01, 0.70, 100) # Capped at 70% as >70% mortality threshold is rarely used

# Key clinical thresholds for the summary table
SUMMARY_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.45]
RANDOM_STATE = 42

def find_optimal_threshold(y_true, y_prob):
    """Calculates Youden's J statistic optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

def compute_net_benefit(y_true, y_prob, threshold):
    """
    Computes Net Benefit for a given threshold.
    Formula: (True Positives / N) - (False Positives / N) * (threshold / (1 - threshold))
    """
    if threshold >= 1.0 or threshold <= 0.0:
        return 0.0
        
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit

def main():
    print("[*] Initiating External Decision Curve Analysis (DCA)...")
    start_time = time.time()
    
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    
    if not PREDS_FILE.exists():
        print(f"[ERROR] Could not find predictions at {PREDS_FILE}")
        return
        
    # 1. Load & Split Data (Exact match to script 11a)
    df_preds = pd.read_csv(PREDS_FILE)
    y_true = df_preds["true_label"].values
    y_prob_raw = df_preds["pred_probability"].values
    
    print("    -> Recreating 80/20 eICU Split & Fitting Calibrators...")
    y_true_cal, y_true_test, y_prob_cal, y_prob_test = train_test_split(
        y_true, y_prob_raw, test_size=0.20, random_state=RANDOM_STATE, stratify=y_true
    )
    
    prevalence = np.mean(y_true_test)
    print(f"       - Test Cohort Size: {len(y_true_test)} | Sepsis Mortality Rate: {prevalence*100:.1f}%")

    # 2. Fit Calibrators
    eps = 1e-15
    logits_cal = logit(np.clip(y_prob_cal, eps, 1 - eps)).reshape(-1, 1)
    
    platt = LogisticRegression(random_state=RANDOM_STATE).fit(logits_cal, y_true_cal)
    isotonic = IsotonicRegression(out_of_bounds='clip').fit(y_prob_cal, y_true_cal)
    
    # 3. Predict & Find Optimal Thresholds
    logits_test = logit(np.clip(y_prob_test, eps, 1 - eps)).reshape(-1, 1)
    y_prob_platt = platt.predict_proba(logits_test)[:, 1]
    y_prob_iso = isotonic.transform(y_prob_test)
    
    opt_uncal = find_optimal_threshold(y_true_cal, y_prob_cal)
    opt_platt = find_optimal_threshold(y_true_cal, platt.predict_proba(logits_cal)[:, 1])
    
    models = {
        "Uncalibrated": y_prob_test,
        "Platt Scaling": y_prob_platt,
        "Isotonic": y_prob_iso
    }

    # 4. Calculate Net Benefit Arrays
    print("    -> Computing Net Benefit across clinical continuum...")
    dca_results = {"Threshold": THRESHOLDS}
    dca_results["Treat None"] = [0.0] * len(THRESHOLDS)
    dca_results["Treat All"] = [compute_net_benefit(y_true_test, np.ones_like(y_true_test), t) for t in THRESHOLDS]
    
    for name, probs in models.items():
        dca_results[name] = [compute_net_benefit(y_true_test, probs, t) for t in THRESHOLDS]
        
    df_dca = pd.DataFrame(dca_results)

    # 5. Generate Summary Table
    summary_data = []
    for t in SUMMARY_THRESHOLDS:
        row = {"Threshold": f"{t*100:.0f}%"}
        row["Treat All"] = compute_net_benefit(y_true_test, np.ones_like(y_true_test), t)
        row["Treat None"] = 0.0
        for name, probs in models.items():
            row[name] = compute_net_benefit(y_true_test, probs, t)
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(OUT_CSV, index=False)

    # 6. Plot Decision Curve
    print(f"    -> Generating Publication Plot at {OUT_PLOT.relative_to(BASE_DIR)}...")
    plt.figure(figsize=(10, 7))
    
    # Baselines
    plt.plot(THRESHOLDS, df_dca["Treat None"], color="black", linestyle="-", linewidth=2, label="Treat None (Net Benefit = 0)")
    plt.plot(THRESHOLDS, df_dca["Treat All"], color="gray", linestyle="--", linewidth=2, label="Treat All (Assume everyone dies)")
    
    # Models
    plt.plot(THRESHOLDS, df_dca["Uncalibrated"], color="red", linestyle=":", linewidth=2, alpha=0.8, label="Champion (Uncalibrated)")
    plt.plot(THRESHOLDS, df_dca["Isotonic"], color="green", linestyle="-.", linewidth=2, alpha=0.8, label="Champion (Isotonic)")
    plt.plot(THRESHOLDS, df_dca["Platt Scaling"], color="blue", linestyle="-", linewidth=3, label="Champion (Platt Scaling)")
    
    # Mark Optimal Thresholds
    plt.axvline(x=opt_platt, color="blue", linestyle="--", alpha=0.5, label=f"Platt Optimal (~{opt_platt*100:.0f}%)")
    plt.axvline(x=opt_uncal, color="red", linestyle="--", alpha=0.3, label=f"Uncal Optimal (~{opt_uncal*100:.0f}%)")

    # Aesthetics
    plt.ylim([-0.05, prevalence + 0.05])
    plt.xlim([0.0, 0.60])
    
    plt.title("Decision Curve Analysis: eICU External Validation\nImpact of Recalibration on Clinical Utility", fontsize=14)
    plt.xlabel("Threshold Probability (Clinician's Risk Tolerance)", fontsize=12)
    plt.ylabel("Net Benefit", fontsize=12)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=300)

    # 7. Console Output
    print("\n==========================================================================================")
    print(" DECISION CURVE ANALYSIS (NET BENEFIT AT KEY THRESHOLDS)")
    print("==========================================================================================")
    header = f"{'Threshold':<10} | {'Treat All':<12} | {'Uncalibrated':<15} | {'Platt Scaling':<15} | {'Isotonic':<15}"
    print(header)
    print("-" * 88)
    for _, row in df_summary.iterrows():
        print(f" {row['Threshold']:<9} | {row['Treat All']:<12.4f} | {row['Uncalibrated']:<15.4f} | {row['Platt Scaling']:<15.4f} | {row['Isotonic']:<15.4f}")
    print("==========================================================================================")
    print(f"[*] Analysis completed in {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()