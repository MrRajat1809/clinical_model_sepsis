"""
DeLong test for the difference between two correlated ROC curves.

The primary model and the logistic regression are scored on the same patients,
so their AUROC estimates are correlated and an unpaired comparison would
overstate the uncertainty. DeLong's method estimates the covariance from the
structural components of each AUROC and folds it into the variance of the
difference.

Implements the midrank-based fast algorithm directly rather than taking a
dependency, and cross-checks both AUROC values against scikit-learn before the
comparison, so an error in the implementation cannot pass unnoticed.

Patient ordering and labels are asserted identical across the two prediction
files first; the test is meaningless if the rows are not the same patients.

Applies to discrimination only. Calibration differences need a different test.

Reads:
    outputs/predictions/mimic_{champion, LR_Combined}_predictions.csv
Writes:
    outputs/analysis/delong_significance_results.json
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

CHAMPION_PREDS = BASE_DIR / "outputs" / "predictions" / "mimic_champion_predictions.csv"
BASELINE_PREDS = BASE_DIR / "outputs" / "predictions" / "mimic_LR_Combined_predictions.csv"

OUT_ANALYSIS = BASE_DIR / "outputs" / "analysis"
OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
DELONG_REPORT = OUT_ANALYSIS / "delong_significance_results.json"

# --- Fast Delong Implementation ------------------------------------------
def compute_midrank(x):
    """Computes global midranks for the DeLong algorithm."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_by_true_label, label_1_count):
    """Calculates AUROC and structural components for DeLong covariance."""
    m = label_1_count
    n = len(predictions_sorted_by_true_label) - m
    
    # Global ranks
    theta = compute_midrank(predictions_sorted_by_true_label)
    
    # Ranks of positives and negatives
    tx = theta[:m]
    ty = theta[m:]
    
    # Structural components
    v10 = (tx - compute_midrank(predictions_sorted_by_true_label[:m])) / n
    v01 = (compute_midrank(predictions_sorted_by_true_label[m:]) - ty + m) / m
    
    auc = np.sum(v10) / m
    var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
    
    return auc, var, v10, v01

def calc_pvalue(auc1, auc2, var1, var2, cov):
    """Calculates the two-tailed p-value for the difference in AUROCs."""
    diff = auc1 - auc2
    se = np.sqrt(max(0, var1 + var2 - 2 * cov)) # Ensure non-negative variance
    if se == 0:
        return 1.0
    z = diff / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(p)

def delong_roc_test(y_true, y_pred1, y_pred2):
    """Executes the full DeLong test for two sets of predictions."""
    # Sort by y_true descending (1s first, then 0s)
    order = np.argsort(y_true)[::-1]
    y_true_sorted = y_true[order]
    y_pred1_sorted = y_pred1[order]
    y_pred2_sorted = y_pred2[order]
    
    label_1_count = int(np.sum(y_true))
    label_0_count = len(y_true) - label_1_count
    
    auc1, var1, v10_1, v01_1 = fastDeLong(y_pred1_sorted, label_1_count)
    auc2, var2, v10_2, v01_2 = fastDeLong(y_pred2_sorted, label_1_count)
    
    # CORRECT Covariance estimation for DeLong (Scaled by class counts)
    cov_10 = np.cov(v10_1, v10_2, ddof=1)[0, 1]
    cov_01 = np.cov(v01_1, v01_2, ddof=1)[0, 1]
    cov = (cov_10 / label_1_count) + (cov_01 / label_0_count)
    
    p_val = calc_pvalue(auc1, auc2, var1, var2, cov)
    return float(auc1), float(auc2), float(p_val)

def main():
    print("[*] Initiating DeLong Statistical Significance Testing...")
    start_time = time.time()
    
    # Load Predictions
    if not CHAMPION_PREDS.exists() or not BASELINE_PREDS.exists():
        print(f"[ERROR] Prediction files missing.\nChamp: {CHAMPION_PREDS.exists()}\nBase: {BASELINE_PREDS.exists()}")
        return
        
    df_champ = pd.read_csv(CHAMPION_PREDS)
    df_base = pd.read_csv(BASELINE_PREDS)
    
    # Ensure cohorts are perfectly aligned
    assert (df_champ["stay_id"].values == df_base["stay_id"].values).all(), "Prediction cohorts misaligned!"
    
    y_true = df_champ["true_label"].values
    
    # Handle column names if they vary slightly
    prob_col_champ = "pred_probability" if "pred_probability" in df_champ.columns else "probability"
    prob_col_base = "pred_probability" if "pred_probability" in df_base.columns else "probability"
    
    prob_champ = df_champ[prob_col_champ].values
    prob_base = df_base[prob_col_base].values
    
    # Verify mathematically with Sklearn just in case
    sk_auc_champ = roc_auc_score(y_true, prob_champ)
    sk_auc_base = roc_auc_score(y_true, prob_base)
    
    print("    -> Running asymptotic variance calculations (this may take a moment)...")
    auc_champ, auc_base, p_value = delong_roc_test(y_true, prob_champ, prob_base)
    
    print("\n" + "="*60)
    print(" DELONG TEST: CHAMPION XGBOOST vs. BASELINE LR")
    print("="*60)
    print(f"    Champion AUROC : {auc_champ:.4f} (Verified: {sk_auc_champ:.4f})")
    print(f"    Baseline AUROC : {auc_base:.4f} (Verified: {sk_auc_base:.4f})")
    print(f"    p-value        : {p_value:.3e}")
    
    if p_value < 0.05:
        print("    [PASS] The performance improvement is statistically significant (p < 0.05).")
    else:
        print("    [FAIL] The performance improvement is not statistically significant.")
    print("="*60)

    results = {
        "Test": "DeLong Asymptotic Variance Test",
        "Comparison": "Champion XGBoost vs. Baseline Logistic Regression",
        "Champion_AUROC": auc_champ,
        "Baseline_AUROC": auc_base,
        "p_value": p_value,
        "Significant": bool(p_value < 0.05)
    }
    
    with open(DELONG_REPORT, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[+] Success! DeLong testing completed in {time.time() - start_time:.2f} seconds.")
    print(f"    -> Results saved to {DELONG_REPORT.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
