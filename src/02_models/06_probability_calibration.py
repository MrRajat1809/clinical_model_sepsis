"""
Internal probability calibration on MIMIC-IV.

Discrimination says whether the ranking is right; this asks whether the numbers
can be read as probabilities. Run before any external work so that calibration
drift observed in eICU can be attributed to the new environment rather than to a
model that was never calibrated to begin with.

Calibrators are fitted on out-of-fold predictions from five stratified folds of
the development partition, never on the model's own training predictions, which
would be optimistic. Two are compared: Platt scaling on the logit, and isotonic
regression on the probability.

Decision thresholds are chosen on the same out-of-fold predictions by Youden's J
and locked before the test set is touched, so neither the calibrator nor the
threshold sees the evaluation data.

Reports AUROC, AUPRC, Brier, calibration slope and intercept, and expected
calibration error over ten bins, with paired bootstrap differences in Brier.

Reads:
    outputs/models/mimic_champion_xgboost.joblib, the shared split indices
Writes:
    outputs/models/mimic_{platt, isotonic}_calibrator.joblib
    outputs/metrics/mimic_calibration_summary.json, curves and plot data
"""

import time
import json
import joblib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, confusion_matrix, precision_score, f1_score, balanced_accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings("ignore")

# --- Configuration & Reproducibility -------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_PREDS = BASE_DIR / "outputs" / "predictions"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"
OUT_PLOT_DATA = BASE_DIR / "outputs" / "plot_data"

for d in [OUT_MODELS, OUT_PREDS, OUT_METRICS, OUT_FIGURES, OUT_PLOT_DATA]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAPS = 1000

def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- Metric Helpers ------------------------------------------------------
def find_optimal_threshold(y_true, y_prob):
    """Calculates Youden's J statistic optimal threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return float(thresholds[optimal_idx])

def evaluate_threshold_metrics(y_true, y_prob, threshold):
    """Evaluates threshold-dependent clinical metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    return float(sensitivity), float(specificity), float(ppv), float(f1), float(bal_acc)

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Computes the Expected Calibration Error (ECE)."""
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        bin_idx = (binids == i)
        if np.sum(bin_idx) > 0:
            bin_prob = np.mean(y_prob[bin_idx])
            bin_acc = np.mean(y_true[bin_idx])
            ece += np.abs(bin_prob - bin_acc) * np.sum(bin_idx)
    return ece / len(y_true)

def compute_slope_intercept(y_true, y_prob):
    """Computes Calibration Slope and Intercept with strict float64 safeguards."""
    y_prob_64 = np.array(y_prob, dtype=np.float64)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob_64, eps, 1.0 - eps)
    
    logits = logit(y_prob_clipped).reshape(-1, 1)
    lr = LogisticRegression(random_state=RANDOM_STATE)
    lr.fit(logits, y_true)
    
    return lr.coef_[0][0], lr.intercept_[0]

# --- Main Execution ------------------------------------------------------
def main():
    set_seed(RANDOM_STATE)
    print("[*] Initiating Phase 5: Probability Calibration & Threshold Optimization...")
    start_time = time.time()
    
    # --- Load Champion Model & Splits ------------------------------------
    champion_model_file = OUT_MODELS / "mimic_champion_xgboost.joblib"
    if not champion_model_file.exists():
        print(f"[ERROR] Champion model not found at {champion_model_file}.")
        return
        
    print("    -> Loading locked Champion XGBoost model...")
    champion_xgb = joblib.load(champion_model_file)
    
    X_imputed = np.load(PROCESSED_DIR / "mimic_sepsis_imputed_tensor.npy")
    stay_ids = np.load(PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy")
    
    df_cohort = pl.read_parquet(PROCESSED_DIR / "mimic_final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    idx_train_val = np.load(OUT_MODELS / "mimic_train_indices.npy")
    idx_test = np.load(OUT_MODELS / "mimic_test_set_indices.npy")
    stay_ids_test = np.load(OUT_MODELS / "mimic_stay_ids_test.npy")

    # --- Extract & Scale Features ----------------------------------------
    print("    -> Constructing feature space...")
    static_cols = [col for col in ["age", "baseline_sofa"] if col in df_cohort.columns]
    df_static = df_cohort[static_cols].copy()
        
    scaler_static = StandardScaler().fit(df_static.fillna(0).values[idx_train_val])
    X_static = scaler_static.transform(df_static.fillna(0).values)

    X_mean, X_min = np.mean(X_imputed, axis=1), np.min(X_imputed, axis=1)
    X_max, X_std = np.max(X_imputed, axis=1), np.std(X_imputed, axis=1)
    
    scaler_agg = StandardScaler().fit(np.concatenate([X_mean, X_min, X_max, X_std], axis=1)[idx_train_val])
    X_temporal_agg = scaler_agg.transform(np.concatenate([X_mean, X_min, X_max, X_std], axis=1))

    X_fused = np.concatenate([X_static, X_temporal_agg], axis=1)
    
    X_train_val, y_train_val = X_fused[idx_train_val], y[idx_train_val]
    X_test, y_test = X_fused[idx_test], y[idx_test]

    # --- Out-of-fold (oof) Predictions for Unbiased Calibration ----------
    print("    -> Generating Out-of-Fold (OOF) predictions to prevent calibration data leakage...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preds_calib_uncal = cross_val_predict(
        champion_xgb, X_train_val, y_train_val, 
        cv=cv, method='predict_proba', n_jobs=-1
    )[:, 1]
    
    y_calib = y_train_val  # The true labels match the OOF predictions
    
    print("    -> Generating Champion predictions on the unseen test set...")
    preds_test_uncal = champion_xgb.predict_proba(X_test)[:, 1]

    # --- Fit Calibrators & Find Optimal Thresholds -----------------------
    print("    -> Fitting Platt Scaling (on logits) & Isotonic Regression...")
    eps = 1e-15
    logits_calib_uncal = logit(np.clip(preds_calib_uncal, eps, 1 - eps)).reshape(-1, 1)
    logits_test_uncal = logit(np.clip(preds_test_uncal, eps, 1 - eps)).reshape(-1, 1)

    platt_calibrator = LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE)
    platt_calibrator.fit(logits_calib_uncal, y_calib)
    
    preds_calib_platt = platt_calibrator.predict_proba(logits_calib_uncal)[:, 1]
    preds_test_platt = platt_calibrator.predict_proba(logits_test_uncal)[:, 1]

    iso_calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_calibrator.fit(preds_calib_uncal, y_calib)
    
    preds_calib_iso = iso_calibrator.predict(preds_calib_uncal)
    preds_test_iso = iso_calibrator.predict(preds_test_uncal)

    print("    -> Locking optimal clinical thresholds (Youden's J) using OOF Calibration Data...")
    thresh_uncal = find_optimal_threshold(y_calib, preds_calib_uncal)
    thresh_platt = find_optimal_threshold(y_calib, preds_calib_platt)
    thresh_iso = find_optimal_threshold(y_calib, preds_calib_iso)
    
    thresholds = {
        "Uncalibrated": thresh_uncal,
        "Platt": thresh_platt,
        "Isotonic": thresh_iso
    }

    # Export Calibrators
    joblib.dump(platt_calibrator, OUT_MODELS / "mimic_platt_calibrator.joblib")
    joblib.dump(iso_calibrator, OUT_MODELS / "mimic_isotonic_calibrator.joblib")

    # --- Paired Bootstrap Statistical Testing ----------------------------
    print(f"    -> Running {N_BOOTSTRAPS}-Iteration Bootstrap Evaluation...")
    rng = np.random.default_rng(RANDOM_STATE)
    
    models = {
        "Uncalibrated": preds_test_uncal,
        "Platt": preds_test_platt,
        "Isotonic": preds_test_iso
    }
    
    boot_results = {name: {"auroc": [], "auprc": [], "brier": []} for name in models.keys()}
    
    boot_diffs = {
        "Uncal_vs_Platt": [],
        "Uncal_vs_Iso": [],
        "Platt_vs_Iso": []
    }
    
    for _ in range(N_BOOTSTRAPS):
        idx = rng.choice(len(y_test), size=len(y_test), replace=True)
        y_b = y_test[idx]
        if len(np.unique(y_b)) < 2: continue
            
        for name, preds in models.items():
            p_b = preds[idx]
            boot_results[name]["auroc"].append(roc_auc_score(y_b, p_b))
            boot_results[name]["auprc"].append(average_precision_score(y_b, p_b))
            boot_results[name]["brier"].append(brier_score_loss(y_b, p_b))
            
        # Calculate reductions (positive = improvement)
        boot_diffs["Uncal_vs_Platt"].append(boot_results["Uncalibrated"]["brier"][-1] - boot_results["Platt"]["brier"][-1])
        boot_diffs["Uncal_vs_Iso"].append(boot_results["Uncalibrated"]["brier"][-1] - boot_results["Isotonic"]["brier"][-1])
        boot_diffs["Platt_vs_Iso"].append(boot_results["Platt"]["brier"][-1] - boot_results["Isotonic"]["brier"][-1])

    # --- Compile Comprehensive Metrics -----------------------------------
    summary_metrics = {}
    for name, preds in models.items():
        slope, intercept = compute_slope_intercept(y_test, preds)
        ece = expected_calibration_error(y_test, preds)
        sens, spec, ppv, f1, bal_acc = evaluate_threshold_metrics(y_test, preds, thresholds[name])
        
        summary_metrics[name] = {
            "AUROC": np.mean(boot_results[name]["auroc"]),
            "AUROC_CI": [np.percentile(boot_results[name]["auroc"], 2.5), np.percentile(boot_results[name]["auroc"], 97.5)],
            "AUPRC": np.mean(boot_results[name]["auprc"]),
            "AUPRC_CI": [np.percentile(boot_results[name]["auprc"], 2.5), np.percentile(boot_results[name]["auprc"], 97.5)],
            "Brier": np.mean(boot_results[name]["brier"]),
            "Brier_CI": [np.percentile(boot_results[name]["brier"], 2.5), np.percentile(boot_results[name]["brier"], 97.5)],
            "Optimal_Threshold": thresholds[name],
            "Sensitivity": sens,
            "Specificity": spec,
            "PPV": ppv,
            "F1": f1,
            "Balanced_Accuracy": bal_acc,
            "Slope": slope,
            "Intercept": intercept,
            "ECE": ece
        }

    # --- Export Predictions ----------------------------------------------
    print("    -> Exporting calibrated predictions...")
    df_preds = pd.DataFrame({
        "stay_id": stay_ids_test,
        "true_label": y_test,
        "uncalibrated": preds_test_uncal,
        "platt": preds_test_platt,
        "isotonic": preds_test_iso
    })
    df_preds.to_csv(OUT_PREDS / "mimic_calibrated_predictions.csv", index=False)

    # --- Export Calibration Plot Data & Figure ---------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    plot_data_curves = {}
    colors = {"Uncalibrated": "firebrick", "Platt": "steelblue", "Isotonic": "forestgreen"}
    
    for name, preds in models.items():
        prob_true, prob_pred = calibration_curve(y_test, preds, n_bins=10, strategy='quantile')
        plot_data_curves[f"{name}_prob_pred"] = prob_pred
        plot_data_curves[f"{name}_prob_true"] = prob_true
        
        ax1.plot(prob_pred, prob_true, marker='o', label=f"{name} (ECE: {summary_metrics[name]['ECE']:.3f})", color=colors[name])
        ax2.hist(preds, bins=50, alpha=0.5, label=name, color=colors[name], density=True)
        
    ax1.set_ylabel("True Probability in Bin")
    ax1.set_title("A. Calibration Curves")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.set_title("B. Probability Distribution")
    ax2.legend(loc="upper center")
    
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "mimic_calibration_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save curve data for external plotting
    pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in plot_data_curves.items() ])).to_csv(OUT_PLOT_DATA / "mimic_calibration_curve_data.csv", index=False)

    # --- Json Export & Print Results -------------------------------------
    with open(OUT_METRICS / "mimic_calibration_summary.json", "w") as f:
        json.dump({"Champion": "Static_Aggregated", "Metrics": summary_metrics}, f, indent=4)

    def print_diff(title, diff_arr):
        mean_diff = np.mean(diff_arr)
        ci = (np.percentile(diff_arr, 2.5), np.percentile(diff_arr, 97.5))
        sign = "+" if mean_diff > 0 else ""
        print(f"     {title:<20}: {sign}{mean_diff:.4f} [95% CI: {sign}{ci[0]:.4f} to {sign}{ci[1]:.4f}]")

    print("\n" + "="*80)
    print(" PROBABILITY CALIBRATION PERFORMANCE (TEST SET)")
    print("="*80)
    for name, m in summary_metrics.items():
        print(f" [{name}]")
        print(f"   AUROC: {m['AUROC']:.4f} | AUPRC: {m['AUPRC']:.4f} | Brier: {m['Brier']:.4f}")
        print(f"   Opt Cutoff: {m['Optimal_Threshold']:.4f} | Sens: {m['Sensitivity']:.4f}  | Spec: {m['Specificity']:.4f}")
        print(f"   Slope: {m['Slope']:>6.3f} | Int: {m['Intercept']:>6.3f}   | ECE: {m['ECE']:.4f}")
        print("-" * 40)
        
    print("\n [+] STATISTICAL TESTING (Brier Score Reduction)")
    print_diff("Uncal vs Platt", boot_diffs["Uncal_vs_Platt"])
    print_diff("Uncal vs Isotonic", boot_diffs["Uncal_vs_Iso"])
    print_diff("Platt vs Isotonic", boot_diffs["Platt_vs_Iso"])
    print("="*80)

    elapsed = time.time() - start_time
    print(f"[*] Calibration completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()
