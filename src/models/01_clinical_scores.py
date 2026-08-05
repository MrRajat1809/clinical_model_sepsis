"""
01_clinical_scores.py

Phase 1: Establish Baselines
Establishes the absolute clinical baseline for mortality prediction.
Evaluates standard clinical markers (SOFA alone, Age + SOFA) using Logistic Regression.
Ensures evaluation on the exact same test patient indices as all other models.
Exports predictions, coefficients, curve coordinates, and comprehensive metadata.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

COHORT_DIR = BASE_DIR / "data" / "processed" / "mimiciv"
METRICS_OUT_DIR = BASE_DIR / "outputs" / "metrics"
PRED_OUT_DIR = BASE_DIR / "outputs" / "predictions"
CURVE_OUT_DIR = BASE_DIR / "outputs" / "curves"

RANDOM_STATE = 42

def save_curves(y_true, preds, model_name):
    """Saves ROC and PR curve coordinates for later plotting."""
    fpr, tpr, _ = roc_curve(y_true, preds)
    prec, rec, _ = precision_recall_curve(y_true, preds)
    
    # Save ROC
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(CURVE_OUT_DIR / f"{model_name}_roc.csv", index=False)
    # Save PR
    pd.DataFrame({"precision": prec, "recall": rec}).to_csv(CURVE_OUT_DIR / f"{model_name}_pr.csv", index=False)

def main():
    print("[*] Initiating Phase 1: Clinical Score Baselines (SOFA, Age + SOFA)...")
    METRICS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD COHORT & TEST INDICES
    # ---------------------------------------------------------
    print("    -> Loading cohort metadata and synchronizing test indices...")
    df_cohort = pl.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet").to_pandas()
    y = df_cohort["hospital_expire_flag"].values
    stay_ids = df_cohort["stay_id"].values
    
    test_indices_file = METRICS_OUT_DIR / "test_set_indices.npy"
    if not test_indices_file.exists():
        print(f"[ERROR] Test indices not found at {test_indices_file}.")
        return
        
    idx_test = np.load(test_indices_file)
    idx_train_val = np.setdiff1d(np.arange(len(y)), idx_test)
    
    y_train = y[idx_train_val]
    y_test = y[idx_test]
    stay_ids_test = stay_ids[idx_test]

    # ---------------------------------------------------------
    # 2. EXTRACT FEATURES
    # ---------------------------------------------------------
    sofa_raw = df_cohort[["baseline_sofa"]].fillna(0).values
    age_sofa_raw = df_cohort[["age", "baseline_sofa"]].fillna(0).values
    
    scaler_sofa = StandardScaler()
    scaler_age_sofa = StandardScaler()
    
    sofa_train = scaler_sofa.fit_transform(sofa_raw[idx_train_val])
    sofa_test = scaler_sofa.transform(sofa_raw[idx_test])
    
    age_sofa_train = scaler_age_sofa.fit_transform(age_sofa_raw[idx_train_val])
    age_sofa_test = scaler_age_sofa.transform(age_sofa_raw[idx_test])
    
    # ---------------------------------------------------------
    # 3. TRAIN & EVALUATE SOFA ONLY
    # ---------------------------------------------------------
    print("    -> Evaluating Baseline 1: SOFA Score Only...")
    lr_sofa = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    lr_sofa.fit(sofa_train, y_train)
    preds_sofa = lr_sofa.predict_proba(sofa_test)[:, 1]
    
    sofa_auroc = roc_auc_score(y_test, preds_sofa)
    sofa_auprc = average_precision_score(y_test, preds_sofa)
    sofa_brier = brier_score_loss(y_test, preds_sofa)
    save_curves(y_test, preds_sofa, "clinical_sofa")

    sofa_coef = lr_sofa.coef_[0][0]
    sofa_intercept = lr_sofa.intercept_[0]

    # ---------------------------------------------------------
    # 4. TRAIN & EVALUATE AGE + SOFA
    # ---------------------------------------------------------
    print("    -> Evaluating Baseline 2: Age + SOFA Score...")
    lr_age_sofa = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    lr_age_sofa.fit(age_sofa_train, y_train)
    preds_age_sofa = lr_age_sofa.predict_proba(age_sofa_test)[:, 1]
    
    age_sofa_auroc = roc_auc_score(y_test, preds_age_sofa)
    age_sofa_auprc = average_precision_score(y_test, preds_age_sofa)
    age_sofa_brier = brier_score_loss(y_test, preds_age_sofa)
    save_curves(y_test, preds_age_sofa, "clinical_age_sofa")

    age_coef, age_sofa_coef = lr_age_sofa.coef_[0]
    age_sofa_intercept = lr_age_sofa.intercept_[0]

    # ---------------------------------------------------------
    # 5. EXPORT PREDICTIONS
    # ---------------------------------------------------------
    df_preds = pd.DataFrame({
        "stay_id": stay_ids_test,
        "true_label": y_test,
        "sofa_probability": preds_sofa,
        "age_sofa_probability": preds_age_sofa
    })
    df_preds.to_csv(PRED_OUT_DIR / "01_clinical_baselines_predictions.csv", index=False)

    # ---------------------------------------------------------
    # 6. EXPORT COMPREHENSIVE JSON
    # ---------------------------------------------------------
    metrics = {
        "metadata": {
            "n_test_patients": int(len(y_test)),
            "n_test_deaths": int(sum(y_test)),
            "random_seed": RANDOM_STATE,
            "script_version": "1.0"
        },
        "SOFA_Only": {
            "metrics": {
                "AUROC": float(sofa_auroc),
                "AUPRC": float(sofa_auprc),
                "Brier": float(sofa_brier)
            },
            "model_parameters": {
                "features": ["baseline_sofa"],
                "intercept": float(sofa_intercept),
                "coefficients": {"baseline_sofa": float(sofa_coef)},
                "odds_ratios": {"baseline_sofa": float(np.exp(sofa_coef))}
            }
        },
        "Age_SOFA": {
            "metrics": {
                "AUROC": float(age_sofa_auroc),
                "AUPRC": float(age_sofa_auprc),
                "Brier": float(age_sofa_brier)
            },
            "model_parameters": {
                "features": ["age", "baseline_sofa"],
                "intercept": float(age_sofa_intercept),
                "coefficients": {
                    "age": float(age_coef),
                    "baseline_sofa": float(age_sofa_coef)
                },
                "odds_ratios": {
                    "age": float(np.exp(age_coef)),
                    "baseline_sofa": float(np.exp(age_sofa_coef))
                }
            }
        }
    }
    
    with open(METRICS_OUT_DIR / "clinical_baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n" + "="*50)
    print(" ESTABLISHED CLINICAL SCORE PERFORMANCE")
    print("="*50)
    print(f" 1. SOFA Score Only")
    print(f"    - AUROC: {sofa_auroc:.3f} | AUPRC: {sofa_auprc:.3f} | Brier: {sofa_brier:.3f}")
    print(f"    - SOFA OR: {np.exp(sofa_coef):.3f}")
    print("-" * 50)
    print(f" 2. Age + SOFA Score")
    print(f"    - AUROC: {age_sofa_auroc:.3f} | AUPRC: {age_sofa_auprc:.3f} | Brier: {age_sofa_brier:.3f}")
    print(f"    - Age OR: {np.exp(age_coef):.3f} | SOFA OR: {np.exp(age_sofa_coef):.3f}")
    print("="*50)
    
    elapsed = time.time() - start_time
    print(f"[*] Benchmarking completed in {elapsed:.1f} seconds.")
    print("    -> Exported predictions, curves, and metadata JSON.")

if __name__ == "__main__":
    main()