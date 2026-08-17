"""
01_ml_baselines.py

Phase 1: ML Baselines
Establishes the traditional machine learning benchmarks for mortality prediction.
Evaluates Logistic Regression (Static), Logistic Regression (Combined), 
Random Forest (Combined), and XGBoost (Combined).

Engineered to be a reusable foundation:
- Exports trained model objects (.joblib) for downstream reuse.
- Exports exact train/test indices and stay_ids to enforce reproducibility across all scripts.
- Exports raw prediction arrays.
- Computes comprehensive metrics (Bootstrapped CIs, Confusion Matrix, Calibration).
"""

import time
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.special import logit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, balanced_accuracy_score, precision_score
)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & DIRECTORIES
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

# Structured outputs based on the flattened artifact paradigm
OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_PREDS = BASE_DIR / "outputs" / "predictions"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FEATS = BASE_DIR / "outputs" / "features"

for d in [OUT_MODELS, OUT_PREDS, OUT_METRICS, OUT_FEATS]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAPS = 100

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def compute_calibration_metrics(y_true, y_prob):
    """Computes calibration slope and intercept using logistic regression on logits."""
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    logits = logit(y_prob_clipped).reshape(-1, 1)
    
    # Fit logistic regression on the log odds to find calibration curve
    lr = LogisticRegression(random_state=RANDOM_STATE)
    lr.fit(logits, y_true)
    
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]
    return slope, intercept

def evaluate_model(y_true, y_prob, threshold=0.5, n_bootstraps=100):
    """Shared evaluation function generating all requested clinical metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # Core Metrics
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    
    # Confusion Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Calibration Metrics
    cal_slope, cal_intercept = compute_calibration_metrics(y_true, y_prob)
    
    # Bootstrapping for Core Metrics
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
    print("[*] Initiating Phase 1: Comprehensive ML Baselines...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    print("    -> Loading imputed tensor and static cohort metadata...")
    X_imputed = np.load(PROCESSED_DIR / "mimic_sepsis_imputed_tensor.npy")
    stay_ids = np.load(PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy")
    tensor_features = list(np.load(PROCESSED_DIR / "mimic_sepsis_tensor_features.npy"))
    
    df_cohort = pl.read_parquet(PROCESSED_DIR / "mimic_final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    # ---------------------------------------------------------
    # 2. EXTRACT FEATURES & EXPORT NAMES
    # ---------------------------------------------------------
    potential_statics = ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"]
    static_cols = [col for col in potential_statics if col in df_cohort.columns]
    
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
        
    X_static = StandardScaler().fit_transform(df_static.fillna(0).values)
    
    # Aggregated Temporal Features
    print("    -> Flattening temporal tensor (Mean, Min, Max, Std)...")
    X_mean = np.mean(X_imputed, axis=1)
    X_min = np.min(X_imputed, axis=1)
    X_max = np.max(X_imputed, axis=1)
    X_std = np.std(X_imputed, axis=1)
    X_temporal_agg = StandardScaler().fit_transform(np.concatenate([X_mean, X_min, X_max, X_std], axis=1))
    
    # Feature Names
    agg_names = []
    for stat in ["Mean", "Min", "Max", "Std"]:
        for feat in tensor_features:
            agg_names.append(f"{feat}_{stat}")
            
    combined_names = static_cols + agg_names
    
    # Save Feature Names
    with open(OUT_FEATS / "mimic_static_features.json", "w") as f: json.dump(static_cols, f)
    with open(OUT_FEATS / "mimic_aggregated_features.json", "w") as f: json.dump(agg_names, f)
    with open(OUT_FEATS / "mimic_combined_features.json", "w") as f: json.dump(combined_names, f)

    X_combined = np.concatenate([X_static, X_temporal_agg], axis=1)

    # ---------------------------------------------------------
    # 3. SPLITS & PREVALENCE
    # ---------------------------------------------------------
    print("    -> Splitting data and saving exact indices...")
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.15, random_state=RANDOM_STATE, stratify=y)
    
    # Save Splits to the models directory (used across pipeline to guarantee same test sets)
    np.save(OUT_MODELS / "mimic_train_indices.npy", idx_train)
    np.save(OUT_MODELS / "mimic_test_set_indices.npy", idx_test)
    np.save(OUT_MODELS / "mimic_stay_ids_train.npy", stay_ids[idx_train])
    np.save(OUT_MODELS / "mimic_stay_ids_test.npy", stay_ids[idx_test])
    
    X_stat_train, X_stat_test = X_static[idx_train], X_static[idx_test]
    X_comb_train, X_comb_test = X_combined[idx_train], X_combined[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    stay_ids_test = stay_ids[idx_test]
    
    print(f"        - Training Cohort: {len(y_train)} patients | Mortality: {(sum(y_train)/len(y_train))*100:.1f}%")
    print(f"        - Testing Cohort:  {len(y_test)} patients | Mortality: {(sum(y_test)/len(y_test))*100:.1f}%")

    scale_weight = float((len(y_train) - sum(y_train)) / sum(y_train))

    # ---------------------------------------------------------
    # 4. INITIALIZE & TRAIN MODELS
    # ---------------------------------------------------------
    print("\n    -> Training and Evaluating Models...")
    results_list = []
    full_metrics_dict = {}

    models_config = {
        "LR_Static": (LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000), X_stat_train, X_stat_test),
        "LR_Combined": (LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000), X_comb_train, X_comb_test),
        "RandomForest_Combined": (RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE), X_comb_train, X_comb_test)
    }

    # Standard Sklearn Models
    for name, (model, X_tr, X_te) in models_config.items():
        print(f"        - Fitting {name}...")
        t0 = time.time()
        model.fit(X_tr, y_train)
        preds = model.predict_proba(X_te)[:, 1]
        
        # Save Model and Predictions
        joblib.dump(model, OUT_MODELS / f"mimic_{name}.joblib")
        pd.DataFrame({"stay_id": stay_ids_test, "true_label": y_test, "pred_probability": preds, "pred_label": (preds >= 0.5).astype(int)}).to_csv(OUT_PREDS / f"mimic_{name}_predictions.csv", index=False)
        
        metrics = evaluate_model(y_test, preds, n_bootstraps=N_BOOTSTRAPS)
        
        full_metrics_dict[name] = {
            "model": name,
            "Execution_Time_sec": round(time.time() - t0, 1),
            "Train_Size": len(y_train),
            "Test_Size": len(y_test),
            "Test_Positives": int(sum(y_test)),
            "Test_Negatives": int(len(y_test) - sum(y_test)),
            **metrics
        }
        results_list.append({"Model": name, "AUROC": metrics["AUROC"], "AUPRC": metrics["AUPRC"], "Brier": metrics["Brier"]})

    # XGBoost with Early Stopping
    print("        - Fitting XGBoost_Combined (with early stopping)...")
    t0 = time.time()
    X_xgb_tr, X_xgb_val, y_xgb_tr, y_xgb_val = train_test_split(X_comb_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train)
    
    xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, scale_pos_weight=scale_weight, eval_metric="aucpr", early_stopping_rounds=30, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_model.fit(X_xgb_tr, y_xgb_tr, eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)
    
    preds_xgb = xgb_model.predict_proba(X_comb_test)[:, 1]
    
    joblib.dump(xgb_model, OUT_MODELS / "mimic_XGBoost_Combined.joblib")
    pd.DataFrame({"stay_id": stay_ids_test, "true_label": y_test, "pred_probability": preds_xgb, "pred_label": (preds_xgb >= 0.5).astype(int)}).to_csv(OUT_PREDS / "mimic_XGBoost_Combined_predictions.csv", index=False)
    
    xgb_metrics = evaluate_model(y_test, preds_xgb, n_bootstraps=N_BOOTSTRAPS)
    full_metrics_dict["XGBoost_Combined"] = {
        "model": "XGBoost_Combined",
        "Execution_Time_sec": round(time.time() - t0, 1),
        "Train_Size": len(y_train),
        "Test_Size": len(y_test),
        "Test_Positives": int(sum(y_test)),
        "Test_Negatives": int(len(y_test) - sum(y_test)),
        **xgb_metrics
    }
    results_list.append({"Model": "XGBoost_Combined", "AUROC": xgb_metrics["AUROC"], "AUPRC": xgb_metrics["AUPRC"], "Brier": xgb_metrics["Brier"]})

    # ---------------------------------------------------------
    # 5. EXPORT & DISPLAY LEADERBOARD
    # ---------------------------------------------------------
    with open(OUT_METRICS / "mimic_detailed_baseline_metrics.json", "w") as f:
        json.dump(full_metrics_dict, f, indent=4)
        
    df_results = pd.DataFrame(results_list).sort_values(by="AUROC", ascending=False)
    df_results.to_csv(OUT_METRICS / "mimic_baseline_results_summary.csv", index=False)

    print("\n" + "="*65)
    print(" BASELINE MODEL PERFORMANCE (TEST SET)")
    print("="*65)
    print(f" {'Model':<25} | {'AUROC':<7} | {'AUPRC':<7} | {'Brier':<7}")
    print("-" * 65)
    for _, row in df_results.iterrows():
        print(f" {row['Model']:<25} | {row['AUROC']:.4f}  | {row['AUPRC']:.4f}  | {row['Brier']:.4f}")
    print("="*65)
    
    elapsed = time.time() - start_time
    print(f"[*] Benchmarking completed in {elapsed:.1f} seconds.")
    print(f"    -> Exported models, predictions, feature names, and split indices to flat outputs/ directories.")

if __name__ == "__main__":
    main()