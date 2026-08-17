"""
04a_champion_xgboost.py

Phase 3: Train the Final Champion Model
Trains and tunes the definitively selected architecture: XGBoost on Static + Aggregated features.
- Excludes Deep Learning (BiGRU) embeddings based on ablation results.
- Runs Optuna Bayesian Optimization for exact hyperparameter tuning.
- Exports the locked champion model, raw predictions, and comprehensive bootstrap metrics 
  for downstream Calibration and SHAP analysis.
"""

import time
import json
import joblib
import random
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import optuna
from optuna.samplers import TPESampler
from scipy.special import logit

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, balanced_accuracy_score, precision_score
)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# CONFIGURATION & REPRODUCIBILITY
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
N_TRIALS = 30
N_BOOTSTRAPS = 1000  # Increased for the final champion evaluation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ==========================================
# EVALUATION HELPERS
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
    set_seed(RANDOM_STATE)
    print("[*] Initiating Phase 4: Tuning & Training Champion Model (Static + Aggregated)...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD DATA & SPLITS
    # ---------------------------------------------------------
    print("    -> Loading shared tensor, cohort metadata, and exact split indices...")
    X_imputed = np.load(PROCESSED_DIR / "mimic_sepsis_imputed_tensor.npy")
    stay_ids = np.load(PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy")
    tensor_features = list(np.load(PROCESSED_DIR / "mimic_sepsis_tensor_features.npy"))
    
    df_cohort = pl.read_parquet(PROCESSED_DIR / "mimic_final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    idx_train_val = np.load(OUT_MODELS / "mimic_train_indices.npy")
    idx_test = np.load(OUT_MODELS / "mimic_test_set_indices.npy")
    stay_ids_test = np.load(OUT_MODELS / "mimic_stay_ids_test.npy")

    # ---------------------------------------------------------
    # 2. EXTRACT & SCALE FEATURES
    # ---------------------------------------------------------
    potential_statics = ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"]
    static_cols = [col for col in potential_statics if col in df_cohort.columns]
    
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
        
    X_static_raw = df_static.fillna(0).values
    
    # Scale static features using ONLY the train_val set
    scaler_static = StandardScaler()
    scaler_static.fit(X_static_raw[idx_train_val])
    X_static = scaler_static.transform(X_static_raw)
    
    # Aggregated Temporal Features
    print("    -> Flattening temporal tensor (Mean, Min, Max, Std)...")
    X_mean = np.mean(X_imputed, axis=1)
    X_min = np.min(X_imputed, axis=1)
    X_max = np.max(X_imputed, axis=1)
    X_std = np.std(X_imputed, axis=1)
    
    X_temporal_agg = StandardScaler().fit_transform(np.concatenate([X_mean, X_min, X_max, X_std], axis=1))
    
    # Export Feature Names for SHAP
    agg_names = []
    for stat in ["Mean", "Min", "Max", "Std"]:
        for feat in tensor_features:
            agg_names.append(f"{feat}_{stat}")
            
    combined_names = static_cols + agg_names
    with open(OUT_FEATS / "mimic_champion_features.json", "w") as f: 
        json.dump(combined_names, f)

    # Final Fused Dataset
    X_fused = np.concatenate([X_static, X_temporal_agg], axis=1)
    X_train_val, y_train_val = X_fused[idx_train_val], y[idx_train_val]
    X_test, y_test = X_fused[idx_test], y[idx_test]

    scale_weight = float((len(y_train_val) - sum(y_train_val)) / sum(y_train_val))

    # ---------------------------------------------------------
    # 3. BAYESIAN HYPERPARAMETER OPTIMIZATION
    # ---------------------------------------------------------
    print(f"\n    -> Running {N_TRIALS} Optuna Trials (3-Fold CV)...")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": scale_weight,
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        
        for train_idx, val_idx in cv.split(X_train_val, y_train_val):
            model = XGBClassifier(**params)
            
            # Added eval_set and early_stopping_rounds
            model.fit(
                X_train_val[train_idx], y_train_val[train_idx],
                eval_set=[(X_train_val[val_idx], y_train_val[val_idx])],
                early_stopping_rounds=50,
                verbose=False
            )
            
            preds = model.predict_proba(X_train_val[val_idx])[:, 1]
            fold_scores.append(roc_auc_score(y_train_val[val_idx], preds))
            
        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    
    import sys
    for i in range(N_TRIALS):
        study.optimize(objective, n_trials=1)
        sys.stdout.write(f"\r        - Trial {i+1}/{N_TRIALS} | Best CV AUROC: {study.best_value:.4f}")
        sys.stdout.flush()
    print()

    best_params = study.best_params
    best_params.update({"scale_pos_weight": scale_weight, "random_state": RANDOM_STATE, "n_jobs": -1})
    
    print("\n    [+] Optimal Hyperparameters Found:")
    for k, v in best_params.items():
        if k not in ["scale_pos_weight", "random_state", "n_jobs"]:
            print(f"        - {k}: {v}")

    # ---------------------------------------------------------
    # 4. TRAIN & EVALUATE FINAL CHAMPION
    # ---------------------------------------------------------
    print("\n    -> Training Final Champion Model on Full Train/Val Set...")
    champion_xgb = XGBClassifier(**best_params)
    champion_xgb.fit(X_train_val, y_train_val)
    
    # Save Model
    joblib.dump(champion_xgb, OUT_MODELS / "mimic_champion_xgboost.joblib")
    
    # Inference
    print(f"    -> Running Evaluation & {N_BOOTSTRAPS}-Iteration Bootstrap...")
    preds = champion_xgb.predict_proba(X_test)[:, 1]
    
    # Save Predictions
    df_preds = pd.DataFrame({
        "stay_id": stay_ids_test,
        "true_label": y_test,
        "pred_probability": preds,
        "pred_label": (preds >= 0.5).astype(int)
    })
    df_preds.to_csv(OUT_PREDS / "mimic_champion_predictions.csv", index=False)
    
    # Export Metrics
    metrics = evaluate_champion(y_test, preds, n_bootstraps=N_BOOTSTRAPS)
    
    out_dict = {
        "model": "Champion_XGBoost_Static_Aggregated",
        "hyperparameters": {k: v for k, v in best_params.items() if k not in ["scale_pos_weight", "random_state", "n_jobs"]},
        "metrics": metrics
    }
    
    with open(OUT_METRICS / "mimic_champion_metrics.json", "w") as f:
        json.dump(out_dict, f, indent=4)
        
    print("\n" + "="*60)
    print(" FINAL CHAMPION MODEL PERFORMANCE (TEST SET)")
    print("="*60)
    print(f"    AUROC : {metrics['AUROC']:.4f}  [95% CI: {metrics['AUROC_95CI'][0]:.4f} - {metrics['AUROC_95CI'][1]:.4f}]")
    print(f"    AUPRC : {metrics['AUPRC']:.4f}  [95% CI: {metrics['AUPRC_95CI'][0]:.4f} - {metrics['AUPRC_95CI'][1]:.4f}]")
    print(f"    Brier : {metrics['Brier']:.4f}  [95% CI: {metrics['Brier_95CI'][0]:.4f} - {metrics['Brier_95CI'][1]:.4f}]")
    print("="*60)
    
    elapsed = time.time() - start_time
    print(f"[*] Pipeline completed in {elapsed/60:.1f} minutes.")

if __name__ == "__main__":
    main()