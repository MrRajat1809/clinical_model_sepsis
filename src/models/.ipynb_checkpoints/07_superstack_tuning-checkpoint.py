"""
07_superstack_tuning.py

Executes Bayesian Hyperparameter Optimization (via Optuna) on the Super-Stack matrix.
Performs 3-Fold Stratified CV on the training data to find the optimal XGBoost parameters 
(optimizing for AUROC) before evaluating the final locked model on the hold-out test set.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

TENSOR_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors"
COHORT_DIR = BASE_DIR / "data" / "processed" / "mimiciv"
MODEL_OUT_DIR = BASE_DIR / "outputs" / "models" / "deep_learning"
METRICS_OUT_DIR = BASE_DIR / "outputs" / "metrics"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
N_TRIALS = 30  # Adjust higher (e.g., 50-100) for exhaustive tuning

class BiGRUEncoder(nn.Module):
    def __init__(self, temporal_dim=30, hidden_dim=64):
        super(BiGRUEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=temporal_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )

    def forward(self, x_temp):
        gru_out, _ = self.gru(x_temp)
        return torch.mean(gru_out, dim=1)

def main():
    print("[*] Initiating Optuna Hyperparameter Tuning for the Super-Stack...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD & ASSEMBLE DATA
    # ---------------------------------------------------------
    print("    -> Constructing fused Super-Stack feature matrix...")
    X_imputed = np.load(TENSOR_DIR / "sepsis_imputed_tensor.npy")
    stay_ids = np.load(TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    
    df_cohort = pl.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    idx_test = np.load(METRICS_OUT_DIR / "test_set_indices.npy")
    idx_train_val = np.setdiff1d(np.arange(len(y)), idx_test)
    
    # Static
    static_cols = [col for col in ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"] if col in df_cohort.columns]
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
    X_static = StandardScaler().fit_transform(df_static.fillna(0).values)

    # Aggregated
    X_mean = np.mean(X_imputed, axis=1)
    X_min = np.min(X_imputed, axis=1)
    X_max = np.max(X_imputed, axis=1)
    X_std = np.std(X_imputed, axis=1)
    X_temporal_agg = StandardScaler().fit_transform(np.concatenate([X_mean, X_min, X_max, X_std], axis=1))

    # Deep BiGRU Latents
    model_path = MODEL_OUT_DIR / "mimic_multimodal_bigru.pth"
    full_state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    encoder_state_dict = {k.replace('gru.', ''): v for k, v in full_state_dict.items() if k.startswith('gru.')}
    encoder = BiGRUEncoder().to(DEVICE)
    encoder.gru.load_state_dict(encoder_state_dict)
    encoder.eval()
    
    embedding_loader = DataLoader(torch.tensor(X_imputed, dtype=torch.float32), batch_size=256, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in embedding_loader:
            embeddings.append(encoder(batch.to(DEVICE)).cpu().numpy())
    X_temporal_emb = np.concatenate(embeddings, axis=0)

    # Fuse
    X_fused = np.concatenate([X_static, X_temporal_agg, X_temporal_emb], axis=1)
    X_train_val = X_fused[idx_train_val]
    y_train_val = y[idx_train_val]
    X_test = X_fused[idx_test]
    y_test = y[idx_test]

    scale_weight = float((len(y_train_val) - sum(y_train_val)) / sum(y_train_val))

    # ---------------------------------------------------------
    # 2. OPTUNA OBJECTIVE FUNCTION
    # ---------------------------------------------------------
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
            X_fold_train, X_fold_val = X_train_val[train_idx], X_train_val[val_idx]
            y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]
            
            model = XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict_proba(X_fold_val)[:, 1]
            fold_scores.append(roc_auc_score(y_fold_val, preds))
            
        return np.mean(fold_scores)

    # ---------------------------------------------------------
    # 3. RUN OPTIMIZATION
    # ---------------------------------------------------------
    print(f"    -> Running {N_TRIALS} Bayesian Optimization Trials (3-Fold CV)...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    
    # Add a progress bar for the study
    import sys
    for i in range(N_TRIALS):
        study.optimize(objective, n_trials=1)
        sys.stdout.write(f"\r       - Trial {i+1}/{N_TRIALS} | Best CV AUROC: {study.best_value:.4f}")
        sys.stdout.flush()
    print()

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_weight
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    
    print("\n    [+] Optimal Hyperparameters Found:")
    for k, v in best_params.items():
        if k not in ["scale_pos_weight", "random_state", "n_jobs"]:
            print(f"        - {k}: {v}")

    # ---------------------------------------------------------
    # 4. TRAIN & EVALUATE FINAL CHAMPION MODEL
    # ---------------------------------------------------------
    print("\n    -> Training Final Locked Model on Full Train/Val Set...")
    champion_xgb = XGBClassifier(**best_params)
    champion_xgb.fit(X_train_val, y_train_val)
    
    preds = champion_xgb.predict_proba(X_test)[:, 1]
    
    auroc = roc_auc_score(y_test, preds)
    auprc = average_precision_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    
    # Save parameters and metrics
    metrics_file = METRICS_OUT_DIR / "baseline_ml_metrics.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    metrics["Tuned_SuperStack"] = {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Brier": float(brier),
        "Hyperparameters": {k: v for k, v in best_params.items() if k not in ["scale_pos_weight", "random_state", "n_jobs"]}
    }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n" + "="*50)
    print(" TUNED SUPER-STACK PERFORMANCE (TEST SET)")
    print("="*50)
    print(f"    - AUROC: {auroc:.4f}")
    print(f"    - AUPRC: {auprc:.4f}")
    print(f"    - Brier: {brier:.4f}")
    print("="*50)
    
    elapsed = time.time() - start_time
    print(f"[*] Pipeline completed in {elapsed/60:.1f} minutes.")

if __name__ == "__main__":
    main()