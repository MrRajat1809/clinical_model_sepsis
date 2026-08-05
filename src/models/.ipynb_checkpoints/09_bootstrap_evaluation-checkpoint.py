"""
09_bootstrap_evaluation.py

Calculates 95% Confidence Intervals for the locked Champion Model (Tuned Super-Stack)
using 1,000 bootstrap iterations on the hold-out test set predictions.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

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
N_BOOTSTRAPS = 1000

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
    print(f"[*] Initiating 1,000-Iteration Bootstrap Analysis on Champion Model...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD OPTIMIZED HYPERPARAMETERS
    # ---------------------------------------------------------
    metrics_file = METRICS_OUT_DIR / "baseline_ml_metrics.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    best_params = metrics.get("Tuned_SuperStack", {}).get("Hyperparameters", {})
    if not best_params:
        print("[ERROR] Tuned hyperparameters not found. Run 07_superstack_tuning.py first.")
        return

    # ---------------------------------------------------------
    # 2. LOAD DATA & ASSEMBLE SUPER-STACK
    # ---------------------------------------------------------
    print("    -> Assembling Super-Stack feature matrix...")
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

    # BiGRU Latents
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

    # Fuse and Split
    X_fused = np.concatenate([X_static, X_temporal_agg, X_temporal_emb], axis=1)
    X_train_val, y_train_val = X_fused[idx_train_val], y[idx_train_val]
    X_test, y_test = X_fused[idx_test], y[idx_test]

    # ---------------------------------------------------------
    # 3. TRAIN CHAMPION & GENERATE PREDICTIONS
    # ---------------------------------------------------------
    print("    -> Training Tuned Champion Model...")
    scale_weight = float((len(y_train_val) - sum(y_train_val)) / sum(y_train_val))
    best_params["scale_pos_weight"] = scale_weight
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    
    champion_xgb = XGBClassifier(**best_params)
    champion_xgb.fit(X_train_val, y_train_val)
    
    print("    -> Generating test set predictions...")
    test_preds = champion_xgb.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 4. BOOTSTRAP CONFIDENCE INTERVALS
    # ---------------------------------------------------------
    print(f"    -> Running {N_BOOTSTRAPS} bootstrap iterations...")
    rng = np.random.default_rng(RANDOM_STATE)
    
    boot_auroc, boot_auprc, boot_brier = [], [], []
    test_size = len(y_test)
    
    for _ in range(N_BOOTSTRAPS):
        # Sample with replacement
        indices = rng.choice(test_size, size=test_size, replace=True)
        y_true_b = y_test[indices]
        y_pred_b = test_preds[indices]
        
        # Ensure both classes are present in the bootstrap sample
        if len(np.unique(y_true_b)) < 2:
            continue
            
        boot_auroc.append(roc_auc_score(y_true_b, y_pred_b))
        boot_auprc.append(average_precision_score(y_true_b, y_pred_b))
        boot_brier.append(brier_score_loss(y_true_b, y_pred_b))

    # Calculate 95% CIs
    ci_auroc = (np.percentile(boot_auroc, 2.5), np.percentile(boot_auroc, 97.5))
    ci_auprc = (np.percentile(boot_auprc, 2.5), np.percentile(boot_auprc, 97.5))
    ci_brier = (np.percentile(boot_brier, 2.5), np.percentile(boot_brier, 97.5))

    print("\n" + "="*60)
    print(" CHAMPION MODEL EVALUATION (95% CONFIDENCE INTERVALS)")
    print("="*60)
    print(f"    AUROC : {np.mean(boot_auroc):.4f}  [95% CI: {ci_auroc[0]:.4f} - {ci_auroc[1]:.4f}]")
    print(f"    AUPRC : {np.mean(boot_auprc):.4f}  [95% CI: {ci_auprc[0]:.4f} - {ci_auprc[1]:.4f}]")
    print(f"    Brier : {np.mean(boot_brier):.4f}  [95% CI: {ci_brier[0]:.4f} - {ci_brier[1]:.4f}]")
    print("="*60)

    elapsed = time.time() - start_time
    print(f"\n[*] Bootstrap completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()