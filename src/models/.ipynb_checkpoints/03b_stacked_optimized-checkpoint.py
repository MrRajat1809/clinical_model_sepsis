"""
03b_stacked_optimized.py

Trains an Optimized Deep-Stacked Hybrid Model ("Super-Stack").
- Extracts Deep BiGRU Embeddings.
- Extracts Aggregated Temporal Features (Min, Max, Mean, Std).
- Fuses Static + Aggregated + Deep Embeddings into a single massive matrix.
- Trains a regularized XGBoost classifier to utilize both explicit thresholds and latent trajectories.
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
    print("[*] Initializing Optimized Super-Stack Pipeline...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD DATA & TEST INDICES
    # ---------------------------------------------------------
    X_imputed = np.load(TENSOR_DIR / "sepsis_imputed_tensor.npy")
    stay_ids = np.load(TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    
    df_cohort = pl.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    idx_test = np.load(METRICS_OUT_DIR / "test_set_indices.npy")
    idx_train_val = np.setdiff1d(np.arange(len(y)), idx_test)
    
    # ---------------------------------------------------------
    # 2. EXTRACT & SCALE STATIC FEATURES
    # ---------------------------------------------------------
    print("    -> Processing static features...")
    static_cols = [col for col in ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"] if col in df_cohort.columns]
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
        
    scaler_static = StandardScaler()
    X_static = scaler_static.fit_transform(df_static.fillna(0).values)

    # ---------------------------------------------------------
    # 3. EXTRACT AGGREGATED TEMPORAL FEATURES
    # ---------------------------------------------------------
    print("    -> Extracting aggregated temporal thresholds (Min, Max, Mean, Std)...")
    X_mean = np.mean(X_imputed, axis=1)
    X_min = np.min(X_imputed, axis=1)
    X_max = np.max(X_imputed, axis=1)
    X_std = np.std(X_imputed, axis=1)
    
    X_temporal_agg = np.concatenate([X_mean, X_min, X_max, X_std], axis=1)
    scaler_agg = StandardScaler()
    X_temporal_agg = scaler_agg.fit_transform(X_temporal_agg)

    # ---------------------------------------------------------
    # 4. EXTRACT DEEP TEMPORAL EMBEDDINGS
    # ---------------------------------------------------------
    print("    -> Extracting latent trajectories from BiGRU...")
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

    # ---------------------------------------------------------
    # 5. FUSE THE SUPER-STACK
    # ---------------------------------------------------------
    print("    -> Building the Super-Stack feature matrix...")
    X_fused = np.concatenate([X_static, X_temporal_agg, X_temporal_emb], axis=1)
    
    X_train_val = X_fused[idx_train_val]
    y_train_val = y[idx_train_val]
    X_test = X_fused[idx_test]
    y_test = y[idx_test]

    # ---------------------------------------------------------
    # 6. TRAIN REGULARIZED XGBOOST
    # ---------------------------------------------------------
    print("    -> Training Regularized XGBoost...")
    scale_weight = float((len(y_train_val) - sum(y_train_val)) / sum(y_train_val))
    
    super_xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8, # Regularization to prevent overfitting on the massive feature set
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    super_xgb.fit(X_train_val, y_train_val)
    
    # ---------------------------------------------------------
    # 7. EVALUATE
    # ---------------------------------------------------------
    preds = super_xgb.predict_proba(X_test)[:, 1]
    
    auroc = roc_auc_score(y_test, preds)
    auprc = average_precision_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    
    metrics_file = METRICS_OUT_DIR / "baseline_ml_metrics.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    metrics["Optimized_SuperStack"] = {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Brier": float(brier)
    }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n" + "="*50)
    print(" OPTIMIZED SUPER-STACK PERFORMANCE (TEST SET)")
    print("="*50)
    print(f"    - AUROC: {auroc:.3f}")
    print(f"    - AUPRC: {auprc:.3f}")
    print(f"    - Brier: {brier:.3f}")
    print("="*50)
    
    elapsed = time.time() - start_time
    print(f"[*] Pipeline completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()