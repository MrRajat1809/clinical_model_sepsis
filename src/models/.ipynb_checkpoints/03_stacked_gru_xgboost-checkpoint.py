"""
03_stacked_gru_xgboost.py

Trains a Deep-Stacked Hybrid Model (BiGRU Embeddings + XGBoost).
- Step 1: Passes the 24x30 temporal tensor through the pre-trained BiGRU to extract deep trajectory embeddings.
- Step 2: Concatenates the temporal embeddings with raw static features (Age, SOFA, CCI, Gender).
- Step 3: Trains an XGBoost classifier on the fused multimodal feature space.
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
from torch.utils.data import Dataset, DataLoader

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

# Reuse the same BiGRU architecture structure to load weights cleanly
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
        # Average pooling across time steps to create a fixed-size trajectory embedding (Batch, hidden_dim*2)
        return torch.mean(gru_out, dim=1)

def main():
    print("[*] Initializing Deep-Stacked BiGRU -> XGBoost Pipeline...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD DATA & TEST INDICES
    # ---------------------------------------------------------
    print("    -> Loading data and test indices for strict alignment...")
    X_imputed = np.load(TENSOR_DIR / "sepsis_imputed_tensor.npy")
    stay_ids = np.load(TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    
    df_cohort = pl.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    test_indices_file = METRICS_OUT_DIR / "test_set_indices.npy"
    idx_test = np.load(test_indices_file)
    all_indices = np.arange(len(y))
    idx_train_val = np.setdiff1d(all_indices, idx_test)
    
    # ---------------------------------------------------------
    # 2. EXTRACT & SCALE STATIC FEATURES
    # ---------------------------------------------------------
    potential_statics = ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"]
    static_cols = [col for col in potential_statics if col in df_cohort.columns]
    
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
        
    X_static_raw = df_static.fillna(0).values
    
    scaler_static = StandardScaler()
    scaler_static.fit(X_static_raw[idx_train_val])
    X_static = scaler_static.transform(X_static_raw)

    # ---------------------------------------------------------
    # 3. EXTRACT DEEP TEMPORAL EMBEDDINGS VIA TRAINED BiGRU
    # ---------------------------------------------------------
    print("    -> Extracting latent trajectory embeddings from pre-trained BiGRU...")
    model_path = MODEL_OUT_DIR / "mimic_multimodal_bigru.pth"
    if not model_path.exists():
        print(f"[ERROR] Trained BiGRU weights not found at {model_path}. Run 02_gru_temporal.py first.")
        return
        
    # Load the full BiGRU model weights into the encoder backbone
    full_state_dict = torch.load(model_path, map_location=DEVICE)
    encoder_state_dict = {k.replace('gru.', ''): v for k, v in full_state_dict.items() if k.startswith('gru.')}
    
    encoder = BiGRUEncoder(temporal_dim=30, hidden_dim=64).to(DEVICE)
    encoder.gru.load_state_dict(encoder_state_dict)
    encoder.eval()
    
    # Pass all temporal tensors through the encoder in batches to avoid VRAM overflow
    embedding_loader = DataLoader(torch.tensor(X_imputed, dtype=torch.float32), batch_size=256, shuffle=False)
    embeddings = []
    
    with torch.no_grad():
        for batch in embedding_loader:
        
            batch = batch.to(DEVICE)
            emb = encoder(batch)
            embeddings.append(emb.cpu().numpy())
            
    X_temporal_emb = np.concatenate(embeddings, axis=0)
    print(f"    -> Extracted Temporal Embeddings Shape: {X_temporal_emb.shape}")

    # ---------------------------------------------------------
    # 4. FUSE MODALITIES (Embeddings + Static Tabular)
    # ---------------------------------------------------------
    print("    -> Fusing temporal embeddings with static tabular features...")
    X_fused = np.concatenate([X_static, X_temporal_emb], axis=1)
    
    # Align splits using the exact same indices
    X_train_val_fused = X_fused[idx_train_val]
    y_train_val = y[idx_train_val]
    
    X_test_fused = X_fused[idx_test]
    y_test = y[idx_test]

    # ---------------------------------------------------------
    # 5. TRAIN STACKED XGBOOST CLASSIFIER
    # ---------------------------------------------------------
    print("    -> Training Stacked XGBoost on fused feature space...")
    scale_weight = float((len(y_train_val) - sum(y_train_val)) / sum(y_train_val))
    
    stacked_xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    stacked_xgb.fit(X_train_val_fused, y_train_val)
    
    # ---------------------------------------------------------
    # 6. EVALUATE ON TEST SET
    # ---------------------------------------------------------
    preds = stacked_xgb.predict_proba(X_test_fused)[:, 1]
    
    auroc = roc_auc_score(y_test, preds)
    auprc = average_precision_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)
    
    # Update metrics file
    metrics_file = METRICS_OUT_DIR / "baseline_ml_metrics.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
        
    metrics["Stacked_BiGRU_XGBoost"] = {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Brier": float(brier)
    }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n" + "="*50)
    print(" STACKED BiGRU -> XGBOOST PERFORMANCE (TEST SET)")
    print("="*50)
    print(f"    - AUROC: {auroc:.3f}")
    print(f"    - AUPRC: {auprc:.3f}")
    print(f"    - Brier: {brier:.3f}")
    print("="*50)
    
    elapsed = time.time() - start_time
    print(f"[*] Pipeline completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()