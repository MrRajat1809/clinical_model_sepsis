"""
08_attention_bigru.py

Trains an End-to-End Multimodal Deep Learning Model (Attention-BiGRU + MLP).
Upgrades the temporal branch by replacing mean pooling with a learned Temporal Attention 
mechanism, allowing the model to dynamically focus on the most critical hours of the 24h window.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import torch
import torch.nn as nn
import torch.optim as optim
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

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

# ==========================================
# PYTORCH DATASET
# ==========================================
class MultimodalSepsisDataset(Dataset):
    def __init__(self, X_temp, X_stat, y):
        self.X_temp = torch.tensor(X_temp, dtype=torch.float32)
        self.X_stat = torch.tensor(X_stat, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_temp[idx], self.X_stat[idx], self.y[idx]

# ==========================================
# ARCHITECTURE WITH TEMPORAL ATTENTION
# ==========================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, gru_output):
        # gru_output shape: (Batch, Seq_Len, Hidden_Dim)
        attn_weights = self.attention(gru_output)  # (Batch, Seq_Len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum across the time dimension
        context_vector = torch.sum(attn_weights * gru_output, dim=1) # (Batch, Hidden_Dim)
        return context_vector, attn_weights

class AttentionBiGRUFusionNet(nn.Module):
    def __init__(self, temporal_dim=30, static_dim=4, hidden_dim=64):
        super(AttentionBiGRUFusionNet, self).__init__()
        
        self.gru = nn.GRU(
            input_size=temporal_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        # Hidden_dim * 2 because of bidirectional GRU
        self.temporal_attention = TemporalAttention(hidden_dim * 2)
        
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear((hidden_dim * 2) + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_temp, x_stat):
        gru_out, _ = self.gru(x_temp)
        x_t, attn_weights = self.temporal_attention(gru_out)
        
        x_s = self.static_mlp(x_stat)
        x_fused = torch.cat([x_t, x_s], dim=1)
        logits = self.classifier(x_fused)
        return logits.squeeze(-1)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Initializing End-to-End Attention-BiGRU Pipeline...")
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    
    # Load Data
    X_imputed = np.load(TENSOR_DIR / "sepsis_imputed_tensor.npy")
    stay_ids = np.load(TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    
    df_cohort = pl.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    idx_test = np.load(METRICS_OUT_DIR / "test_set_indices.npy")
    idx_train_val = np.setdiff1d(np.arange(len(y)), idx_test)
    
    # Static Processing
    static_cols = [col for col in ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"] if col in df_cohort.columns]
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
        
    X_static_raw = df_static.fillna(0).values
    scaler_static = StandardScaler()
    scaler_static.fit(X_static_raw[idx_train_val])
    X_static = scaler_static.transform(X_static_raw)
    
    # Train/Val Split
    y_train_val = y[idx_train_val]
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=0.1765, random_state=RANDOM_STATE, stratify=y_train_val 
    )
    
    # DataLoaders
    train_loader = DataLoader(MultimodalSepsisDataset(X_imputed[idx_train], X_static[idx_train], y[idx_train]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultimodalSepsisDataset(X_imputed[idx_val], X_static[idx_val], y[idx_val]), batch_size=BATCH_SIZE)
    test_loader = DataLoader(MultimodalSepsisDataset(X_imputed[idx_test], X_static[idx_test], y[idx_test]), batch_size=BATCH_SIZE)

    # Initialize
    model = AttentionBiGRUFusionNet(temporal_dim=30, static_dim=len(static_cols)).to(DEVICE)
    pos_weight = torch.tensor([(len(y[idx_train]) - sum(y[idx_train])) / sum(y[idx_train])]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    model_save_path = MODEL_OUT_DIR / "mimic_multimodal_attention_bigru.pth"
    best_val_auprc = 0.0

    print("    -> Commencing Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x_t_batch, x_s_batch, y_batch in train_loader:
            x_t_batch, x_s_batch, y_batch = x_t_batch.to(DEVICE), x_s_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x_t_batch, x_s_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_t_batch, x_s_batch, y_batch in val_loader:
                logits = model(x_t_batch.to(DEVICE), x_s_batch.to(DEVICE))
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(y_batch.numpy())
                
        val_auroc = roc_auc_score(val_targets, val_preds)
        val_auprc = average_precision_score(val_targets, val_preds)
        
        print(f"       Epoch {epoch+1:02d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val AUROC: {val_auroc:.3f} | Val AUPRC: {val_auprc:.3f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            torch.save(model.state_dict(), model_save_path)
            
    # Final Test
    print("\n    -> Evaluating Best Attention Model on Hold-Out Test Set...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x_t_batch, x_s_batch, y_batch in test_loader:
            logits = model(x_t_batch.to(DEVICE), x_s_batch.to(DEVICE))
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_targets.extend(y_batch.numpy())
            
    test_auroc = roc_auc_score(test_targets, test_preds)
    test_auprc = average_precision_score(test_targets, test_preds)
    test_brier = brier_score_loss(test_targets, test_preds)
    
    with open(METRICS_OUT_DIR / "baseline_ml_metrics.json", "r") as f:
        metrics = json.load(f)
    metrics["Attention_BiGRU_Multimodal"] = {"AUROC": float(test_auroc), "AUPRC": float(test_auprc), "Brier": float(test_brier)}
    with open(METRICS_OUT_DIR / "baseline_ml_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("\n" + "="*50)
    print(" ATTENTION-BiGRU MODEL PERFORMANCE (TEST SET)")
    print("="*50)
    print(f"    - AUROC: {test_auroc:.4f}")
    print(f"    - AUPRC: {test_auprc:.4f}")
    print(f"    - Brier: {test_brier:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()