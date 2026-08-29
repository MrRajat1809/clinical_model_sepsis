"""
Joint model: BiGRU over the trajectory fused with an MLP over static variables.

Gives the neural arm the same information the primary gradient-boosted model
sees, so a difference in performance is attributable to the architecture rather
than to the inputs. The temporal branch is the 03a BiGRU and the static branch
is the 03b MLP; the two representations are concatenated before the classifier.

Training configuration is identical to the unimodal models.

Reads:
    mimic_sepsis_imputed_tensor.npy, mimic_final_sepsis3_cohort.parquet,
    the shared split indices
Writes:
    model weights, predictions, temporal embeddings, training history and curve
"""

import time
import json
import random
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, balanced_accuracy_score, precision_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import warnings
warnings.filterwarnings("ignore")

# --- Configuration & Reproducibility -------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_PREDS = BASE_DIR / "outputs" / "predictions"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FEATS = BASE_DIR / "outputs" / "features"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

for d in [OUT_MODELS, OUT_PREDS, OUT_METRICS, OUT_FEATS, OUT_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
HIDDEN_DIM = 64
STATIC_DIM = 2
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Dataset & Architecture ----------------------------------------------
class MultimodalSepsisDataset(Dataset):
    def __init__(self, X_temp, X_stat, y):
        self.X_temp = torch.tensor(X_temp, dtype=torch.float32)
        self.X_stat = torch.tensor(X_stat, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_temp[idx], self.X_stat[idx], self.y[idx]

class BiGRUFusionNet(nn.Module):
    def __init__(self, temporal_dim=30, static_dim=2, hidden_dim=64):
        super(BiGRUFusionNet, self).__init__()
        
        # 1. Temporal Branch (BiGRU)
        self.gru = nn.GRU(
            input_size=temporal_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        # 2. Static Branch (MLP)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. Fusion & Classification Head
        self.classifier = nn.Sequential(
            nn.Linear((hidden_dim * 2) + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_temp, x_stat):
        gru_out, _ = self.gru(x_temp)
        x_t = torch.mean(gru_out, dim=1) 
        
        x_s = self.static_mlp(x_stat)
        
        x_fused = torch.cat([x_t, x_s], dim=1)
        logits = self.classifier(x_fused)
        
        # We return x_t (pure temporal embedding) for downstream XGBoost stacking
        return logits.squeeze(-1), x_t

# --- Evaluation Helper ---------------------------------------------------
def evaluate_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Sensitivity": float(tp / (tp + fn) if (tp + fn) > 0 else 0.0),
        "Specificity": float(tn / (tn + fp) if (tn + fp) > 0 else 0.0),
        "PPV": float(precision_score(y_true, y_pred, zero_division=0)),
        "NPV": float(tn / (tn + fn) if (tn + fn) > 0 else 0.0),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred))
    }

# --- Main Execution ------------------------------------------------------
def main():
    set_seed(RANDOM_STATE)
    print("[*] Initiating Phase 2: Multimodal BiGRU (Temporal + Static)...")
    start_time = time.time()
    
    # --- Load Shared Data & Splits ---------------------------------------
    print("    -> Loading shared tensor, cohort metadata, and exact split indices...")
    X_imputed = np.load(PROCESSED_DIR / "mimic_sepsis_imputed_tensor.npy")
    stay_ids = np.load(PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy")
    
    df_cohort = pl.read_parquet(PROCESSED_DIR / "mimic_final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    idx_train_val = np.load(OUT_MODELS / "mimic_train_indices.npy")
    idx_test = np.load(OUT_MODELS / "mimic_test_set_indices.npy")
    stay_ids_test = np.load(OUT_MODELS / "mimic_stay_ids_test.npy")

    # --- Extract & Scale Static Features ---------------------------------
    potential_statics = ["age", "baseline_sofa"]
    static_cols = [col for col in potential_statics if col in df_cohort.columns]
    
    df_static = df_cohort[static_cols].copy()
        
    X_static_raw = df_static.fillna(0).values
    
    # Scale strictly on the train_val set to prevent data leakage
    scaler_static = StandardScaler()
    scaler_static.fit(X_static_raw[idx_train_val])
    X_static = scaler_static.transform(X_static_raw)

    # --- Build Dataloaders -----------------------------------------------
    y_train_val = y[idx_train_val]
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_val 
    )

    X_t_train, X_s_train, y_train = X_imputed[idx_train], X_static[idx_train], y[idx_train]
    X_t_val, X_s_val, y_val = X_imputed[idx_val], X_static[idx_val], y[idx_val]
    X_t_test, X_s_test, y_test = X_imputed[idx_test], X_static[idx_test], y[idx_test]

    train_loader = DataLoader(MultimodalSepsisDataset(X_t_train, X_s_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultimodalSepsisDataset(X_t_val, X_s_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(MultimodalSepsisDataset(X_t_test, X_s_test, y_test), batch_size=BATCH_SIZE)
    full_loader = DataLoader(MultimodalSepsisDataset(X_imputed, X_static, y), batch_size=BATCH_SIZE, shuffle=False)

    # --- Initialize Architecture & Amp -----------------------------------
    print(f"    -> Initializing BiGRUFusionNet & Mixed Precision on {DEVICE}...")
    model = BiGRUFusionNet(temporal_dim=X_imputed.shape[2], static_dim=STATIC_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # --- Training Loop with Early Stopping -------------------------------
    print("    -> Commencing Training Loop...")
    best_val_auprc = 0.0
    epochs_no_improve = 0
    history = []
    model_save_path = OUT_MODELS / "mimic_multimodal_bigru_best.pth"

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for x_t_batch, x_s_batch, y_batch in train_loader:
            x_t_batch, x_s_batch, y_batch = x_t_batch.to(DEVICE), x_s_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    logits, _ = model(x_t_batch, x_s_batch)
                    loss = criterion(logits, y_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(x_t_batch, x_s_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_t_batch, x_s_batch, y_batch in val_loader:
                logits, _ = model(x_t_batch.to(DEVICE), x_s_batch.to(DEVICE))
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(y_batch.numpy())
                
        val_auroc = roc_auc_score(val_targets, val_preds)
        val_auprc = average_precision_score(val_targets, val_preds)
        
        scheduler.step(val_auprc)
        history.append({"Epoch": epoch+1, "Train_Loss": train_loss/len(train_loader), "Val_AUROC": val_auroc, "Val_AUPRC": val_auprc})
        
        print(f"        Epoch {epoch+1:02d}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val AUROC: {val_auroc:.4f} | Val AUPRC: {val_auprc:.4f}")
        
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"    [!] Early stopping triggered at epoch {epoch+1}")
                break

    # --- Post-training: Logs & Curves ------------------------------------
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(OUT_METRICS / "mimic_multimodal_bigru_history.csv", index=False)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_hist["Epoch"], df_hist["Val_AUROC"], label="Val AUROC", marker='o')
    plt.plot(df_hist["Epoch"], df_hist["Val_AUPRC"], label="Val AUPRC", marker='s')
    plt.title("Multimodal BiGRU Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUT_FIGURES / "mimic_multimodal_bigru_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Inference: Embeddings & Test Evaluation -------------------------
    print("\n    -> Extracting Latent Embeddings & Evaluating Test Set...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    all_embeddings = []
    with torch.no_grad():
        for x_t_batch, x_s_batch, _ in full_loader:
            _, emb = model(x_t_batch.to(DEVICE), x_s_batch.to(DEVICE))
            all_embeddings.append(emb.cpu().numpy())
    
    X_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(OUT_FEATS / "mimic_multimodal_bigru_embeddings.npy", X_embeddings)
    np.save(OUT_FEATS / "mimic_multimodal_bigru_stay_ids.npy", stay_ids)

    # Test Set Predictions
    test_preds = []
    with torch.no_grad():
        for x_t_batch, x_s_batch, _ in test_loader:
            logits, _ = model(x_t_batch.to(DEVICE), x_s_batch.to(DEVICE))
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())
            
    df_preds = pd.DataFrame({
        "stay_id": stay_ids_test,
        "true_label": y_test,
        "pred_probability": test_preds,
        "pred_label": (np.array(test_preds) >= 0.5).astype(int)
    })
    df_preds.to_csv(OUT_PREDS / "mimic_multimodal_bigru_predictions.csv", index=False)
    
    # Export Config
    config = {
        "model": "Multimodal_BiGRU",
        "hyperparameters": {"hidden_dim": HIDDEN_DIM, "static_dim": STATIC_DIM, "layers": 2, "dropout": 0.2, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "patience": PATIENCE},
        "best_epoch": int(df_hist.loc[df_hist['Val_AUPRC'].idxmax()]['Epoch'])
    }
    with open(OUT_MODELS / "mimic_multimodal_bigru_config.json", "w") as f:
        json.dump(config, f, indent=4)

    # --- Report ----------------------------------------------------------
    metrics = evaluate_model(y_test, np.array(test_preds))
    
    with open(OUT_METRICS / "mimic_multimodal_bigru_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n" + "="*50)
    print(" MULTIMODAL BiGRU PERFORMANCE (TEST SET)")
    print("="*50)
    print(f"    - AUROC: {metrics['AUROC']:.4f}")
    print(f"    - AUPRC: {metrics['AUPRC']:.4f}")
    print(f"    - Brier: {metrics['Brier']:.4f}")
    print("="*50)
    
    elapsed = time.time() - start_time
    print(f"[*] Pipeline completed in {elapsed/60:.1f} minutes.")

if __name__ == "__main__":
    main()
