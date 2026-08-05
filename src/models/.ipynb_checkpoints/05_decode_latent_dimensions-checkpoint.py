"""
05_decode_latent_dimensions.py

Reverse-engineers the most important latent trajectory embeddings (e.g., Dim 122) 
by calculating their Pearson correlation against the raw clinical physiological statistics.
This bridges the gap between deep learning abstract math and interpretable ICU physiology.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_DIM = 122  # The top SHAP feature from our Super-Stack

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
    print(f"[*] Decoding BiGRU Latent Dimension {TARGET_DIM}...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD DATA & FEATURES
    # ---------------------------------------------------------
    X_imputed = np.load(TENSOR_DIR / "sepsis_imputed_tensor.npy")
    stay_ids = np.load(TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    tensor_features = list(np.load(TENSOR_DIR / "sepsis_tensor_features.npy"))
    
    df_cohort = pl.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    
    # ---------------------------------------------------------
    # 2. CONSTRUCT RAW CLINICAL FEATURE DATAFRAME
    # ---------------------------------------------------------
    # Static
    static_cols = [col for col in ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"] if col in df_cohort.columns]
    df_clinical = df_cohort[static_cols].copy()
    if "gender" in df_clinical.columns and df_clinical["gender"].dtype == 'O':
        df_clinical["gender"] = (df_clinical["gender"] == "M").astype(int)
    
    df_clinical.columns = [f"Static_{col.upper()}" for col in df_clinical.columns]
    
    # Aggregated Temporal
    X_mean = np.mean(X_imputed, axis=1)
    X_min = np.min(X_imputed, axis=1)
    X_max = np.max(X_imputed, axis=1)
    X_std = np.std(X_imputed, axis=1)
    
    for i, feat in enumerate(tensor_features):
        df_clinical[f"{feat}_Mean"] = X_mean[:, i]
        df_clinical[f"{feat}_Min"] = X_min[:, i]
        df_clinical[f"{feat}_Max"] = X_max[:, i]
        df_clinical[f"{feat}_Std"] = X_std[:, i]

    # ---------------------------------------------------------
    # 3. EXTRACT BIGRU EMBEDDINGS
    # ---------------------------------------------------------
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
    
    # Isolate the target dimension
    df_clinical[f"Latent_Dim_{TARGET_DIM}"] = X_temporal_emb[:, TARGET_DIM]

    # ---------------------------------------------------------
    # 4. CORRELATION ANALYSIS
    # ---------------------------------------------------------
    print("    -> Computing Pearson correlations against all raw physiology...")
    correlations = df_clinical.corr()[f"Latent_Dim_{TARGET_DIM}"].drop(f"Latent_Dim_{TARGET_DIM}")
    
    # Separate positive and negative correlations
    top_positive = correlations.sort_values(ascending=False).head(10)
    top_negative = correlations.sort_values(ascending=True).head(10)

    print("\n" + "="*65)
    print(f" CLINICAL TRANSLATION: BiGRU_Latent_Dim_{TARGET_DIM}")
    print("="*65)
    print(" POSITIVE CORRELATIONS (Higher Dim 122 = Higher Value):")
    for feat, corr in top_positive.items():
        print(f"   + {corr:.3f} | {feat}")
        
    print("\n NEGATIVE CORRELATIONS (Higher Dim 122 = Lower Value):")
    for feat, corr in top_negative.items():
        print(f"   - {corr:.3f} | {feat}")
    print("="*65)
    
    elapsed = time.time() - start_time
    print(f"\n[*] Decoding completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()