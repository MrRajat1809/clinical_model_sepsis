"""
04_shap_interpretability.py

Calculates global SHAP (Shapley Additive exPlanations) values for the 
Optimized Super-Stack model to interpret feature importance.
Names all static, aggregated, and deep embedding features to trace exactly 
what physiological signals drive the mortality predictions.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import shap
from sklearn.preprocessing import StandardScaler
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
FEATURE_OUT_DIR = BASE_DIR / "outputs" / "features"

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
    print("[*] Initiating SHAP Interpretability Analysis on Super-Stack...")
    FEATURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD RAW DATA & METADATA
    # ---------------------------------------------------------
    X_imputed = np.load(TENSOR_DIR / "sepsis_imputed_tensor.npy")
    stay_ids = np.load(TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    tensor_features = list(np.load(TENSOR_DIR / "sepsis_tensor_features.npy"))
    
    df_cohort = pl.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    # We will train on the entire dataset for the global SHAP explanation
    # to understand the full physiological manifold
    
    # ---------------------------------------------------------
    # 2. PROCESS STATIC FEATURES
    # ---------------------------------------------------------
    static_cols = [col for col in ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"] if col in df_cohort.columns]
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
        
    X_static = StandardScaler().fit_transform(df_static.fillna(0).values)
    feature_names = [f"Static_{col.upper()}" for col in static_cols]

    # ---------------------------------------------------------
    # 3. PROCESS AGGREGATED FEATURES
    # ---------------------------------------------------------
    X_mean = np.mean(X_imputed, axis=1)
    X_min = np.min(X_imputed, axis=1)
    X_max = np.max(X_imputed, axis=1)
    X_std = np.std(X_imputed, axis=1)
    
    X_temporal_agg = np.concatenate([X_mean, X_min, X_max, X_std], axis=1)
    X_temporal_agg = StandardScaler().fit_transform(X_temporal_agg)
    
    agg_names = []
    for stat in ["Mean", "Min", "Max", "Std"]:
        for feat in tensor_features:
            agg_names.append(f"{feat}_{stat}")
            
    feature_names.extend(agg_names)

    # ---------------------------------------------------------
    # 4. EXTRACT DEEP BIGRU EMBEDDINGS
    # ---------------------------------------------------------
    print("    -> Reconstructing latent trajectories from BiGRU...")
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
    
    emb_names = [f"BiGRU_Latent_Dim_{i}" for i in range(X_temporal_emb.shape[1])]
    feature_names.extend(emb_names)

    # ---------------------------------------------------------
    # 5. FUSE AND TRAIN
    # ---------------------------------------------------------
    X_fused = np.concatenate([X_static, X_temporal_agg, X_temporal_emb], axis=1)
    print(f"    -> Super-Stack constructed. Total Features: {X_fused.shape[1]}")
    
    print("    -> Training Super-Stack for SHAP Extraction...")
    scale_weight = float((len(y) - sum(y)) / sum(y))
    super_xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=5, 
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_weight, 
        random_state=RANDOM_STATE, n_jobs=-1
    )
    super_xgb.fit(X_fused, y)

    # ---------------------------------------------------------
    # 6. SHAP VALUE COMPUTATION
    # ---------------------------------------------------------
    print("    -> Calculating TreeExplainer SHAP values (This may take a moment)...")
    explainer = shap.TreeExplainer(super_xgb)
    shap_values = explainer.shap_values(X_fused)
    
    # Calculate Mean Absolute SHAP to find the most important features globally
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": shap_importance
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False)
    
    # Export for later plotting
    df_importance.to_csv(FEATURE_OUT_DIR / "superstack_shap_importance.csv", index=False)
    
    print("\n" + "="*60)
    print(" TOP 20 PHYSIOLOGICAL DRIVERS OF SEPSIS MORTALITY")
    print("="*60)
    for idx, row in df_importance.head(20).iterrows():
        print(f" {row['Feature']:<35} | SHAP: {row['Mean_Absolute_SHAP']:.4f}")
    print("="*60)
    
    elapsed = time.time() - start_time
    print(f"\n[*] SHAP Analysis completed in {elapsed:.1f} seconds.")
    print(f"    -> Exported full importance matrix to: {FEATURE_OUT_DIR.name}/")

if __name__ == "__main__":
    main()