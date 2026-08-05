"""
04b_shap_consensus_waterfall.py

Calculates Consensus SHAP values over 50 iterative XGBoost training cycles 
to ensure absolute stability of the feature attributions.
Generates a Local Waterfall Plot for a representative non-surviving patient 
to visually decode their physiological collapse.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
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
FIGURE_OUT_DIR = BASE_DIR / "outputs" / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ITERATIONS = 50

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
    print(f"[*] Initiating Consensus SHAP Analysis ({N_ITERATIONS} iterations)...")
    FEATURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD RAW DATA & METADATA
    # ---------------------------------------------------------
    print("    -> Loading data and reconstructing features...")
    X_imputed = np.load(TENSOR_DIR / "sepsis_imputed_tensor.npy")
    stay_ids = np.load(TENSOR_DIR / "sepsis_tensor_stay_ids.npy")
    tensor_features = list(np.load(TENSOR_DIR / "sepsis_tensor_features.npy"))
    
    df_cohort = pl.read_parquet(COHORT_DIR / "final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    # ---------------------------------------------------------
    # 2. ASSEMBLE SUPER-STACK MATRIX
    # ---------------------------------------------------------
    # Static
    static_cols = [col for col in ["age", "baseline_sofa", "charlson_comorbidity_index", "gender"] if col in df_cohort.columns]
    df_static = df_cohort[static_cols].copy()
    if "gender" in df_static.columns and df_static["gender"].dtype == 'O':
        df_static["gender"] = (df_static["gender"] == "M").astype(int)
        
    X_static = StandardScaler().fit_transform(df_static.fillna(0).values)
    feature_names = [f"Static_{col.upper()}" for col in static_cols]

    # Aggregated
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
    
    emb_names = [f"BiGRU_Latent_Dim_{i}" for i in range(X_temporal_emb.shape[1])]
    feature_names.extend(emb_names)

    X_fused = np.concatenate([X_static, X_temporal_agg, X_temporal_emb], axis=1)
    
    # ---------------------------------------------------------
    # 3. 50-ITERATION SHAP CONSENSUS
    # ---------------------------------------------------------
    print("    -> Executing iterative model training and SHAP extraction...")
    scale_weight = float((len(y) - sum(y)) / sum(y))
    
    accumulated_shap_values = np.zeros(X_fused.shape)
    accumulated_base_values = np.zeros(len(y))

    for i in tqdm(range(N_ITERATIONS), desc="SHAP Iterations"):
        model = XGBClassifier(
            n_estimators=100, # Lower estimators for faster iteration, sufficient for stable SHAP
            learning_rate=0.05, 
            max_depth=5, 
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight, 
            random_state=i, 
            n_jobs=-1
        )
        model.fit(X_fused, y)
        
        explainer = shap.TreeExplainer(model)
        # Get full Explanation object to track base values (expected values)
        shap_out = explainer(X_fused) 
        
        accumulated_shap_values += shap_out.values
        # Handle cases where base_values might be a scalar or an array
        if isinstance(shap_out.base_values, np.ndarray) and len(shap_out.base_values.shape) > 0:
             accumulated_base_values += shap_out.base_values
        else:
             accumulated_base_values += np.full(len(y), shap_out.base_values)
             
    # Average the values
    consensus_shap_values = accumulated_shap_values / N_ITERATIONS
    consensus_base_values = accumulated_base_values / N_ITERATIONS

    # ---------------------------------------------------------
    # 4. SELECT REPRESENTATIVE PATIENT & PLOT WATERFALL
    # ---------------------------------------------------------
    print("\n    -> Generating Local Consensus Waterfall Plot...")
    
    # We want a patient who actually died (y == 1) and who the model correctly identified
    # Let's use the final trained model from the loop to find a True Positive
    probs = model.predict_proba(X_fused)[:, 1]
    
    # Find a patient who died and had a very high predicted probability of death (> 0.85)
    true_positive_indices = np.where((y == 1) & (probs > 0.85))[0]
    
    if len(true_positive_indices) > 0:
        target_patient_idx = true_positive_indices[0] # Pick the first one
    else:
        target_patient_idx = np.where(y == 1)[0][0]   # Fallback to first non-survivor
        
    print(f"       - Selected Patient Index: {target_patient_idx} | True Mortality: 1 | Model Prob: {probs[target_patient_idx]:.3f}")

    # Build a SHAP Explanation object strictly for this patient
    patient_explanation = shap.Explanation(
        values=consensus_shap_values[target_patient_idx, :],
        base_values=consensus_base_values[target_patient_idx],
        data=X_fused[target_patient_idx, :],
        feature_names=feature_names
    )

    # Plot and Save
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(patient_explanation, max_display=12, show=False)
    
    plot_path = FIGURE_OUT_DIR / "SHAP_Consensus_Waterfall_Patient.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n[*] Success! SHAP Waterfall plot exported to: {plot_path.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()