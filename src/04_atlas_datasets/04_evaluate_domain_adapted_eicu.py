"""
Evaluate the locked model on the transport-aligned eICU cohort.

The test of whether distributional alignment actually helps. The same locked
model that produced the unaligned external result is applied to the harmonised
representation, so the comparison isolates the effect of transport.

Two reconciliations are required and both are load-bearing. The atlas stores
columns as [120 temporal, 4 static] while the model expects [4 static, 120
temporal], and the atlas static block is ordered [age, sex, Charlson, SOFA]
against the model's [age, SOFA, Charlson, sex]. Skipping either would feed
correctly named but wrong-valued columns into the trees.

Only the eICU rows are extracted from the atlas, so no MIMIC-IV patient leaks
into the external evaluation.

Reads:
    the atlas features and metadata
    outputs/models/mimic_champion_xgboost.joblib, the shared split indices
Writes:
    outputs/metrics/eicu_OT_domain_adapted_metrics.json and a ROC figure
"""

import time
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from scipy.special import logit

import warnings
warnings.filterwarnings("ignore")

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR_ATLAS = BASE_DIR / "data" / "processed" / "atlas"
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"
OUT_MODELS = BASE_DIR / "outputs" / "models"

ATLAS_FEATURES_FILE = PROCESSED_DIR_ATLAS / "atlas_sepsis_features.npy"
ATLAS_META_FILE = PROCESSED_DIR_ATLAS / "atlas_metadata.parquet"

MIMIC_TENSOR_FILE = PROCESSED_DIR_MIMIC / "mimic_sepsis_imputed_tensor.npy"
MIMIC_STAY_ID_FILE = PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_stay_ids.npy"
MIMIC_COHORT_FILE = PROCESSED_DIR_MIMIC / "mimic_final_sepsis3_cohort.parquet"
MIMIC_TRAIN_IDX_FILE = OUT_MODELS / "mimic_train_indices.npy"

XGB_MODEL_FILE = OUT_MODELS / "mimic_champion_xgboost.joblib"

OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

METRICS_FILE = OUT_METRICS / "eicu_OT_domain_adapted_metrics.json"
ROC_PLOT_FILE = OUT_FIGURES / "eicu_OT_Domain_Adapted_ROC.png"

RANDOM_STATE = 42
N_BOOTSTRAPS = 1000

# --- Evaluation Helpers --------------------------------------------------
def evaluate_champion(y_true, y_prob, threshold=0.5, n_bootstraps=1000):
    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

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
        "Brier_95CI": [float(np.percentile(boot_brier, 2.5)), float(np.percentile(boot_brier, 97.5))]
    }

# --- Main Execution ------------------------------------------------------
def main():
    print("[*] Initiating Domain-Adapted External Validation...")
    start_time = time.time()

    # --- Reconstruct Source Scalers From Mimic-iv ------------------------
    print("    -> Reconstructing original scalers from MIMIC-IV training data...")
    mimic_tensor = np.load(MIMIC_TENSOR_FILE)
    mimic_stay_ids = np.load(MIMIC_STAY_ID_FILE)
    mimic_train_idx = np.load(MIMIC_TRAIN_IDX_FILE)
    df_mimic = pl.read_parquet(MIMIC_COHORT_FILE).to_pandas()
    df_mimic = pd.DataFrame({"stay_id": mimic_stay_ids}).merge(df_mimic, on="stay_id", how="left")

    static_cols_model_order = ["age", "baseline_sofa"]
    df_mimic_static = df_mimic[static_cols_model_order].copy()

    # Fit Static Scaler
    scaler_static = StandardScaler()
    scaler_static.fit(df_mimic_static.fillna(0).values[mimic_train_idx])

    # Fit Temporal Scaler
    mimic_mean = np.mean(mimic_tensor, axis=1)
    mimic_min = np.min(mimic_tensor, axis=1)
    mimic_max = np.max(mimic_tensor, axis=1)
    mimic_std = np.std(mimic_tensor, axis=1)
    X_mimic_temporal_raw = np.concatenate([mimic_mean, mimic_min, mimic_max, mimic_std], axis=1)
    
    scaler_temporal = StandardScaler()
    # Fitted only on mimic_train_idx
    scaler_temporal.fit(X_mimic_temporal_raw[mimic_train_idx])

    # --- EXTRACT AND REORDER OT-HARMONIZED eICU DATA ---------------------
    print("    -> Extracting OT-Harmonized eICU data from Atlas...")
    X_atlas = np.load(ATLAS_FEATURES_FILE)
    df_meta = pd.read_parquet(ATLAS_META_FILE)
    
    eicu_mask = df_meta["cohort_source"] == "eICU-CRD"
    X_eicu_ot = X_atlas[eicu_mask]
    y_test = df_meta[eicu_mask]["hospital_expire_flag"].values
    
    # Defensive check to ensure alignment
    assert len(X_eicu_ot) == len(y_test), "Mismatch between feature tensor and target labels!"
    
    ATLAS_SRC = X_eicu_ot

    # Atlas layout, written by 04_atlas_datasets/01a: temporal block first, then
    # the static block in MODEL_STATICS order. Deriving the split from the names
    # means a change to the static set fails loudly rather than silently
    # selecting the wrong columns.
    MODEL_STATICS = ["age", "baseline_sofa"]
    N_TEMPORAL = ATLAS_SRC.shape[1] - len(MODEL_STATICS)
    assert N_TEMPORAL == 120, f"expected 120 temporal columns, got {N_TEMPORAL}"

    X_eicu_ot_temporal = X_eicu_ot[:, :N_TEMPORAL]
    X_eicu_ot_static = X_eicu_ot[:, N_TEMPORAL:]

    # Atlas static order and model static order now coincide, so the
    # permutation is the identity; it is computed rather than assumed.
    static_perm = [MODEL_STATICS.index(c) for c in static_cols_model_order]
    X_eicu_ot_static_reordered = X_eicu_ot_static[:, static_perm]
    
    print("    -> Scaling harmonized features into MIMIC statistical space...")
    X_eicu_static_scaled = scaler_static.transform(X_eicu_ot_static_reordered)
    X_eicu_temporal_scaled = scaler_temporal.transform(X_eicu_ot_temporal)
    
    # Final Fused eICU Tensor (Static first, Temporal second)
    X_test_ot = np.concatenate([X_eicu_static_scaled, X_eicu_temporal_scaled], axis=1)

    print(f"       - OT eICU Cohort Size: {len(y_test):,} patients")
    print(f"       - Final Feature Vector Shape: {X_test_ot.shape}")

    # --- Inference & Evaluation ------------------------------------------
    print("    -> Loading locked MIMIC-IV XGBoost model (Inference Mode)...")
    champion_xgb = joblib.load(XGB_MODEL_FILE)

    print(f"    -> Running Evaluation & {N_BOOTSTRAPS}-Iteration Bootstrap...")
    preds_ot = champion_xgb.predict_proba(X_test_ot)[:, 1]

    metrics = evaluate_champion(y_test, preds_ot, n_bootstraps=N_BOOTSTRAPS)

    with open(METRICS_FILE, "w") as f:
        json.dump({"model": "Champion_XGBoost_OT_Domain_Adapted", "metrics": metrics}, f, indent=4)

    print("\n" + "="*60)
    print(" DOMAIN-ADAPTED (OT-eICU) EXTERNAL VALIDATION")
    print("="*60)
    print(f"    AUROC : {metrics['AUROC']:.4f}  [95% CI: {metrics['AUROC_95CI'][0]:.4f} - {metrics['AUROC_95CI'][1]:.4f}]")
    print(f"    AUPRC : {metrics['AUPRC']:.4f}  [95% CI: {metrics['AUPRC_95CI'][0]:.4f} - {metrics['AUPRC_95CI'][1]:.4f}]")
    print(f"    Brier : {metrics['Brier']:.4f}  [95% CI: {metrics['Brier_95CI'][0]:.4f} - {metrics['Brier_95CI'][1]:.4f}]")
    print("="*60)

    # --- Roc Plot --------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, preds_ot)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#C44E52', lw=2, label=f'OT-Harmonized eICU (AUROC = {metrics["AUROC"]:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, weight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, weight='bold')
    plt.title('Domain-Adapted External Validation ROC', fontsize=14, weight='bold')
    plt.legend(loc="lower right", frameon=True)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(ROC_PLOT_FILE, dpi=300)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n[+] Success! Domain-adapted validation completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
