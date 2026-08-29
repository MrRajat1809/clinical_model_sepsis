"""
Constrained transport variants, to locate which part of the mapping causes harm.

If unconstrained alignment degrades external performance, the useful question is
which part of the mapping is responsible. Two restricted variants isolate it,
both fitted against the MIMIC-IV development partition only.

    interval projection  each transported feature is clipped to the 0.5th-99.5th
                         percentile of the development distribution, so mapped
                         values cannot fall outside the range the model was
                         fitted on
    feature gating       nine objective laboratory features are returned to
                         their untransported values after mapping, so alignment
                         applies only to charted and static variables

Spearman correlation between each variant's predictions and the unaligned
predictions measures how far patient ranking survived. A variant that restores
AUROC while also restoring rank correlation identifies what the unconstrained
map was disturbing.

A feature-count assertion against the exported model feature list runs before
anything else, since a silent shape mismatch would invalidate every number.

Reads:
    both imputed tensors and cohort tables
    outputs/models/mimic_champion_xgboost.joblib
    outputs/features/mimic_champion_features.json
Writes:
    outputs/metrics/atlas_ot_constrained_variants.json
"""

import time
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import ot
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parents[2]

OUT_MODELS = BASE_DIR / "outputs" / "models"
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"

def main():
    print("[*] Initiating Diagnostic Optimal Transport Ablation...")
    start_time = time.time()

    # 1. Load Model & Expected Feature Names FIRST
    model = joblib.load(OUT_MODELS / "mimic_champion_xgboost.joblib")
    with open(BASE_DIR / "outputs" / "features" / "mimic_champion_features.json", "r") as f:
        feature_names = json.load(f)

    # 2. Load MIMIC Data & Train Indices
    mimic_tensor = np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_imputed_tensor.npy")
    mimic_stay_ids = np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_stay_ids.npy")
    mimic_train_idx = np.load(OUT_MODELS / "mimic_train_indices.npy")
    df_mimic = pl.read_parquet(PROCESSED_DIR_MIMIC / "mimic_final_sepsis3_cohort.parquet").to_pandas()
    df_mimic = pd.DataFrame({"stay_id": mimic_stay_ids}).merge(df_mimic, on="stay_id", how="left")
    
    static_cols = ["age", "baseline_sofa"]
    df_mimic_static = df_mimic[static_cols].copy()
    X_mimic_static_raw = df_mimic_static.fillna(0).values

    scaler_static = StandardScaler()
    scaler_static.fit(X_mimic_static_raw[mimic_train_idx])

    X_mimic_temporal_raw = np.concatenate([
        np.mean(mimic_tensor, axis=1), np.min(mimic_tensor, axis=1),
        np.max(mimic_tensor, axis=1), np.std(mimic_tensor, axis=1)
    ], axis=1)

    scaler_temporal = StandardScaler()
    scaler_temporal.fit(X_mimic_temporal_raw[mimic_train_idx]) # Fit only on train

    X_mimic_scaled = np.concatenate([
        scaler_static.transform(X_mimic_static_raw),
        scaler_temporal.transform(X_mimic_temporal_raw)
    ], axis=1)

    # FEATURE ORDERING AUDIT
    assert X_mimic_scaled.shape[1] == len(feature_names), \
        f"[FATAL] Feature count mismatch: Array has {X_mimic_scaled.shape[1]}, model expects {len(feature_names)}"

    # 3. Load eICU Target Data
    eicu_tensor = np.load(PROCESSED_DIR_EICU / "eicu_sepsis_imputed_tensor.npy")
    eicu_stay_ids = np.load(PROCESSED_DIR_EICU / "eicu_sepsis_tensor_stay_ids.npy")
    df_eicu = pl.read_parquet(PROCESSED_DIR_EICU / "eicu_final_sepsis3_cohort.parquet").to_pandas()
    df_eicu = pd.DataFrame({"stay_id": eicu_stay_ids}).merge(df_eicu, on="stay_id", how="left")
    y_test = df_eicu["hospital_expire_flag"].values

    df_eicu_static = df_eicu[static_cols].copy()
    X_eicu_static_raw = df_eicu_static.fillna(0).values

    X_eicu_temporal_raw = np.concatenate([
        np.mean(eicu_tensor, axis=1), np.min(eicu_tensor, axis=1),
        np.max(eicu_tensor, axis=1), np.std(eicu_tensor, axis=1)
    ], axis=1)

    X_eicu_scaled = np.concatenate([
        scaler_static.transform(X_eicu_static_raw),
        scaler_temporal.transform(X_eicu_temporal_raw)
    ], axis=1)

    # 4. Baseline Raw Inference
    raw_preds = model.predict_proba(X_eicu_scaled)[:, 1]
    raw_auc = roc_auc_score(y_test, raw_preds)
    print(f"\n[1] Raw eICU AUROC (Locked Benchmark)  : {raw_auc:.4f}")

    # 5. Standard OT baseline (Strict Source Training Distribution)
    X_mimic_source = X_mimic_scaled[mimic_train_idx] # Prevent Leakage
    
    SAMPLE_SIZE = 3000
    rng = np.random.default_rng(42)
    idx_m = rng.choice(X_mimic_source.shape[0], size=min(SAMPLE_SIZE, X_mimic_source.shape[0]), replace=False)
    idx_e = rng.choice(X_eicu_scaled.shape[0], size=min(SAMPLE_SIZE, X_eicu_scaled.shape[0]), replace=False)

    ot_std = ot.da.SinkhornTransport(reg_e=0.1, max_iter=500, tol=1e-4, norm="median")
    ot_std.fit(Xs=X_eicu_scaled[idx_e], Xt=X_mimic_source[idx_m])
    X_eicu_ot_std = ot_std.transform(Xs=X_eicu_scaled)
    
    std_ot_preds = model.predict_proba(X_eicu_ot_std)[:, 1]
    std_ot_auc = roc_auc_score(y_test, std_ot_preds)
    print(f"[2] Standard OT AUROC (Unconstrained)  : {std_ot_auc:.4f} (Shift: {std_ot_auc - raw_auc:+.4f})")

    # 6. Method 1: Post-Transport Empirical Interval Projection
    # Restrict mapped features within the source model's training distribution bounds
    mimic_train_min = np.percentile(X_mimic_source, 0.5, axis=0)
    mimic_train_max = np.percentile(X_mimic_source, 99.5, axis=0)
    
    X_eicu_projected = np.clip(X_eicu_ot_std, mimic_train_min, mimic_train_max)
    projected_preds = model.predict_proba(X_eicu_projected)[:, 1]
    projected_auc = roc_auc_score(y_test, projected_preds)
    print(f"[3] Post-OT Interval Projection AUROC  : {projected_auc:.4f} (Shift: {projected_auc - raw_auc:+.4f})")

    # 7. Method 2: Bio-Gated OT (Protecting Objective Biological Signals)
    bio_keywords = ["lactate", "aptt", "pt", "creatinine", "bun", "albumin", "platelet", "wbc", "bilirubin"]
    bio_mask = np.array([any(k in f.lower() for k in bio_keywords) for f in feature_names])
    
    print(f"\n[*] Auditing Bio-Gated Mask: Protecting {bio_mask.sum()} / {len(bio_mask)} objective features.")
    
    # Keep bio features raw/unperturbed, apply OT only to charting/static
    X_eicu_bio_gated = X_eicu_ot_std.copy()
    X_eicu_bio_gated[:, bio_mask] = X_eicu_scaled[:, bio_mask] 
    
    bio_gated_preds = model.predict_proba(X_eicu_bio_gated)[:, 1]
    bio_gated_auc = roc_auc_score(y_test, bio_gated_preds)
    print(f"[4] Bio-Gated OT AUROC               : {bio_gated_auc:.4f} (Shift: {bio_gated_auc - raw_auc:+.4f})")

    # 8. Spearman Rank Correlation (Diagnosing Structural Destruction)
    print("\n" + "="*60)
    print(" SPEARMAN RANK CORRELATION (MODEL LOGIC PRESERVATION)")
    print("="*60)
    print(f"Raw vs Standard OT         : {spearmanr(raw_preds, std_ot_preds).statistic:.4f}")
    print(f"Raw vs Interval Projection : {spearmanr(raw_preds, projected_preds).statistic:.4f}")
    print(f"Raw vs Bio-Gated OT        : {spearmanr(raw_preds, bio_gated_preds).statistic:.4f}")

    import json as _json
    _metrics = BASE_DIR / "outputs" / "metrics"
    _metrics.mkdir(parents=True, exist_ok=True)
    with open(_metrics / "atlas_ot_constrained_variants.json", "w") as _f:
        _json.dump({
            "raw_eicu_auroc": float(raw_auc),
            "standard_ot_auroc": float(std_ot_auc),
            "interval_projection_auroc": float(projected_auc),
            "bio_gated_auroc": float(bio_gated_auc),
            "n_features_protected_by_gate": int(bio_mask.sum()),
            "n_features_total": int(len(bio_mask)),
            "spearman_raw_vs_standard_ot": float(spearmanr(raw_preds, std_ot_preds).statistic),
            "spearman_raw_vs_interval_projection": float(spearmanr(raw_preds, projected_preds).statistic),
            "spearman_raw_vs_bio_gated": float(spearmanr(raw_preds, bio_gated_preds).statistic),
        }, _f, indent=4)

    print("\n" + "="*60)
    print(" SUMMARY OF DIAGNOSTIC OT ABLATIONS")
    print("="*60)
    print(f"Raw eICU (Locked Benchmark)  : {raw_auc:.4f}")
    print(f"Standard Sinkhorn OT         : {std_ot_auc:.4f}")
    print(f"Post-OT Interval Projection  : {projected_auc:.4f}")
    print(f"Bio-Gated Preserved OT       : {bio_gated_auc:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
