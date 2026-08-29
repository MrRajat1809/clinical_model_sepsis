"""
Sensitivity of the transport result to the entropic regularisation parameter.

Guards against the objection that the transport finding is an artefact of one
arbitrary hyperparameter. The alignment is repeated across five orders of
regularisation strength and the locked model is evaluated on each result; if the
effect on external discrimination holds across all of them, it is a property of
the method rather than of the setting.

Operates directly in the standardised feature space the model consumes, so each
transported representation can be scored without further transformation.

For tractability, each transport plan is fitted on random 3000-patient subsamples
from each cohort and then applied out of sample to the complete external cohort.
Fitting on the full cross-product would be prohibitive and adds little, since the
plan is estimated from the distributions rather than from individuals.

An assertion on the unaligned baseline AUROC catches a scaling mismatch before
the sweep runs, which would otherwise produce five equally wrong numbers.

Reads:
    both imputed tensors and cohort tables
    outputs/models/mimic_champion_xgboost.joblib, the shared split indices
Writes:
    outputs/metrics/atlas_ot_epsilon_sweep.json
"""

import time
import joblib
import warnings
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import ot
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]

OUT_MODELS = BASE_DIR / "outputs" / "models"
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"

def main():
    print("[*] Initiating Accelerated Optimal Transport Sweep...")

    # 1. LOAD EXACT TRAINING SCALERS
    mimic_tensor = np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_imputed_tensor.npy")
    mimic_stay_ids = np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_stay_ids.npy")
    mimic_train_idx = np.load(OUT_MODELS / "mimic_train_indices.npy")
    df_mimic = pl.read_parquet(PROCESSED_DIR_MIMIC / "mimic_final_sepsis3_cohort.parquet").to_pandas()

    df_mimic = pd.DataFrame({"stay_id": mimic_stay_ids}).merge(df_mimic, on="stay_id", how="left")
    static_cols = ["age", "baseline_sofa"]

    df_mimic_static = df_mimic[static_cols].copy()
    X_mimic_static_raw = df_mimic_static.fillna(0).values

    # Fitted only on mimic_train_idx
    scaler_static = StandardScaler()
    scaler_static.fit(X_mimic_static_raw[mimic_train_idx])

    X_mimic_temporal_raw = np.concatenate([
        np.mean(mimic_tensor, axis=1), np.min(mimic_tensor, axis=1),
        np.max(mimic_tensor, axis=1), np.std(mimic_tensor, axis=1)
    ], axis=1)

    # Fitted only on mimic_train_idx
    scaler_temporal = StandardScaler()
    scaler_temporal.fit(X_mimic_temporal_raw[mimic_train_idx])

    X_mimic_model_ready = np.concatenate([
        scaler_static.transform(X_mimic_static_raw),
        scaler_temporal.transform(X_mimic_temporal_raw)
    ], axis=1)

    # 2. EXTRACT TARGET DATA (eICU)
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

    X_eicu_model_ready = np.concatenate([
        scaler_static.transform(X_eicu_static_raw),
        scaler_temporal.transform(X_eicu_temporal_raw)
    ], axis=1)

    # 3. BASELINE EVALUATION
    model = joblib.load(OUT_MODELS / "mimic_champion_xgboost.joblib")
    raw_preds = model.predict_proba(X_eicu_model_ready)[:, 1]
    raw_auc = roc_auc_score(y_test, raw_preds)

    print(f"\n[BASELINE] Raw eICU AUROC (No OT) : {raw_auc:.4f}")
    assert raw_auc > 0.75, "Baseline AUROC is critically low. Scaling mismatch detected."

    # 4. FAST OT HYPERPARAMETER SWEEP
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = []

    # Subsample parameters for Out-of-Sample mapping
    SAMPLE_SIZE = 3000
    rng = np.random.default_rng(42)
    
    idx_m = rng.choice(X_mimic_model_ready.shape[0], size=min(SAMPLE_SIZE, X_mimic_model_ready.shape[0]), replace=False)
    idx_e = rng.choice(X_eicu_model_ready.shape[0], size=min(SAMPLE_SIZE, X_eicu_model_ready.shape[0]), replace=False)
    
    Xt_fit = X_mimic_model_ready[idx_m]
    Xs_fit = X_eicu_model_ready[idx_e]

    print(f"\n[*] Subsampled OT Fitting Matrix: {SAMPLE_SIZE}x{SAMPLE_SIZE} (10x Speedup)")
    print("=======================================================")
    print(" SINKHORN OT SENSITIVITY ANALYSIS (Target = MIMIC-IV)")
    print("=======================================================\n")

    for reg in epsilons:
        start_time = time.time()
        print(f"--- Running OT Mapping (reg_e = {reg}) ---")
        try:
            ot_mapping = ot.da.SinkhornTransport(
                reg_e=reg, max_iter=500, tol=1e-4, norm="median", verbose=True
            )
            
            # Fit on the lightweight subsampled arrays
            ot_mapping.fit(Xs=Xs_fit, Xt=Xt_fit)

            # Transform the FULL eICU array out-of-sample
            X_eicu_ot = ot_mapping.transform(Xs=X_eicu_model_ready)

            ot_preds = model.predict_proba(X_eicu_ot)[:, 1]
            ot_auc = roc_auc_score(y_test, ot_preds)
            shift = ot_auc - raw_auc

            elapsed = time.time() - start_time
            results.append((reg, ot_auc, shift))
            print(f"-> Done in {elapsed:.1f}s | AUROC: {ot_auc:.4f} (Shift: {shift:+.4f})\n")

        except Exception as e:
            print(f"-> Failed: {e}\n")
            results.append((reg, np.nan, np.nan))

    import json as _json
    _metrics = BASE_DIR / "outputs" / "metrics"
    _metrics.mkdir(parents=True, exist_ok=True)
    with open(_metrics / "atlas_ot_epsilon_sweep.json", "w") as _f:
        _json.dump({"baseline_raw_eicu_auroc": float(raw_auc),
                    "sample_size_per_cohort": SAMPLE_SIZE,
                    "sweep": [{"epsilon": r[0],
                               "ot_auroc": None if pd.isna(r[1]) else float(r[1]),
                               "shift_from_raw": None if pd.isna(r[2]) else float(r[2])}
                              for r in results]}, _f, indent=4)

    print("\n=======================================================")
    print(" FINAL SWEEP SUMMARY")
    print("=======================================================")
    print(f"{'Epsilon (reg_e)':<15} | {'OT-eICU AUROC':<15} | {'Shift from Raw'}")
    print("-" * 55)

    for reg, auc, shift in results:
        if pd.isna(auc):
            print(f"{reg:<15} | {'Failed':<15} | {'N/A'}")
        else:
            marker = "🔴 Destructive" if shift < -0.01 else "🟡 Minor Impact"
            print(f"{reg:<15} | {auc:<15.4f} | {shift:+.4f} ({marker})")

    print("=======================================================\n")

if __name__ == "__main__":
    main()
