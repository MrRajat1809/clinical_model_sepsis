"""
Does transport stabilise performance across individual hospitals.

Aggregate external AUROC hides between-site variation: a model can look
acceptable overall while failing at particular hospitals. eICU spans many
centres, so hospital identifiers are joined back from the raw patient table and
the ten largest contributing hospitals are scored separately.

The locked model is applied twice per hospital, once to the unaligned
representation and once to the transport-aligned one, and the dispersion of
hospital-level AUROCs is compared before and after. A fall in standard deviation
means alignment made performance more uniform across sites, which is a different
claim from improving the mean and can be true even when the mean falls.

Hospitals with only one outcome class are skipped, since AUROC is undefined
there.

Reads:
    data/raw/eicu-crd/2.0/patient for hospital identifiers
    the atlas features and metadata, the eICU tensors and cohort
    outputs/models/mimic_champion_xgboost.joblib, the shared split indices
Writes:
    outputs/analysis/eicu_multicenter_variance_report.csv and a dumbbell plot
"""

import time
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_EICU_PATIENT = BASE_DIR / "data" / "raw" / "eicu-crd" / "2.0" / "patient.csv.gz"

PROCESSED_DIR_ATLAS = BASE_DIR / "data" / "processed" / "atlas"
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"
OUT_MODELS = BASE_DIR / "outputs" / "models"

ATLAS_FEATURES_FILE = PROCESSED_DIR_ATLAS / "atlas_sepsis_features.npy"
ATLAS_META_FILE = PROCESSED_DIR_ATLAS / "atlas_metadata.parquet"

EICU_TENSOR_FILE = PROCESSED_DIR_EICU / "eicu_sepsis_imputed_tensor.npy"
EICU_STAY_ID_FILE = PROCESSED_DIR_EICU / "eicu_sepsis_tensor_stay_ids.npy"
EICU_COHORT_FILE = PROCESSED_DIR_EICU / "eicu_final_sepsis3_cohort.parquet"

MIMIC_TENSOR_FILE = PROCESSED_DIR_MIMIC / "mimic_sepsis_imputed_tensor.npy"
MIMIC_STAY_ID_FILE = PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_stay_ids.npy"
MIMIC_COHORT_FILE = PROCESSED_DIR_MIMIC / "mimic_final_sepsis3_cohort.parquet"
MIMIC_TRAIN_IDX_FILE = OUT_MODELS / "mimic_train_indices.npy"

XGB_MODEL_FILE = OUT_MODELS / "mimic_champion_xgboost.joblib"

# Outputs (The newly permitted folder)
OUT_ANALYSIS = BASE_DIR / "outputs" / "analysis"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"
OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUT_ANALYSIS / "eicu_multicenter_variance_report.csv"
PLOT_FILE = OUT_FIGURES / "eicu_multicenter_variance_plot.png"

def main():
    print("[*] Initiating Multi-Center Variance Statistical Proof...")
    start_time = time.time()

    # --- Load Hospital Mappings ------------------------------------------
    print("    -> Linking cohort to raw hospital IDs...")
    df_raw_patients = pd.read_csv(RAW_EICU_PATIENT, usecols=['patientunitstayid', 'hospitalid'])
    df_raw_patients = df_raw_patients.rename(columns={'patientunitstayid': 'stay_id'})
    
    eicu_stay_ids = np.load(EICU_STAY_ID_FILE)
    df_eicu_cohort = pl.read_parquet(EICU_COHORT_FILE).to_pandas()
    df_eicu = pd.DataFrame({"stay_id": eicu_stay_ids}).merge(df_eicu_cohort, on="stay_id", how="left")
    
    # Merge hospital IDs
    df_eicu = df_eicu.merge(df_raw_patients, on="stay_id", how="left")
    y_eicu = df_eicu["hospital_expire_flag"].values
    hospital_ids = df_eicu["hospitalid"].values

    # --- Reconstruct Scalers & Load Model --------------------------------
    print("    -> Reconstructing original scalers and loading model...")
    mimic_tensor = np.load(MIMIC_TENSOR_FILE)
    mimic_train_idx = np.load(MIMIC_TRAIN_IDX_FILE)
    # Load tensor stay_ids to enforce alignment
    mimic_stay_ids = np.load(MIMIC_STAY_ID_FILE)
    df_mimic = pl.read_parquet(MIMIC_COHORT_FILE).to_pandas()
    # Align df_mimic to match the exact tensor sorting
    df_mimic = pd.DataFrame({"stay_id": mimic_stay_ids}).merge(df_mimic, on="stay_id", how="left")
    df_mimic_static = df_mimic[["age", "baseline_sofa"]].copy()
    scaler_static = StandardScaler().fit(df_mimic_static.fillna(0).values[mimic_train_idx])

    mimic_temporal_raw = np.concatenate([
        np.mean(mimic_tensor, axis=1), np.min(mimic_tensor, axis=1),
        np.max(mimic_tensor, axis=1), np.std(mimic_tensor, axis=1)
    ], axis=1)
    scaler_temporal = StandardScaler().fit(mimic_temporal_raw[mimic_train_idx])
    
    champion_xgb = joblib.load(XGB_MODEL_FILE)

    # --- Prepare Raw and Ot Data -----------------------------------------
    print("    -> Formatting Pre-OT (Raw) and Post-OT (Atlas) Tensors...")
    
    # RAW (Pre-OT)
    eicu_tensor = np.load(EICU_TENSOR_FILE)
    df_eicu_static = df_eicu[["age", "baseline_sofa"]].copy()
    X_raw_static_scaled = scaler_static.transform(df_eicu_static.fillna(0).values)

    eicu_temporal_raw = np.concatenate([
        np.mean(eicu_tensor, axis=1), np.min(eicu_tensor, axis=1),
        np.max(eicu_tensor, axis=1), np.std(eicu_tensor, axis=1)
    ], axis=1)
    X_raw_temporal_scaled = scaler_temporal.transform(eicu_temporal_raw)
    X_test_raw = np.concatenate([X_raw_static_scaled, X_raw_temporal_scaled], axis=1)
    preds_raw = champion_xgb.predict_proba(X_test_raw)[:, 1]

    # OT (Post-OT)
    X_atlas = np.load(ATLAS_FEATURES_FILE)
    df_meta = pd.read_parquet(ATLAS_META_FILE)
    eicu_mask = df_meta["cohort_source"] == "eICU-CRD"
    X_eicu_ot = X_atlas[eicu_mask]
    
    ATLAS_SRC = X_eicu_ot

    # Atlas layout, written by 04_atlas_datasets/01a: temporal block first, then
    # the static block in MODEL_STATICS order. Deriving the split from the names
    # means a change to the static set fails loudly rather than silently
    # selecting the wrong columns.
    MODEL_STATICS = ["age", "baseline_sofa"]
    N_TEMPORAL = ATLAS_SRC.shape[1] - len(MODEL_STATICS)
    assert N_TEMPORAL == 120, f"expected 120 temporal columns, got {N_TEMPORAL}"

    X_ot_temporal = X_eicu_ot[:, :N_TEMPORAL]
    X_ot_static = X_eicu_ot[:, N_TEMPORAL:]

    static_perm = [MODEL_STATICS.index(c) for c in ["age", "baseline_sofa"]]
    X_ot_static_reordered = X_ot_static[:, static_perm]
    
    X_ot_static_scaled = scaler_static.transform(X_ot_static_reordered)
    X_ot_temporal_scaled = scaler_temporal.transform(X_ot_temporal)
    X_test_ot = np.concatenate([X_ot_static_scaled, X_ot_temporal_scaled], axis=1)
    preds_ot = champion_xgb.predict_proba(X_test_ot)[:, 1]

    # --- Multi-center Analysis (top 10 Hospitals) ------------------------
    print("    -> Computing metrics per hospital...")
    
    # Identify Top 10 Hospitals with at least some mortality to allow AUROC calculation
    top_hospitals = df_eicu["hospitalid"].value_counts().head(10).index.tolist()
    
    results = []
    for hosp_id in top_hospitals:
        mask = (hospital_ids == hosp_id)
        y_true_hosp = y_eicu[mask]
        
        if len(np.unique(y_true_hosp)) < 2:
            continue # Skip if no deaths in this subset
            
        auc_raw = roc_auc_score(y_true_hosp, preds_raw[mask])
        auc_ot = roc_auc_score(y_true_hosp, preds_ot[mask])
        n_patients = mask.sum()
        mortality_rate = y_true_hosp.mean() * 100
        
        results.append({
            "Hospital_ID": f"Hosp_{hosp_id}",
            "N_Patients": n_patients,
            "Mortality_Rate_%": mortality_rate,
            "Pre_OT_AUROC": auc_raw,
            "Post_OT_AUROC": auc_ot,
            "Delta_AUROC": auc_ot - auc_raw
        })

    df_results = pd.DataFrame(results)
    
    var_raw = df_results["Pre_OT_AUROC"].std()
    var_ot = df_results["Post_OT_AUROC"].std()
    
    print("\n" + "="*60)
    print(" MULTI-CENTER VARIANCE TEST (TOP 10 HOSPITALS)")
    print("="*60)
    print(f"    Global Pre-OT AUROC Std. Dev : {var_raw:.4f}")
    print(f"    Global Post-OT AUROC Std. Dev: {var_ot:.4f}")
    print(f"    Variance Reduction           : {((var_raw - var_ot)/var_raw)*100:.1f}%")
    print("="*60)
    
    df_results.to_csv(REPORT_FILE, index=False)

    # --- Visualization (dumbbell Plot) -----------------------------------
    print("    -> Generating Variance Stabilization Plot...")
    plt.figure(figsize=(10, 6))
    
    # Sort hospitals by Pre_OT_AUROC for a cleaner aesthetic
    df_results = df_results.sort_values(by="Pre_OT_AUROC", ascending=True).reset_index(drop=True)
    
    for i, row in df_results.iterrows():
        # Draw the connecting line
        plt.plot([row["Pre_OT_AUROC"], row["Post_OT_AUROC"]], [i, i], color="gray", zorder=1, linestyle="--")
        # Draw dots
        plt.scatter(row["Pre_OT_AUROC"], i, color="#4C72B0", s=100, zorder=2, label="Raw eICU (Pre-OT)" if i==0 else "")
        plt.scatter(row["Post_OT_AUROC"], i, color="#C44E52", s=100, zorder=3, label="Harmonized (Post-OT)" if i==0 else "")

    plt.yticks(range(len(df_results)), df_results["Hospital_ID"])
    plt.xlabel("AUROC", weight="bold")
    plt.title("Cross-Hospital Variance Stabilization via Optimal Transport", weight="bold", fontsize=14)
    
    # Highlight the variance compression
    plt.axvline(df_results["Pre_OT_AUROC"].mean(), color="#4C72B0", linestyle=":", alpha=0.6)
    plt.axvline(df_results["Post_OT_AUROC"].mean(), color="#C44E52", linestyle=":", alpha=0.6)
    
    plt.legend(frameon=True, loc="lower right")
    sns.despine()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n[+] Success! Completed in {elapsed:.2f} seconds.")
    print(f"    -> Report saved to: {REPORT_FILE.relative_to(BASE_DIR)}")
    print(f"    -> Plot saved to: {PLOT_FILE.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
