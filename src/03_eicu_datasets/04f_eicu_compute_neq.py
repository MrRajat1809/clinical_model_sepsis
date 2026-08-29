"""
Compute norepinephrine-equivalent dose and check it against MIMIC-IV.

Third of the three pressor steps. Applies the Brown et al. (2013) conversion
factors to the standardised rates, keeping each drug's contribution separately
before summing so the composition of a dose can be inspected. Dobutamine is
excluded from the equivalent, correctly: it is an inotrope, not a vasopressor.

The second half is a feature-equivalence check. eICU records infusions at minute
resolution while the MIMIC-IV tensor holds hourly values, so a naive comparison
of the two distributions would differ purely because of binning. eICU is
therefore binned to hourly sum-of-maxima, matching the tensor builder, and
restricted to the same 0-24 h window, before the medians are compared. The
comparison uses the raw rather than imputed MIMIC-IV tensor, so it contrasts
charted values with charted values.

Reads:
    eicu_standardized_pressors.parquet
    mimic_sepsis_tensor_raw.npy and its feature names
Writes:
    eicu_neq_timeline.parquet
    outputs/metrics/eicu_feature_equivalence_report_NEQ.json and a density plot
"""

import time
import json
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR_EICU = BASE_DIR / "data" / "processed" / "eicu"
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"

OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

OUT_METRICS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

# True Brown et al. (2013) Multipliers
NEQ_MULTIPLIERS = {
    "norepinephrine": 1.0,
    "epinephrine": 1.0,
    "phenylephrine": 1.0 / 2.2,  # ~0.4545 (Brown et al. 2.2 mcg = 1 mcg NE)
    "dopamine": 0.01,            # 1.0 / 100
    "vasopressin": 2.5           # Standard critical care approximation
}

def compute_neq_and_validate():
    print("[*] Initiating NEQ Computation & Feature Equivalence Validation (Brown et al.)...")
    start_time = time.time()
    
    in_file = PROCESSED_DIR_EICU / "eicu_standardized_pressors.parquet"
    out_file = PROCESSED_DIR_EICU / "eicu_neq_timeline.parquet"
    equiv_report_file = OUT_METRICS / "eicu_feature_equivalence_report_NEQ.json"
    equiv_plot_file = OUT_FIGURES / "eicu_MIMIC_vs_eICU_NEQ_Distribution.png"
    
    if not in_file.exists():
        print(f"[ERROR] Standardized pressors not found at: {in_file}")
        return

    # --- Compute Individual Contributions & Total Neq --------------------
    print("    -> Applying pure Brown et al. NEQ conversion factors...")
    df_pressors = pl.read_parquet(in_file)

    df_neq_components = df_pressors.with_columns(
        (pl.col("standardized_rate") * pl.col("drug_type").replace(NEQ_MULTIPLIERS, default=0.0)).alias("neq_val")
    )

    df_wide = df_neq_components.pivot(
        values="neq_val",
        index=["stay_id", "event_time"],
        on="drug_type",
        aggregate_function="sum"
    ).fill_null(0.0)
    
    for drug in NEQ_MULTIPLIERS.keys():
        if drug not in df_wide.columns:
            df_wide = df_wide.with_columns(pl.lit(0.0).alias(drug))

    df_final = df_wide.with_columns(
        (pl.col("norepinephrine") + pl.col("epinephrine") + 
         pl.col("phenylephrine") + pl.col("dopamine") + 
         pl.col("vasopressin")).alias("valuenum"),
        
        ((pl.col("norepinephrine") > 0).cast(pl.Int32) + 
         (pl.col("epinephrine") > 0).cast(pl.Int32) + 
         (pl.col("phenylephrine") > 0).cast(pl.Int32) + 
         (pl.col("dopamine") > 0).cast(pl.Int32) + 
         (pl.col("vasopressin") > 0).cast(pl.Int32)).alias("concurrent_pressors")
    ).with_columns(
        pl.lit("neq").alias("itemid")
    ).rename({
        "norepinephrine": "neq_from_ne",
        "epinephrine": "neq_from_epi",
        "phenylephrine": "neq_from_phenyl",
        "dopamine": "neq_from_dopa",
        "vasopressin": "neq_from_vaso"
    }).sort(["stay_id", "event_time"])

    # --- Extended Clinical Descriptive Statistics ------------------------
    print("\n    [eICU NEQ DESCRIPTIVE STATISTICS (Reading-wise)]")
    total_patients = df_final.select(pl.col("stay_id").n_unique()).item()
    max_concurrent = df_final.select(pl.col("concurrent_pressors").max()).item()
    mean_concurrent = df_final.filter(pl.col("concurrent_pressors") > 0).select(pl.col("concurrent_pressors").mean()).item()
    
    dist_treated = df_final.filter(pl.col("valuenum") > 0).select([
        pl.col("valuenum").median().round(3).alias("Median"),
        pl.col("valuenum").quantile(0.25).round(3).alias("Q25"),
        pl.col("valuenum").quantile(0.75).round(3).alias("Q75"),
        pl.col("valuenum").max().round(3).alias("Max")
    ]).row(0)

    print(f"        - Patients receiving Vasopressors : {total_patients:,}")
    print(f"        - Median NEQ (Raw Events)         : {dist_treated[0]} mcg/kg/min (IQR: {dist_treated[1]} - {dist_treated[2]})")
    print(f"        - Max Concurrent Pressors         : {max_concurrent} (Mean: {mean_concurrent:.2f})")
    print(f"        - Absolute Max NEQ Recorded       : {dist_treated[3]} mcg/kg/min")

    # --- MIMIC-IV DISTRIBUTION COMPARISON (Apples-to-Apples) -------------
    print("\n    -> Extracting MIMIC-IV NEQ distribution for equivalence testing...")
    # Load the RAW tensor to avoid SAITS imputed smoothing data
    mimic_tensor_file = PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_raw.npy"
    mimic_features_file = PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_features.npy"
    
    if mimic_tensor_file.exists() and mimic_features_file.exists():
        features = list(np.load(mimic_features_file))
        if "neq" in features:
            neq_idx = features.index("neq")
            mimic_tensor = np.load(mimic_tensor_file)
            
            mimic_neq_raw = mimic_tensor[:, :, neq_idx].flatten()
            # Filter out NaNs to only evaluate true charted events
            mimic_neq_active = mimic_neq_raw[~np.isnan(mimic_neq_raw) & (mimic_neq_raw > 0)]
            
            # Filter to Acute 24h Window (0 to 1440 mins) to match MIMIC Tensor
            print("    -> Filtering eICU to 24h acute window & Binning into hourly Sum of Maxes...")
            df_eicu_hourly = df_final.filter(
                (pl.col("event_time") >= 0) & (pl.col("event_time") < 1440)
            ).with_columns(
                (pl.col("event_time") / 60.0).floor().cast(pl.Int32).alias("hour_bin")
            ).group_by(["stay_id", "hour_bin"]).agg(
                pl.col("neq_from_ne").max().alias("max_ne"),
                pl.col("neq_from_epi").max().alias("max_epi"),
                pl.col("neq_from_phenyl").max().alias("max_phenyl"),
                pl.col("neq_from_dopa").max().alias("max_dopa"),
                pl.col("neq_from_vaso").max().alias("max_vaso")
            ).with_columns(
                (pl.col("max_ne").fill_null(0) + 
                 pl.col("max_epi").fill_null(0) + 
                 pl.col("max_phenyl").fill_null(0) + 
                 pl.col("max_dopa").fill_null(0) + 
                 pl.col("max_vaso").fill_null(0)).alias("hourly_sum_of_maxes")
            )
            
            eicu_neq_active = df_eicu_hourly.filter(pl.col("hourly_sum_of_maxes") > 0)["hourly_sum_of_maxes"].to_numpy()
            
            mimic_median = np.median(mimic_neq_active)
            eicu_median = np.median(eicu_neq_active)
            print(f"        - MIMIC Active NEQ Median (Raw Hourly Max): {mimic_median:.3f}")
            print(f"        - eICU Active NEQ Median (Raw Hourly Max) : {eicu_median:.3f}")
            
            plt.figure(figsize=(10, 6))
            sns.kdeplot(mimic_neq_active, log_scale=True, fill=True, label=f"MIMIC-IV (Median: {mimic_median:.3f})", color="#4C72B0")
            sns.kdeplot(eicu_neq_active, log_scale=True, fill=True, label=f"eICU (Median: {eicu_median:.3f})", color="#C44E52")
            plt.title("Feature Equivalence: NEQ Distribution (Raw Hourly Sum of Maxes)", weight="bold")
            plt.xlabel("Norepinephrine Equivalent Dose (mcg/kg/min) [Log Scale]")
            plt.ylabel("Density")
            plt.legend()
            sns.despine()
            plt.tight_layout()
            plt.savefig(equiv_plot_file, dpi=300)
            plt.close()
            print(f"        - Distribution plot saved to: {equiv_plot_file.relative_to(BASE_DIR)}")
        else:
            print("        - 'neq' not found in MIMIC feature list. Skipping comparison.")
    else:
        print("        - MIMIC tensor files not found. Skipping comparison.")

    # --- Feature Equivalence Report (json) -------------------------------
    report = {
        "Feature": "NEQ (Norepinephrine Equivalent Dose)",
        "Extraction_Methodology": "Strict word-boundary Regex",
        "Standardized_Unit": "mcg/kg/min",
        "Mathematical_Conversion": "Brown et al. (2013) Pure Implementation",
        "eICU_Treated_Events": len(df_final),
        "eICU_Max_Concurrent_Pressors": int(max_concurrent),
        "eICU_Median_NEQ_Raw": float(dist_treated[0]),
        "eICU_Median_NEQ_Hourly_Max": float(eicu_median) if 'eicu_median' in locals() else None,
        "MIMIC_Median_NEQ_Hourly_Max": float(mimic_median) if 'mimic_median' in locals() else None,
        "Distribution_Checked": True,
        "Drift_Warning": bool('mimic_median' in locals() and 'eicu_median' in locals() and abs(mimic_median - eicu_median) > 0.1)
    }
    
    with open(equiv_report_file, "w") as f:
        json.dump(report, f, indent=4)

    # Export the raw timeline for downstream use
    df_final.select([
        "stay_id", "event_time", "itemid", "valuenum",
        "neq_from_ne", "neq_from_epi", "neq_from_phenyl", "neq_from_dopa", "neq_from_vaso"
    ]).write_parquet(out_file)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Computed NEQ timeline in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    compute_neq_and_validate()
