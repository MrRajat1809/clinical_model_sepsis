"""
07a_mimic_tensor_builder.py

Reshapes the cleaned temporal extraction into a dense 3D tensor [Patients, 24 Steps, Features].
Creates parallel static feature vectors and mortality label arrays for downstream ML.
Bins the first 24 hours post-onset into consistent 1-hour intervals.

Features included:
- Added Urine Output (UO), aggregated via sum() per 1-hour bin.
- Replaced synthetic NEQ logic with the validated Brown et al. (2013) 
  pharmacological conversion formula, computed dynamically from raw pressors.
- Phenylephrine divisor explicitly updated to 2.2 to exactly match Brown et al. 
  equivalencies (1 mcg NEQ = 2.2 mcg Phenylephrine).
- Maintained tensor compactness by dropping raw pressors post-NEQ calculation.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

def main():
    print("[*] Executing Advanced MIMIC-IV Tensor Generation Pipeline (Brown et al. Corrected)...")
    start_time = time.time()
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read explicitly named input files
    cohort_file = PROCESSED_DIR / "mimic_final_sepsis3_cohort.parquet"
    temporal_file = PROCESSED_DIR / "mimic_sepsis_temporal_data_cleaned.parquet"
    
    # 3D Tensor Outputs (Flattened and Prefixed)
    out_tensor = PROCESSED_DIR / "mimic_sepsis_tensor_raw.npy"
    id_file = PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy"
    feature_file = PROCESSED_DIR / "mimic_sepsis_tensor_features.npy"
    policy_file = PROCESSED_DIR / "mimic_aggregation_policy.npy"
    mask_file = PROCESSED_DIR / "mimic_sepsis_tensor_mask.npy"
    
    # Static & Label Outputs (Flattened and Prefixed)
    static_file = PROCESSED_DIR / "mimic_sepsis_tensor_static.npy"
    static_feature_file = PROCESSED_DIR / "mimic_sepsis_tensor_static_features.npy"
    label_file = PROCESSED_DIR / "mimic_sepsis_tensor_labels.npy"
    
    if not temporal_file.exists():
        print(f"[ERROR] Cleaned temporal data not found at: {temporal_file}")
        return

    # ---------------------------------------------------------
    # 1. LOAD DATA & MAP CHANNELS
    # ---------------------------------------------------------
    print("    -> Loading cohort demographics and mapping clinical features...")
    
    df_cohort = pl.read_parquet(cohort_file).select([
        "stay_id", "sepsis_onset_time", "age", "gender", "race",
        "admission_type", "first_careunit", "charlson_comorbidity_index", 
        "baseline_sofa", "baseline_pf_ratio", "hospital_expire_flag"
    ])
    
    # Convert gender to binary (M=1, F=0) for numerical matrix
    df_cohort = df_cohort.with_columns(
        pl.when(pl.col("gender") == "M").then(1).otherwise(0).alias("gender")
    )
    
    df_vitals = pl.scan_parquet(temporal_file)
    
    # Upgraded feature map with raw vasopressors and UO
    feature_map = {
        # Vitals
        220045: "hr", 220181: "map", 220052: "map", 220210: "rr", 
        223762: "temp_c", 223761: "temp_c", 220277: "spo2",
        # Labs
        51301: "wbc", 51300: "wbc", 51265: "platelets", 227457: "platelets", 
        51222: "hemoglobin", 50912: "creatinine", 220615: "creatinine", 
        51006: "bun", 50885: "bilirubin", 225664: "bilirubin", 
        50820: "ph", 50813: "lactate", 227442: "lactate", 
        51274: "pt", 51275: "aptt", 50862: "albumin", 220862: "albumin", 
        50971: "potassium", 50822: "potassium", 50983: "sodium", 
        50824: "sodium", 50931: "glucose", 50809: "glucose", 
        50902: "chloride", 50806: "chloride", 
        # Respiratory & Neuro
        50818: "paco2", 50821: "pao2", 223835: "fio2", 
        220739: "gcs_eye", 223900: "gcs_verbal", 223901: "gcs_motor",
        # Raw Interventions
        221906: "norepinephrine", 221289: "epinephrine", 221662: "dopamine", 
        221653: "dobutamine", 222315: "vasopressin", 221749: "phenylephrine",
        888888: "vent",
        # Urine Output
        226559: "urine_output", 226560: "urine_output", 226561: "urine_output", 
        226584: "urine_output", 226563: "urine_output", 226564: "urine_output", 
        226565: "urine_output", 226567: "urine_output", 226557: "urine_output", 
        226558: "urine_output"
    }
    
    mapping_df = pl.DataFrame({
        "itemid": list(feature_map.keys()),
        "feature": list(feature_map.values())
    }, schema={"itemid": pl.Int64, "feature": pl.Utf8})

    # ---------------------------------------------------------
    # 2. TEMPORAL BINNING (24 UNIFORM 1-HOUR BINS)
    # ---------------------------------------------------------
    print("    -> Filtering temporal window and enforcing 24x 1-hour bins...")
    df_joined = df_vitals.join(df_cohort.lazy().select(["stay_id", "sepsis_onset_time"]), on="stay_id")
    df_joined = df_joined.join(mapping_df.lazy(), on="itemid")
    
    df_window = df_joined.with_columns(
        ((pl.col("event_time") - pl.col("sepsis_onset_time")).dt.total_hours()).alias("hours_from_onset")
    ).filter(
        (pl.col("hours_from_onset") >= 0) & (pl.col("hours_from_onset") < 24)
    )

    df_binned = df_window.with_columns(
        pl.col("hours_from_onset").floor().cast(pl.Int32).alias("time_step")
    ).collect()

    # ---------------------------------------------------------
    # 3. CLINICAL AGGREGATION
    # ---------------------------------------------------------
    print("    -> Applying physiological aggregation logic (Mean/Max/Min/Sum) per bin...")
    
    mean_vars = ["temp_c", "sodium", "potassium", "chloride"]
    min_vars  = ["map", "spo2", "platelets", "hemoglobin", "albumin", "ph", "gcs_eye", "gcs_verbal", "gcs_motor"]
    sum_vars  = ["urine_output"]
    max_vars  = [
        "hr", "rr", "lactate", "bun", "bilirubin", "pt", "aptt", "wbc", "glucose", "paco2", 
        "creatinine", "pao2", "fio2", "vent", "norepinephrine", "epinephrine", 
        "dopamine", "dobutamine", "vasopressin", "phenylephrine"
    ]
    
    df_agg = df_binned.group_by(["stay_id", "time_step", "feature"]).agg(
        pl.col("valuenum").mean().alias("val_mean"),
        pl.col("valuenum").max().alias("val_max"),
        pl.col("valuenum").min().alias("val_min"),
        pl.col("valuenum").sum().alias("val_sum"),
        pl.col("valuenum").last().alias("val_last")
    ).with_columns(
        pl.when(pl.col("feature").is_in(mean_vars)).then(pl.col("val_mean"))
        .when(pl.col("feature").is_in(max_vars)).then(pl.col("val_max"))
        .when(pl.col("feature").is_in(min_vars)).then(pl.col("val_min"))
        .when(pl.col("feature").is_in(sum_vars)).then(pl.col("val_sum"))
        .otherwise(pl.col("val_last")).alias("val")
    ).select(["stay_id", "time_step", "feature", "val"])

    df_wide = df_agg.pivot(
        values="val", index=["stay_id", "time_step"], on="feature", aggregate_function="first"
    ).to_pandas()

    # ---------------------------------------------------------
    # 4. FEATURE ENGINEERING (P/F RATIO & NEQ)
    # ---------------------------------------------------------
    print("    -> Engineering PaO2/FiO2 ratio and validated NEQ conversion...")
    
    # PaO2/FiO2 Ratio
    if "fio2" in df_wide.columns and "pao2" in df_wide.columns:
        df_wide["fio2"] = np.where(df_wide["fio2"] > 1.0, df_wide["fio2"] / 100.0, df_wide["fio2"])
        df_wide["pf_ratio"] = df_wide["pao2"] / df_wide["fio2"]
    else:
        df_wide["pf_ratio"] = np.nan

    # NEQ Calculation (Brown et al. 2013 true standardization)
    pressors = ["norepinephrine", "epinephrine", "phenylephrine", "dopamine", "vasopressin"]
    for p in pressors:
        if p not in df_wide.columns:
            df_wide[p] = np.nan

    # Calculate NEQ only for rows where at least one vasopressor was administered
    has_pressor = df_wide[pressors].notna().any(axis=1)
    df_wide.loc[has_pressor, "neq"] = (
        df_wide["norepinephrine"].fillna(0) +
        df_wide["epinephrine"].fillna(0) +
        (df_wide["phenylephrine"].fillna(0) / 2.2) +
        (df_wide["dopamine"].fillna(0) / 100.0) +
        (df_wide["vasopressin"].fillna(0) * 2.5)
    )
    df_wide.loc[~has_pressor, "neq"] = np.nan

    # Explicit deterministic feature order (raw pressors are intentionally dropped here)
    FEATURE_ORDER = [
        "hr", "map", "rr", "temp_c", "spo2", 
        "gcs_eye", "gcs_verbal", "gcs_motor", 
        "pao2", "fio2", "pf_ratio", "paco2", 
        "lactate", "creatinine", "bun", "bilirubin", "platelets", "wbc", "hemoglobin", 
        "ph", "pt", "aptt", "albumin", "potassium", "sodium", "glucose", "chloride", 
        "urine_output", "neq", "vent"
    ]

    for f in FEATURE_ORDER:
        if f not in df_wide.columns:
            df_wide[f] = np.nan

    # ---------------------------------------------------------
    # 5. TENSOR RESHAPING
    # ---------------------------------------------------------
    print("    -> Reshaping data into dense 3D Tensor format...")
    stay_ids = sorted(df_cohort["stay_id"].to_list())

    multi_idx = pd.MultiIndex.from_product([stay_ids, range(24)], names=["stay_id", "time_step"])
    df_wide = df_wide.set_index(["stay_id", "time_step"])
    df_tensor = df_wide[FEATURE_ORDER].reindex(multi_idx)
    
    X_3d = df_tensor.values.reshape(len(stay_ids), 24, len(FEATURE_ORDER))
    missingness_mask = np.isnan(X_3d)
    
    print(f"        - 3D Temporal Shape : {X_3d.shape} [Patients, Steps, Features]")
    print(f"        - Missingness Rate  : {missingness_mask.mean() * 100:.2f}%")

    # ---------------------------------------------------------
    # 6. STATIC & LABEL EXTRACTION
    # ---------------------------------------------------------
    print("    -> Extracting parallel 2D Static Context and 1D Labels...")
    df_static = df_cohort.to_pandas().set_index("stay_id").reindex(stay_ids)
    
    # Label encode categorical variables for the numeric array
    cat_cols = ["race", "admission_type", "first_careunit"]
    for c in cat_cols:
        df_static[c] = df_static[c].astype("category").cat.codes
        
    static_cols = [
        "age", "gender", "race", "admission_type", "first_careunit", 
        "charlson_comorbidity_index", "baseline_sofa", "baseline_pf_ratio"
    ]
    
    X_static = df_static[static_cols].values
    y_labels = df_static["hospital_expire_flag"].fillna(0).astype(np.int64).values
    
    print(f"        - 2D Static Shape   : {X_static.shape}")
    print(f"        - 1D Target Shape   : {y_labels.shape}")

    # ---------------------------------------------------------
    # 7. SERIALIZATION
    # ---------------------------------------------------------
    np.save(out_tensor, X_3d)
    np.save(id_file, np.array(stay_ids))
    np.save(feature_file, np.array(FEATURE_ORDER))
    np.save(mask_file, missingness_mask)
    
    np.save(static_file, X_static)
    np.save(static_feature_file, np.array(static_cols))
    np.save(label_file, y_labels)
    
    aggregation_policy = {}
    for f in FEATURE_ORDER:
        if f in mean_vars:
            aggregation_policy[f] = "mean"
        elif f in max_vars:
            aggregation_policy[f] = "max"
        elif f in min_vars:
            aggregation_policy[f] = "min"
        elif f in sum_vars:
            aggregation_policy[f] = "sum"
        elif f in ["pf_ratio", "neq"]:
            aggregation_policy[f] = "engineered"
        else:
            aggregation_policy[f] = "last"
            
    np.save(policy_file, aggregation_policy)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! All arrays saved to {PROCESSED_DIR.relative_to(BASE_DIR)} in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()