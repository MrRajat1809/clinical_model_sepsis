"""
05_eicu_sofa_calculator.py

Calculates the dynamic 6-organ SOFA score to identify an acute increase of >= 2 points.
Establishes the final eICU Sepsis-3 external validation cohort and re-evaluates 
Sepsis Onset Time as the intersection of suspected infection and organ failure.

Features included:
- Baseline Window: -48 hours up to Suspected Infection Time (SIT)
- Acute Window: SIT up to +24 hours
- Converts eICU minute offsets into hours for accurate windowing.
- Missing baseline SOFA scores (like GCS or FiO2) are clinically imputed as 0 (normal).
"""

import time
from pathlib import Path
import polars as pl

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Executing eICU dynamic SOFA calculation pipeline...")
    start_time = time.time()
    
    temporal_file = PROCESSED_DIR / "eicu_sepsis_temporal_data_cleaned.parquet"
    cohort_file = PROCESSED_DIR / "eicu_sepsis_phenotype_cohort.parquet"
    out_file = PROCESSED_DIR / "eicu_final_sepsis3_cohort.parquet"
    
    if not temporal_file.exists() or not cohort_file.exists():
        print(f"[ERROR] Required inputs not found in {PROCESSED_DIR}")
        return

    print("\n[*] Initializing Polars lazy engine and mapping eICU variables...")
    try:
        df_vitals = pl.scan_parquet(temporal_file)
        df_cohort = pl.read_parquet(cohort_file)
    except Exception as e:
        print(f"[ERROR] Failed to load data. Error: {e}")
        return

    # Map eICU string itemids to standardized SOFA variables
    itemid_map = {
        "systemicmean": "map",
        "noninvasivemean": "map",         
        "platelets x 1000": "platelets",
        "total bilirubin": "bilirubin",
        "creatinine": "creatinine",
        "pao2": "pao2",
        "vasopressor": "norepinephrine", # Treat unified eICU vasopressors as high-potency CV SOFA
        "888888": "vent"                 # Unified Mechanical Ventilation flag
    }
    
    mapping_df = pl.DataFrame({
        "itemid": list(itemid_map.keys()),
        "variable": list(itemid_map.values())
    }, schema={"itemid": pl.Utf8, "variable": pl.Utf8}).lazy()
    
    # Filter vitals to just SOFA components and map names
    df_mapped = df_vitals.join(mapping_df, on="itemid", how="inner")
    
    # Join with cohort to get SIT offset
    df_joined = df_mapped.join(df_cohort.lazy().select(["stay_id", "sit_offset"]), on="stay_id")

    print("[*] Segmenting temporal data into Baseline (-48h to SIT) and Acute (SIT to +24h) windows...")
    
    # Define dynamic windows relative to SIT using eICU minute offsets converted to hours
    df_windows = df_joined.with_columns(
        ((pl.col("event_time") - pl.col("sit_offset")) / 60.0).alias("hours_from_sit")
    ).filter(
        (pl.col("hours_from_sit") >= -48) & (pl.col("hours_from_sit") <= 24)
    ).with_columns(
        pl.when(pl.col("hours_from_sit") < 0).then(pl.lit("baseline"))
          .otherwise(pl.lit("acute")).alias("window")
    )

    print("[*] Aggregating worst physiological values per window and calculating SOFA scores...")
    
    # Find the worst values for each window
    df_agg = df_windows.group_by(["stay_id", "window", "variable"]).agg(
        pl.col("valuenum").min().alias("min_val"),
        pl.col("valuenum").max().alias("max_val")
    ).with_columns(
        pl.when(pl.col("variable").is_in(["map", "platelets", "pao2"]))
          .then(pl.col("min_val"))
          .otherwise(pl.col("max_val")).alias("worst_value")
    )

    # Pivot to wide format
    df_wide = df_agg.collect().pivot(
        values="worst_value", 
        index=["stay_id", "window"], 
        on="variable", 
        aggregate_function="first"
    )

    # Inject missing variables that eICU lacks (GCS, FiO2, specific pressors) to maintain strict MIMIC math
    expected_cols = [
        "map", "platelets", "bilirubin", "creatinine", 
        "gcs_eye", "gcs_verbal", "gcs_motor", 
        "pao2", "fio2", "dopamine", "dobutamine", 
        "epinephrine", "norepinephrine", "vent"
    ]
    
    for col in expected_cols:
        if col not in df_wide.columns:
            df_wide = df_wide.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    # Calculate individual SOFA components (missing values yield 0)
    df_sofa = df_wide.with_columns(
        (pl.col("gcs_eye").fill_null(4) + 
         pl.col("gcs_verbal").fill_null(5) + 
         pl.col("gcs_motor").fill_null(6)).alias("gcs_total")
    ).with_columns(
        pl.when(pl.col("fio2") > 1.0).then(pl.col("fio2") / 100.0)
          .otherwise(pl.col("fio2")).alias("fio2_fraction")
    ).with_columns(
        (pl.col("pao2") / pl.col("fio2_fraction")).alias("pao2_fio2_ratio")
    )

    df_sofa = df_sofa.with_columns(
        pl.when((pl.col("pao2_fio2_ratio") < 100) & (pl.col("vent") == 1.0)).then(4)
          .when((pl.col("pao2_fio2_ratio") < 200) & (pl.col("vent") == 1.0)).then(3)
          .when(pl.col("pao2_fio2_ratio") < 300).then(2)
          .when(pl.col("pao2_fio2_ratio") < 400).then(1)
          .otherwise(0).alias("sofa_resp"),
          
        pl.when(pl.col("platelets") < 20).then(4)
          .when(pl.col("platelets") < 50).then(3)
          .when(pl.col("platelets") < 100).then(2)
          .when(pl.col("platelets") < 150).then(1)
          .otherwise(0).alias("sofa_coag"),

        pl.when(pl.col("bilirubin") >= 12.0).then(4)
          .when(pl.col("bilirubin") >= 6.0).then(3)
          .when(pl.col("bilirubin") >= 2.0).then(2)
          .when(pl.col("bilirubin") >= 1.2).then(1)
          .otherwise(0).alias("sofa_liver"),

        # Strict Sepsis-3 Cardiovascular SOFA thresholds (Routing eICU generic pressors through norepi)
        pl.when(
            (pl.col("dopamine") > 15.0) | 
            (pl.col("epinephrine") > 0.1) | 
            (pl.col("norepinephrine") > 0.1)
        ).then(4)
        .when(
            ((pl.col("dopamine") > 5.0) & (pl.col("dopamine") <= 15.0)) | 
            ((pl.col("epinephrine") <= 0.1) & (pl.col("epinephrine") > 0.0)) | 
            ((pl.col("norepinephrine") <= 0.1) & (pl.col("norepinephrine") > 0.0))
        ).then(3)
        .when(
            ((pl.col("dopamine") <= 5.0) & (pl.col("dopamine") > 0.0)) | 
            (pl.col("dobutamine") > 0.0)
        ).then(2)
        .when(pl.col("map") < 70).then(1)
        .otherwise(0).alias("sofa_cv"),

        pl.when(pl.col("gcs_total") < 6).then(4)
          .when(pl.col("gcs_total") <= 9).then(3)
          .when(pl.col("gcs_total") <= 12).then(2)
          .when(pl.col("gcs_total") <= 14).then(1)
          .otherwise(0).alias("sofa_cns"),

        pl.when(pl.col("creatinine") >= 5.0).then(4)
          .when(pl.col("creatinine") >= 3.5).then(3)
          .when(pl.col("creatinine") >= 2.0).then(2)
          .when(pl.col("creatinine") >= 1.2).then(1)
          .otherwise(0).alias("sofa_renal")
    )

    df_total_sofa = df_sofa.with_columns(
        (pl.col("sofa_resp") + pl.col("sofa_coag") + pl.col("sofa_liver") + 
         pl.col("sofa_cv") + pl.col("sofa_cns") + pl.col("sofa_renal")).alias("total_sofa")
    )
    
    # Isolate Baseline metrics for downstream prediction
    df_baseline = df_total_sofa.filter(pl.col("window") == "baseline").select(
        pl.col("stay_id"),
        pl.col("total_sofa").alias("baseline_sofa"),
        pl.col("pao2_fio2_ratio").alias("baseline_pf_ratio")
    )

    df_final_pivot = df_total_sofa.pivot(
        values="total_sofa", 
        index="stay_id", 
        on="window", 
        aggregate_function="first"
    )
    
    print("[*] Imputing missing baseline SOFA scores as 0 and enforcing Sepsis-3 delta >= 2 criteria...")
    if "baseline" not in df_final_pivot.columns:
        df_final_pivot = df_final_pivot.with_columns(pl.lit(0).cast(pl.Int64).alias("baseline"))
    
    df_final_pivot = df_final_pivot.with_columns(pl.col("baseline").fill_null(0))
    df_final_pivot = df_final_pivot.drop_nulls(subset=["acute"])
    
    df_final_pivot = df_final_pivot.with_columns(
        (pl.col("acute") - pl.col("baseline")).alias("sofa_delta")
    )

    # Filter for Sepsis-3 Criteria
    sepsis3_stay_ids = df_final_pivot.filter(pl.col("sofa_delta") >= 2).select("stay_id")
    
    print("[*] Re-evaluating exact Sepsis Onset Time (intersection of infection and deterioration)...")
    
    # Find exact Sepsis Onset Time by scanning for the earliest abnormal SOFA-triggering event in the acute window
    df_abnormal_onset = df_windows.filter(
        (pl.col("window") == "acute") & (
            ((pl.col("variable") == "map") & (pl.col("valuenum") < 70)) |
            ((pl.col("variable") == "platelets") & (pl.col("valuenum") < 150)) |
            ((pl.col("variable") == "bilirubin") & (pl.col("valuenum") >= 1.2)) |
            ((pl.col("variable") == "creatinine") & (pl.col("valuenum") >= 1.2)) |
            ((pl.col("variable").is_in(["norepinephrine", "epinephrine", "dopamine", "dobutamine"])) & (pl.col("valuenum") > 0)) |
            ((pl.col("variable").is_in(["gcs_motor", "gcs_verbal", "gcs_eye"])) & (pl.col("valuenum") < 6))
        )
    ).group_by("stay_id").agg(
        pl.col("event_time").min().alias("sofa_deterioration_offset")
    ).collect()

    # Merge back to original cohort and append baseline metrics
    final_cohort = df_cohort.join(sepsis3_stay_ids, on="stay_id")
    final_cohort = final_cohort.join(df_abnormal_onset, on="stay_id", how="left")
    final_cohort = final_cohort.join(df_baseline, on="stay_id", how="left")
    
    # Impute missing baseline SOFA logic for the static feature
    final_cohort = final_cohort.with_columns(
        pl.col("baseline_sofa").fill_null(0)
    )
    
    # Set final Sepsis Onset Time to max(SIT, deterioration_time) (using offsets)
    final_cohort = final_cohort.with_columns(
        pl.max_horizontal(
            pl.col("sit_offset"), 
            pl.col("sofa_deterioration_offset").fill_null(pl.col("sit_offset"))
        ).alias("sepsis_onset_offset")
    )
    
    final_cohort.write_parquet(out_file)

    elapsed = time.time() - start_time
    print(f"\n[+] Success! eICU Sepsis-3 cohort locked in {elapsed:.2f} seconds.")
    print(f"    -> Total Validated Sepsis-3 Patients: {len(final_cohort):,}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()