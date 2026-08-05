"""
04d_eicu_extract_pressors.py

Phase 9: External Validation (eICU Vasopressor Extraction)
Uses strict word boundaries to identify Norepinephrine, Epinephrine, Vasopressin, 
Dopamine, and Phenylephrine from the eICU infusionDrug table.
Extracts the embedded unit string (e.g., 'mcg/kg/min' from 'norepinephrine (mcg/kg/min)') 
and prioritizes the `drugrate` column without applying any mathematical conversions.

[FIX]: Correctly fetches 'admissionweight' from the raw eICU patient table 
       and joins it to the cohort before extraction.
"""

import time
from pathlib import Path
import polars as pl

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_EICU_DIR = BASE_DIR / "data" / "raw" / "eicu-crd" / "2.0"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

def extract_pressors():
    print("[*] Initiating Strict Regex Extraction of eICU Vasopressors...")
    start_time = time.time()
    
    infusion_file = RAW_EICU_DIR / "infusionDrug.csv.gz"
    patient_file = RAW_EICU_DIR / "patient.csv.gz"
    cohort_file = PROCESSED_DIR / "eicu_final_sepsis3_cohort.parquet"
    out_file = PROCESSED_DIR / "eicu_extracted_pressors_raw.parquet"
    
    if not infusion_file.exists() or not patient_file.exists():
        print(f"[ERROR] Required raw eICU files not found in {RAW_EICU_DIR}")
        return

    print("    -> Loading Sepsis-3 cohort and fetching patient weights...")
    df_cohort_base = pl.read_parquet(cohort_file).select(["stay_id", "sepsis_onset_offset"])
    
    df_patient = pl.scan_csv(
        patient_file, 
        infer_schema_length=10000,
        null_values=["", "NA", "null"]
    ).select([
        pl.col("patientunitstayid").alias("stay_id"),
        pl.col("admissionweight").alias("patientweight")
    ]).collect()
    
    # Merge weight into the cohort dataframe
    df_cohort = df_cohort_base.join(df_patient, on="stay_id", how="left")
    
    print("    -> Scanning infusion data...")
    df_infusion = pl.scan_csv(
        infusion_file, 
        infer_schema_length=10000, 
        null_values=["", "NA", "null", "None"]
    ).select([
        "patientunitstayid", "infusionoffset", "drugname", "drugrate", "infusionrate"
    ]).rename({"patientunitstayid": "stay_id", "infusionoffset": "event_time"})
    
    # Filter to cohort and acute window (Onset to +24h)
    df_joined = df_infusion.join(df_cohort.lazy(), on="stay_id", how="inner")
    
    df_window = df_joined.with_columns(
        ((pl.col("event_time") - pl.col("sepsis_onset_offset")) / 60.0).alias("hours_from_onset")
    ).filter(
        (pl.col("hours_from_onset") >= 0) & (pl.col("hours_from_onset") <= 24)
    )

    print("    -> Applying strict regex word boundaries for drug identification...")
    df_drugs = df_window.with_columns(pl.col("drugname").str.to_lowercase().alias("drugname_lower"))
    
    # Strict Regex for drug categorization
    regex_ne = r"\b(norepinephrine|levophed|norepi)\b"
    regex_epi = r"\b(epinephrine|adrenaline|epi)\b"
    regex_vaso = r"\b(vasopressin|pitressin)\b"
    regex_dopa = r"\b(dopamine)\b"
    regex_phenyl = r"\b(phenylephrine|neo-synephrine|neosynephrine)\b"
    
    df_mapped = df_drugs.with_columns(
        pl.when(pl.col("drugname_lower").str.contains(regex_ne)).then(pl.lit("norepinephrine"))
        .when(pl.col("drugname_lower").str.contains(regex_epi)).then(pl.lit("epinephrine"))
        .when(pl.col("drugname_lower").str.contains(regex_vaso)).then(pl.lit("vasopressin"))
        .when(pl.col("drugname_lower").str.contains(regex_dopa)).then(pl.lit("dopamine"))
        .when(pl.col("drugname_lower").str.contains(regex_phenyl)).then(pl.lit("phenylephrine"))
        .otherwise(pl.lit("other"))
        .alias("drug_type")
    ).filter(pl.col("drug_type") != "other")

    print("    -> Extracting explicit units embedded in parentheses...")
    # Extracts the LAST occurrence of text within parentheses
    df_units = df_mapped.with_columns(
        pl.col("drugname_lower").str.extract(r".*\(([^)]+)\).*", 1).alias("embedded_unit")
    )

    print("    -> Parsing numeric rates (Prioritizing drugrate)...")
    # Clean the rate columns to contain only numbers/decimals
    df_parsed = df_units.with_columns(
        pl.col("drugrate").cast(pl.Utf8).str.replace_all(r"[^0-9\.]", "").cast(pl.Float64, strict=False).alias("num_drugrate"),
        pl.col("infusionrate").cast(pl.Utf8).str.replace_all(r"[^0-9\.]", "").cast(pl.Float64, strict=False).alias("num_infrate")
    ).with_columns(
        pl.when(pl.col("num_drugrate").is_not_null() & (pl.col("num_drugrate") > 0))
        .then(pl.col("num_drugrate"))
        .otherwise(pl.col("num_infrate"))
        .alias("raw_rate")
    ).filter(pl.col("raw_rate").is_not_null() & (pl.col("raw_rate") > 0))

    # Final cleanup of columns to pass to the standardizer
    df_final = df_parsed.select([
        "stay_id", "event_time", "patientweight", 
        "drugname_lower", "drug_type", "raw_rate", "embedded_unit"
    ]).collect()

    # Quick profile of the extracted units
    print("\n    [UNIT EXTRACTION PROFILE]")
    unit_counts = df_final.group_by("embedded_unit").len().sort("len", descending=True).head(10)
    for row in unit_counts.iter_rows():
        print(f"       - {row[0]}: {row[1]:,}")

    print("\n    -> Exporting raw extracted pressors...")
    df_final.write_parquet(out_file)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Extracted {df_final.height:,} clean vasopressor records in {elapsed:.2f} seconds.")
    print(f"    -> Saved to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    extract_pressors()