"""
Extract vasopressor and inotrope administrations with their recorded units.

First of three steps that turn eICU's free-text infusion records into
comparable doses. This step identifies drugs and parses values; it deliberately
performs no arithmetic, so that extraction and conversion can be audited
separately.

Drugs are matched with word-boundary regular expressions rather than plain
substrings, which prevents norepinephrine from also matching as epinephrine.
Six agents are captured: norepinephrine, epinephrine, vasopressin, dopamine,
dobutamine and phenylephrine. Dobutamine is included because MIMIC-IV extracts
it and both SOFA calculators score it for cardiovascular points.

Units are read from the parenthetical suffix of the drug description, which is
where eICU actually records them. Rates prefer the drugrate column and fall back
to infusionrate. Admission weight is joined from the patient table for the
weight-based conversions in 04e.

Reads:
    eicu_sepsis_phenotype_cohort.parquet
    data/raw/eicu-crd/2.0/{infusionDrug, patient}
Writes:
    eicu_extracted_pressors_raw.parquet
"""

import time
from pathlib import Path
import polars as pl

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_EICU_DIR = BASE_DIR / "data" / "raw" / "eicu-crd" / "2.0"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

def extract_pressors():
    print("[*] Initiating Strict Regex Extraction of eICU Vasopressors...")
    start_time = time.time()
    
    infusion_file = RAW_EICU_DIR / "infusionDrug.csv.gz"
    patient_file = RAW_EICU_DIR / "patient.csv.gz"
    # Point to the Phenotype cohort to allow execution before script 05
    cohort_file = PROCESSED_DIR / "eicu_sepsis_phenotype_cohort.parquet"
    out_file = PROCESSED_DIR / "eicu_extracted_pressors_raw.parquet"
    
    if not infusion_file.exists() or not patient_file.exists():
        print(f"[ERROR] Required raw eICU files not found in {RAW_EICU_DIR}")
        return

    print("    -> Loading Sepsis Phenotype cohort and fetching patient weights...")
    try:
        # Load sit_offset instead of sepsis_onset_offset
        df_cohort_base = pl.read_parquet(cohort_file).select(["stay_id", "sit_offset"])
    except Exception as e:
        print(f"[ERROR] Failed to load cohort file at {cohort_file}. Error: {e}")
        return
    
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
    
    # Filter to cohort
    df_joined = df_infusion.join(df_cohort.lazy(), on="stay_id", how="inner")
    
    # Calculate temporal window based on SIT and expand to +/- 48 hours
    df_window = df_joined.with_columns(
        ((pl.col("event_time") - pl.col("sit_offset")) / 60.0).alias("hours_from_sit")
    ).filter(
        (pl.col("hours_from_sit") >= -48) & (pl.col("hours_from_sit") <= 48)
    )

    print("    -> Applying strict regex word boundaries for drug identification...")
    df_drugs = df_window.with_columns(pl.col("drugname").str.to_lowercase().alias("drugname_lower"))
    
    # Strict Regex for drug categorization
    regex_ne = r"\b(norepinephrine|levophed|norepi)\b"
    regex_epi = r"\b(epinephrine|adrenaline|epi)\b"
    regex_vaso = r"\b(vasopressin|pitressin)\b"
    regex_dopa = r"\b(dopamine)\b"
    # Parity with MIMIC, which extracts dobutamine for SOFA CV.
    regex_dobu = r"\b(dobutamine|dobutrex)\b"
    regex_phenyl = r"\b(phenylephrine|neo-synephrine|neosynephrine)\b"
    
    df_mapped = df_drugs.with_columns(
        pl.when(pl.col("drugname_lower").str.contains(regex_ne)).then(pl.lit("norepinephrine"))
        .when(pl.col("drugname_lower").str.contains(regex_epi)).then(pl.lit("epinephrine"))
        .when(pl.col("drugname_lower").str.contains(regex_vaso)).then(pl.lit("vasopressin"))
        .when(pl.col("drugname_lower").str.contains(regex_dopa)).then(pl.lit("dopamine"))
        .when(pl.col("drugname_lower").str.contains(regex_dobu)).then(pl.lit("dobutamine"))
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
        print(f"        - {row[0]}: {row[1]:,}")

    print("\n    -> Exporting raw extracted pressors...")
    df_final.write_parquet(out_file)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Extracted {df_final.height:,} clean vasopressor records in {elapsed:.2f} seconds.")
    print(f"    -> Saved to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    extract_pressors()
