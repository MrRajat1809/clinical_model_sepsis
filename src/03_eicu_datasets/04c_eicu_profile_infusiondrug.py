"""
Profile the infusionDrug table before writing any extraction rules.

Diagnostic, not part of the data path. eICU records infusion drug names and
rates as free text with no controlled vocabulary, so the extraction and unit
conversion in 04d and 04e have to be written against what is actually present
rather than what the schema suggests.

Reports, restricted to the sepsis cohort so the counts reflect the population
that matters:
    missingness in drug name, drug rate and infusion rate
    the 200 most frequent drug names
    the 100 most frequent rate strings, which is where the embedded units show

Run this first when the eICU version changes or the pressor extraction looks
wrong.

Reads:
    eicu_sepsis_phenotype_cohort.parquet
    data/raw/eicu-crd/2.0/infusionDrug
Writes:
    outputs/metrics/eicu_infusiondrug_profile.json and a readable text report
"""

import time
import json
from pathlib import Path
import polars as pl

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_EICU_DIR = BASE_DIR / "data" / "raw" / "eicu-crd" / "2.0"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

OUT_METRICS = BASE_DIR / "outputs" / "metrics"

def profile_infusion_drugs():
    print("[*] Initiating Data-Driven Profiling of eICU infusionDrug.csv.gz...")
    start_time = time.time()
    
    OUT_METRICS.mkdir(parents=True, exist_ok=True)
    
    infusion_file = RAW_EICU_DIR / "infusionDrug.csv.gz"
    # Point to the Phenotype cohort to allow execution before script 05
    cohort_file = PROCESSED_DIR / "eicu_sepsis_phenotype_cohort.parquet"
    out_json = OUT_METRICS / "eicu_infusiondrug_profile.json"
    out_txt = OUT_METRICS / "eicu_infusiondrug_profile_report.txt"
    
    if not infusion_file.exists():
        print(f"[ERROR] eICU infusionDrug not found at: {infusion_file}")
        return

    print("    -> Loading Sepsis Phenotype cohort stay_ids...")
    try:
        df_cohort = pl.read_parquet(cohort_file).select(["stay_id"])
    except Exception as e:
        print(f"[ERROR] Failed to load cohort file at {cohort_file}. Ensure it has been generated. Error: {e}")
        return
    
    print("    -> Scanning and filtering infusion data...")
    # Polars can lazily scan .gz files directly
    df_infusion = pl.scan_csv(
        infusion_file, 
        infer_schema_length=10000, 
        null_values=["", "NA", "null", "None"]
    ).select([
        "patientunitstayid", "drugname", "drugrate", "infusionrate"
    ]).rename({"patientunitstayid": "stay_id"})
    
    # Filter only to our cohort to reduce noise from irrelevant patients
    df_joined = df_infusion.join(df_cohort.lazy(), on="stay_id", how="inner").collect()
    
    total_records = df_joined.height
    print(f"    -> Extracted {total_records:,} infusion records for the cohort.")

    # --- Profiling Metrics -----------------------------------------------
    print("    -> Calculating missingness and generating frequency distributions...")
    
    # 1. Missingness
    missing_stats = {
        "missing_drugname": df_joined.select(pl.col("drugname").is_null().sum()).item(),
        "missing_drugrate": df_joined.select(pl.col("drugrate").is_null().sum()).item(),
        "missing_infusionrate": df_joined.select(pl.col("infusionrate").is_null().sum()).item(),
    }
    
    # 2. Top 200 Drug Names (Lowercased for grouping)
    top_drugs = df_joined.with_columns(
        pl.col("drugname").str.to_lowercase().alias("drugname_lower")
    ).group_by("drugname_lower").count().sort("count", descending=True).head(200)
    
    # 3. Top 100 Drugrate Strings (To inspect embedded units like "mcg/kg/min")
    top_drugrates = df_joined.group_by("drugrate").count().sort("count", descending=True).head(100)
    
    # 4. Top 100 Infusionrate Strings
    top_infusionrates = df_joined.group_by("infusionrate").count().sort("count", descending=True).head(100)
    
    # Compile JSON report
    profile_data = {
        "total_cohort_infusion_records": total_records,
        "missingness": missing_stats,
        "top_200_drugnames": dict(zip(top_drugs["drugname_lower"].to_list(), top_drugs["count"].to_list())),
        "top_100_drugrates": dict(zip(top_drugrates["drugrate"].to_list(), top_drugrates["count"].to_list())),
        "top_100_infusionrates": dict(zip(top_infusionrates["infusionrate"].to_list(), top_infusionrates["count"].to_list()))
    }
    
    with open(out_json, "w") as f:
        json.dump(profile_data, f, indent=4)
        
    # Compile Readable Text Report
    with open(out_txt, "w") as f:
        f.write("=========================================\n")
        f.write(" eICU INFUSIONDRUG PROFILING REPORT\n")
        f.write("=========================================\n\n")
        f.write(f"Total Cohort Records: {total_records:,}\n\n")
        
        f.write("--- MISSINGNESS ---\n")
        for k, v in missing_stats.items():
            f.write(f"{k}: {v:,} ({(v/total_records)*100:.2f}%)\n")
            
        f.write("\n--- TOP 30 DRUG NAMES ---\n")
        for row in top_drugs.head(30).iter_rows():
            f.write(f"{row[0]}: {row[1]:,}\n")
            
        f.write("\n--- TOP 30 DRUG RATES (UNITS VISIBLE) ---\n")
        for row in top_drugrates.head(30).iter_rows():
            f.write(f"{row[0]}: {row[1]:,}\n")
            
        f.write("\n--- TOP 30 INFUSION RATES ---\n")
        for row in top_infusionrates.head(30).iter_rows():
            f.write(f"{row[0]}: {row[1]:,}\n")

    elapsed = time.time() - start_time
    print(f"\n[+] Success! Profiling complete in {elapsed:.2f} seconds.")
    print(f"    -> Full JSON: {out_json.relative_to(BASE_DIR)}")
    print(f"    -> Summary Report: {out_txt.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    profile_infusion_drugs()
