"""
04h_eicu_extract_fio2.py

Mines the unstructured `respiratoryCharting` table to extract FiO2 percentages.
Converts decimal representations (e.g., 0.21) to standard percentages (21-100),
applies strict physiological bounds, and exports a clean timeline for the tensor builder.

Features included:
- Swapped input dependency to `eicu_sepsis_phenotype_cohort.parquet` to break the 
  circular dependency.
- Anchors the extraction window to `sit_offset` and widens it to +/- 48 hours.
- Enabled Polars streaming engine to prevent OOM crashes during the processing 
  of the massive respiratoryCharting table.
"""

import time
import sys
import threading
import itertools
from pathlib import Path
import polars as pl

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_EICU_DIR = BASE_DIR / "data" / "raw" / "eicu-crd" / "2.0"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

class ProgressSpinner:
    def __init__(self, message="Processing"):
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.stop_running = False
        self.message = message
        self.thread = threading.Thread(target=self.spin)

    def spin(self):
        start_time = time.time()
        while not self.stop_running:
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r    {next(self.spinner)} {self.message} (Elapsed: {elapsed:.0f}s)...")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 40) + '\r')
        sys.stdout.flush()

    def start(self):
        self.stop_running = False
        self.thread.start()

    def stop(self):
        self.stop_running = True
        self.thread.join()

def extract_fio2():
    print("[*] Initiating eICU FiO2 Extraction and Standardization...")
    start_time = time.time()
    
    charting_file = RAW_EICU_DIR / "respiratoryCharting.csv.gz"
    # [FIX]: Point to the Phenotype cohort to allow execution before script 05
    cohort_file = PROCESSED_DIR / "eicu_sepsis_phenotype_cohort.parquet"
    out_file = PROCESSED_DIR / "eicu_fio2_timeline.parquet"
    
    if not charting_file.exists():
        print(f"[ERROR] Required raw eICU file not found: {charting_file}")
        return

    print("    -> Loading Sepsis Phenotype cohort...")
    try:
        # [FIX]: Load sit_offset instead of sepsis_onset_offset
        df_cohort = pl.read_parquet(cohort_file).select(["stay_id", "sit_offset"])
    except Exception as e:
        print(f"[ERROR] Failed to load cohort file at {cohort_file}. Error: {e}")
        return
    
    # Lazy scan
    df_chart = pl.scan_csv(
        charting_file, 
        infer_schema_length=10000, 
        null_values=["", "NA", "null", "None"]
    ).select([
        pl.col("patientunitstayid").alias("stay_id"),
        pl.col("respchartoffset").alias("event_time"),
        pl.col("respchartvaluelabel").cast(pl.Utf8).str.to_lowercase().alias("chart_name"),
        # Strip out '%' or text, leaving only numbers and decimals
        pl.col("respchartvalue").cast(pl.Utf8).str.replace_all(r"[^0-9\.]", "").alias("chart_value")
    ])
    
    # Filter for FiO2 labels and ensure value isn't empty
    df_fio2 = df_chart.filter(
        pl.col("chart_name").str.contains(r"fio2|fi02|o2 %") & 
        (pl.col("chart_value") != "")
    )
    
    df_joined = df_fio2.join(df_cohort.lazy(), on="stay_id", how="inner")
    
    # [FIX]: Calculate temporal window based on SIT and expand to +/- 48 hours
    df_window = df_joined.with_columns(
        ((pl.col("event_time") - pl.col("sit_offset")) / 60.0).alias("hours_from_sit"),
        pl.col("chart_value").cast(pl.Float64, strict=False).alias("raw_valuenum")
    ).filter(
        (pl.col("hours_from_sit") >= -48) & (pl.col("hours_from_sit") <= 48) &
        pl.col("raw_valuenum").is_not_null()
    )

    # Standardize Units: Multiply decimals (0.20 to 1.0) by 100, then bound between 21 and 100
    lazy_query = df_window.with_columns(
        pl.when(pl.col("raw_valuenum").is_between(0.20, 1.0))
          .then(pl.col("raw_valuenum") * 100.0)
          .otherwise(pl.col("raw_valuenum"))
          .alias("valuenum")
    ).filter(
        pl.col("valuenum").is_between(21.0, 100.0)
    ).with_columns(
        pl.lit("fio2").alias("itemid")
    ).select([
        "stay_id", "event_time", "itemid", "valuenum"
    ])

    spinner = ProgressSpinner("Scanning respiratoryCharting & standardizing FiO2...")
    spinner.start()
    
    try:
        df_clean = lazy_query.collect(engine="streaming") 
    finally:
        spinner.stop()

    print("\n    [FIO2 EXTRACTION SUMMARY]")
    dist = df_clean.group_by("itemid").agg([
        pl.len().alias("count"),
        pl.col("valuenum").median().alias("median_score"),
        pl.col("valuenum").min().alias("min"),
        pl.col("valuenum").max().alias("max")
    ])
    
    for row in dist.iter_rows():
        print(f"       - {row[0]:<12} | N: {row[1]:<10,} | Median: {row[2]:.1f} | Range: {row[3]:.1f}-{row[4]:.1f}")

    print("\n    -> Exporting standardized FiO2 timeline...")
    df_clean.write_parquet(out_file)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Extracted {df_clean.height:,} valid FiO2 events in {elapsed:.2f} seconds.")
    print(f"    -> Saved to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    extract_fio2()