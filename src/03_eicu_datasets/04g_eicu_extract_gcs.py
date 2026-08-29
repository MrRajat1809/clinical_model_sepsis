"""
Extract Glasgow Coma Scale components from nurse charting.

GCS does not appear in the eICU tables the main temporal slice reads, so it is
recovered separately from nurseCharting, which is the largest table in the
database. The lazy scan runs on the Polars streaming engine to keep memory
bounded.

Values are stored as clean numeric strings, so they are cast directly after
stripping non-numeric characters, then constrained to the valid range of each
component: eye 1-4, verbal 1-5, motor 1-6. Out-of-range values are dropped
rather than repaired, since a GCS component outside its range is a charting
error with no recoverable intent.

Reads:
    eicu_sepsis_phenotype_cohort.parquet
    data/raw/eicu-crd/2.0/nurseCharting
Writes:
    eicu_gcs_timeline.parquet
"""

import time
import sys
import threading
import itertools
from pathlib import Path
import polars as pl

# --- Configuration -------------------------------------------------------
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

def extract_gcs():
    print("[*] Initiating eICU GCS Direct Numeric Extraction...")
    start_time = time.time()
    
    charting_file = RAW_EICU_DIR / "nurseCharting.csv.gz"
    # Point to the Phenotype cohort to allow execution before script 05
    cohort_file = PROCESSED_DIR / "eicu_sepsis_phenotype_cohort.parquet"
    out_file = PROCESSED_DIR / "eicu_gcs_timeline.parquet"
    
    if not charting_file.exists():
        print(f"[ERROR] Required raw eICU file not found: {charting_file}")
        return

    print("    -> Loading Sepsis Phenotype cohort...")
    try:
        # Load sit_offset instead of sepsis_onset_offset
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
        pl.col("nursingchartoffset").alias("event_time"),
        pl.col("nursingchartcelltypevalname").cast(pl.Utf8).str.to_lowercase().alias("chart_name"),
        pl.col("nursingchartvalue").cast(pl.Utf8).str.replace_all(r"[^0-9\.]", "").alias("chart_value")
    ])
    
    # Filter for GCS names and drop empty values
    df_gcs = df_chart.filter(
        pl.col("chart_name").str.contains(r"motor|verbal|eye") & 
        (pl.col("chart_value") != "")
    )
    
    df_joined = df_gcs.join(df_cohort.lazy(), on="stay_id", how="inner")
    
    # Calculate temporal window based on SIT and expand to +/- 48 hours
    df_window = df_joined.with_columns(
        ((pl.col("event_time") - pl.col("sit_offset")) / 60.0).alias("hours_from_sit"),
        pl.col("chart_value").cast(pl.Float64, strict=False).alias("valuenum")
    ).filter(
        (pl.col("hours_from_sit") >= -48) & (pl.col("hours_from_sit") <= 48) &
        pl.col("valuenum").is_not_null()
    )

    # Map to canonical names and apply strict physiological bounds
    lazy_query = df_window.with_columns(
        pl.when(pl.col("chart_name").str.contains("motor")).then(pl.lit("gcs_motor"))
        .when(pl.col("chart_name").str.contains("verbal")).then(pl.lit("gcs_verbal"))
        .when(pl.col("chart_name").str.contains("eye")).then(pl.lit("gcs_eye"))
        .otherwise(pl.lit("unknown"))
        .alias("itemid")
    ).filter(
        ((pl.col("itemid") == "gcs_motor") & (pl.col("valuenum").is_between(1.0, 6.0))) |
        ((pl.col("itemid") == "gcs_verbal") & (pl.col("valuenum").is_between(1.0, 5.0))) |
        ((pl.col("itemid") == "gcs_eye") & (pl.col("valuenum").is_between(1.0, 4.0)))
    ).select([
        "stay_id", "event_time", "itemid", "valuenum"
    ])

    spinner = ProgressSpinner("Scanning 1.6GB nurseCharting & extracting scores...")
    spinner.start()
    
    try:
        df_clean = lazy_query.collect(engine="streaming") 
    finally:
        spinner.stop()

    print("\n    [GCS EXTRACTION SUMMARY]")
    dist = df_clean.group_by("itemid").agg([
        pl.len().alias("count"),
        pl.col("valuenum").median().alias("median_score"),
        pl.col("valuenum").min().alias("min"),
        pl.col("valuenum").max().alias("max")
    ])
    
    for row in dist.iter_rows():
        print(f"       - {row[0]:<12} | N: {row[1]:<10,} | Median: {row[2]} | Range: {row[3]}-{row[4]}")

    print("\n    -> Exporting standardized GCS timeline...")
    df_clean.write_parquet(out_file)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Extracted {df_clean.height:,} valid GCS events in {elapsed:.2f} seconds.")
    print(f"    -> Saved to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    extract_gcs()
