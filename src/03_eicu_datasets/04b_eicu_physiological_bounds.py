"""
Void physiologically impossible values in the eICU time series.

Same intent as the MIMIC-IV bounding step: out-of-range values become NULL for
the imputation model to reconstruct, rather than being clipped to a boundary or
dropped. The bounds table is keyed by eICU's string identifiers rather than
numeric item identifiers, but the limits themselves match the MIMIC-IV ones so
that neither cohort is filtered more aggressively than the other.

Reads:
    eicu_sepsis_temporal_data.parquet
Writes:
    eicu_sepsis_temporal_data_cleaned.parquet
"""

import time
from pathlib import Path

import polars as pl

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

# --- Main Execution ------------------------------------------------------
def main():
    print("[*] Executing eICU physiological bounding pipeline...")
    start_time = time.time()
    
    in_file = PROCESSED_DIR / "eicu_sepsis_temporal_data.parquet"
    out_file = PROCESSED_DIR / "eicu_sepsis_temporal_data_cleaned.parquet"
    
    if not in_file.exists():
        print(f"[ERROR] Raw temporal data not found at: {in_file}")
        return

    print("\n[*] Loading temporal dataset into Polars lazy engine...")
    df_raw = pl.scan_parquet(in_file)
    
    # Define physiological bounds: {itemid_string: (min_val, max_val)}
    bounds = {
        # Vitals (Periodic & Aperiodic)
        'heartrate': (20, 300),
        'systemicmean': (20, 300),
        'noninvasivemean': (20, 300),
        'systemicsystolic': (20, 300),
        'noninvasivesystolic': (20, 300),
        'systemicdiastolic': (20, 300),
        'noninvasivediastolic': (20, 300),
        'respiration': (0, 100),
        'temperature': (25.0, 45.0),
        'sao2': (0, 100),
        
        # Labs
        'pao2': (20, 800),
        'paco2': (10, 200),
        'ph': (6.5, 8.0),
        'lactate': (0, 50),
        'creatinine': (0, 40),
        'bun': (0, 300),
        'total bilirubin': (0, 80),
        'wbc x 1000': (0, 300),
        'platelets x 1000': (0, 2500),
        'hgb': (0, 30),
        'pt': (0, 200),
        'ptt': (0, 300),
        'albumin': (0, 15),
        'potassium': (0, 15),
        'sodium': (90, 200),
        'glucose': (10, 2000),
        'chloride': (50, 160),
        
        # Interventions
        # Vasopressors are bounded in 04e, after unit conversion, not here.
        'urine_output': (0, 5000),
        '888888': (0, 2)           # Unified Mechanical Ventilation (Binary)
    }

    # Convert bounds dictionary into a Polars DataFrame for efficient joining
    itemids = list(bounds.keys())
    mins = [val[0] for val in bounds.values()]
    maxs = [val[1] for val in bounds.values()]
    
    # Note: eICU itemid is a String (Utf8), not Int64 like in MIMIC
    df_bounds = pl.DataFrame({
        "itemid": itemids,
        "bound_min": mins,
        "bound_max": maxs
    }, schema={"itemid": pl.Utf8, "bound_min": pl.Float64, "bound_max": pl.Float64}).lazy()

    print("[*] Applying physiological limits across Vitals, Labs, UO, and Interventions...")
    
    # Join the bounds, apply the rules, and convert out-of-bounds values to null
    df_cleaned = df_raw.join(
        df_bounds, on="itemid", how="left"
    ).with_columns(
        pl.when(
            (pl.col("valuenum") < pl.col("bound_min")) | 
            (pl.col("valuenum") > pl.col("bound_max"))
        )
        .then(pl.lit(None).cast(pl.Float64))
        .otherwise(pl.col("valuenum"))
        .alias("valuenum_cleaned")
    )

    print("    - Evaluating graph and materializing bounds (collecting)...")
    df_collected = df_cleaned.collect()
    
    nulls_before = df_collected["valuenum"].null_count()
    nulls_after = df_collected["valuenum_cleaned"].null_count()
    outliers_removed = nulls_after - nulls_before

    # Drop the intermediate bounding columns and replace the old valuenum
    df_final = (df_collected
                .drop(["valuenum", "bound_min", "bound_max"])
                .rename({"valuenum_cleaned": "valuenum"}))
    
    df_final.write_parquet(out_file)

    elapsed = time.time() - start_time
    print(f"\n[+] Success! Physiological limits applied in {elapsed:.2f} seconds.")
    print(f"    -> Clinical artifacts converted to NULL for downstream imputation: {outliers_removed:,}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
