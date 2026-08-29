"""
Void physiologically impossible values in the extracted time series.

Applies a per-item plausible range and sets anything outside it to NULL rather
than dropping the row or clipping to the boundary. Clipping would invent a value
at the limit; NULL lets the downstream SAITS model reconstruct it from the
patient's own trajectory, which is the intended treatment for charting artefacts
such as a mean arterial pressure of 80,000.

Runs on the Polars lazy engine and reports how many values were voided.

Reads:
    mimic_sepsis_temporal_data.parquet
Writes:
    mimic_sepsis_temporal_data_cleaned.parquet
"""

import time
from pathlib import Path
import polars as pl

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

# --- Main Execution ------------------------------------------------------
def main():
    print("Executing MIMIC-IV physiological bounding pipeline...")
    start_time = time.time()
    
    # Read explicitly named file from Script 04a
    in_file = PROCESSED_DIR / "mimic_sepsis_temporal_data.parquet"
    out_file = PROCESSED_DIR / "mimic_sepsis_temporal_data_cleaned.parquet"
    
    if not in_file.exists():
        print(f"[ERROR] Raw temporal data not found at: {in_file}")
        return

    print("\n[*] Loading temporal dataset into Polars lazy engine...")
    df_raw = pl.scan_parquet(in_file)
    
    # Define physiological bounds: {itemid: (min_val, max_val)}
    bounds = {
        # Vitals
        220045: (20, 300),      # HR
        220181: (20, 300),      # MAP (Non-invasive)
        220052: (20, 300),      # MAP (Invasive)
        220210: (0, 100),       # RR
        223761: (25.0, 45.0),   # Temp F (Converted to C in 04a, but bounded just in case)
        223762: (25.0, 45.0),   # Temp C
        220277: (0, 100),       # SpO2
        223835: (0.2, 100),     # FiO2
        
        # GCS
        220739: (1, 4),         # GCS Eye
        223900: (1, 5),         # GCS Verbal
        223901: (1, 6),         # GCS Motor
        
        # Labs
        50821: (20, 800),       # PaO2
        50818: (10, 200),       # PaCO2
        50820: (6.5, 8.0),      # pH
        50813: (0, 50),         # Lactate
        50912: (0, 40),         # Creatinine
        51006: (0, 300),        # BUN
        50885: (0, 80),         # tBil
        51301: (0, 300),        # WBC
        51300: (0, 300),        # WBC
        51265: (0, 2500),       # Platelets
        51222: (0, 30),         # Hemoglobin
        51274: (0, 200),        # PT
        51275: (0, 300),        # APTT
        50862: (0, 15),         # Albumin
        50971: (0, 15),         # Potassium
        50822: (0, 15),         # Potassium (Blood Gas)
        50983: (90, 200),       # Sodium
        50824: (90, 200),       # Sodium (Blood Gas)
        50931: (10, 2000),      # Glucose
        50809: (10, 2000),      # Glucose (Blood Gas)
        50902: (50, 160),       # Chloride
        50806: (50, 160),       # Chloride (Blood Gas)
        
        # Raw Vasopressors
        221906: (0, 1000),      # Norepinephrine
        221289: (0, 1000),      # Epinephrine
        221662: (0, 1000),      # Dopamine
        221653: (0, 1000),      # Dobutamine
        222315: (0, 100),       # Vasopressin
        221749: (0, 1000),      # Phenylephrine
        
        # Urine Output (Removing egregious keyboard smash errors, e.g., > 5000 mL at once)
        226559: (0, 5000), 226560: (0, 5000), 226561: (0, 5000), 
        226584: (0, 5000), 226563: (0, 5000), 226564: (0, 5000), 
        226565: (0, 5000), 226567: (0, 5000), 226557: (0, 5000), 
        226558: (0, 5000),
        
        # Unified Synthetic Interventions (Generated in 04a)
        888888: (0, 2)          # Unified Mechanical Ventilation (Binary)
    }

    # Convert bounds dictionary into a Polars DataFrame for efficient joining
    itemids = list(bounds.keys())
    mins = [val[0] for val in bounds.values()]
    maxs = [val[1] for val in bounds.values()]
    
    df_bounds = pl.DataFrame({
        "itemid": itemids,
        "bound_min": mins,
        "bound_max": maxs
    }, schema={"itemid": pl.Int64, "bound_min": pl.Float64, "bound_max": pl.Float64}).lazy()

    print("[*] Applying physiological limits across Vitals, Labs, GCS, UO, and Interventions...")
    
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
    # Track how many artifacts were dropped (requires a collect step)
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
    print(f"    -> Clinical artifacts converted to NULL for SAITS imputation: {outliers_removed}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
