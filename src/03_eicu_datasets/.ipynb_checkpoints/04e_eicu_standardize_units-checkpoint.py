"""
04e_eicu_standardize_units.py

Takes the raw extracted pressor rates and their embedded unit strings, and explicitly 
converts them into standardized mass-based dosing (mcg/kg/min) or units/min (Vasopressin).

Features included:
- Tracks weight source (measured vs. imputed 80kg) to maintain transparency.
- Logs the exact conversion method pathway applied to every row.
- Relies on a JSON configuration file (audited to outputs/metrics/) for standard ICU 
  concentration assumptions to salvage volumetric (ml/hr) rates.
- Isolates and exports unprocessable rows to an audit log for debugging and manual review.
- Validates the resulting distributions (Median, IQR, 99th, Max) prior to NEQ calculation.
"""

import time
import json
from pathlib import Path
import polars as pl

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

# Flattened Output for Audits & Configs
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_METRICS.mkdir(parents=True, exist_ok=True)

# Re-routed the config file to the metrics audit folder
CONFIG_FILE = OUT_METRICS / "eicu_standard_pressor_concentrations.json"

# Default fallback config if it doesn't exist
DEFAULT_CONCENTRATIONS = {
    "norepinephrine": 16.0,  # 4 mg / 250 mL -> 16 mcg/mL
    "epinephrine": 16.0,     # 4 mg / 250 mL -> 16 mcg/mL
    "phenylephrine": 80.0,   # 20 mg / 250 mL -> 80 mcg/mL
    "dopamine": 1600.0,      # 400 mg / 250 mL -> 1600 mcg/mL
    "vasopressin": 0.2       # 20 units / 100 mL -> 0.2 units/mL
}

# Physiological upper bounds (mcg/kg/min; Vaso in units/min)
UPPER_BOUNDS = {
    "norepinephrine": 5.0,
    "epinephrine": 5.0,
    "phenylephrine": 20.0,
    "dopamine": 50.0,
    "vasopressin": 0.2
}

def setup_config():
    """Ensures the configuration file exists and loads it."""
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONCENTRATIONS, f, indent=4)
        return DEFAULT_CONCENTRATIONS
    
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def standardize_units():
    print("[*] Initiating Audited Unit Standardization for eICU Pressors...")
    start_time = time.time()
    
    concentrations = setup_config()
    print(f"    -> Loaded standard ICU concentrations from {CONFIG_FILE.name}")
    
    in_file = PROCESSED_DIR / "eicu_extracted_pressors_raw.parquet"
    out_file = PROCESSED_DIR / "eicu_standardized_pressors.parquet"
    unconvertible_file = OUT_METRICS / "eicu_unprocessable_pressors.csv"
    
    if not in_file.exists():
        print(f"[ERROR] Raw extracted pressors not found at: {in_file}")
        return

    print("    -> Loading raw extracted pressors...")
    df_raw = pl.read_parquet(in_file)
    
    # 1. Explicit Weight Tracking
    df_weight = df_raw.with_columns(
        pl.col("patientweight").is_null().alias("is_weight_imputed"),
        pl.col("patientweight").fill_null(80.0).alias("weight_kg")
    ).with_columns(
        pl.when(pl.col("is_weight_imputed")).then(pl.lit("imputed_80kg"))
        .otherwise(pl.lit("measured"))
        .alias("weight_source")
    )
    
    measured_count = df_weight.filter(pl.col("weight_source") == "measured").height
    imputed_count = df_weight.filter(pl.col("weight_source") == "imputed_80kg").height
    print(f"\n    [WEIGHT TRACKING]")
    print(f"       - Measured weights used: {measured_count:,}")
    print(f"       - Imputed weights (80kg fallback): {imputed_count:,}")

    # 2. Mathematical Conversion & Pathway Logging
    print("\n    -> Applying explicit mathematical conversions...")
    
    df_converted = df_weight.with_columns(
        # RATE CONVERSION
        pl.when(pl.col("drug_type") == "vasopressin").then(
            pl.when(pl.col("embedded_unit") == "units/min").then(pl.col("raw_rate"))
            .when(pl.col("embedded_unit") == "units/hr").then(pl.col("raw_rate") / 60.0)
            .when(pl.col("embedded_unit") == "ml/hr").then((pl.col("raw_rate") * concentrations["vasopressin"]) / 60.0)
            .otherwise(None)
        ).otherwise(
            pl.when(pl.col("embedded_unit") == "mcg/kg/min").then(pl.col("raw_rate"))
            .when(pl.col("embedded_unit") == "mcg/min").then(pl.col("raw_rate") / pl.col("weight_kg"))
            .when(pl.col("embedded_unit") == "mg/kg/min").then(pl.col("raw_rate") * 1000.0)
            .when(pl.col("embedded_unit") == "mg/min").then((pl.col("raw_rate") * 1000.0) / pl.col("weight_kg"))
            .when(pl.col("embedded_unit") == "mcg/hr").then((pl.col("raw_rate") / 60.0) / pl.col("weight_kg"))
            .when(pl.col("embedded_unit") == "ml/hr").then(
                pl.when(pl.col("drug_type") == "norepinephrine").then((pl.col("raw_rate") * concentrations["norepinephrine"]) / 60.0 / pl.col("weight_kg"))
                .when(pl.col("drug_type") == "epinephrine").then((pl.col("raw_rate") * concentrations["epinephrine"]) / 60.0 / pl.col("weight_kg"))
                .when(pl.col("drug_type") == "phenylephrine").then((pl.col("raw_rate") * concentrations["phenylephrine"]) / 60.0 / pl.col("weight_kg"))
                .when(pl.col("drug_type") == "dopamine").then((pl.col("raw_rate") * concentrations["dopamine"]) / 60.0 / pl.col("weight_kg"))
                .otherwise(None)
            ).otherwise(None)
        ).alias("standardized_rate"),

        # CONVERSION PATHWAY LOGGING
        pl.when(pl.col("drug_type") == "vasopressin").then(
            pl.when(pl.col("embedded_unit") == "units/min").then(pl.lit("direct"))
            .when(pl.col("embedded_unit") == "units/hr").then(pl.lit("time_normalized"))
            .when(pl.col("embedded_unit") == "ml/hr").then(pl.lit("concentration_assumed"))
            .otherwise(None)
        ).otherwise(
            pl.when(pl.col("embedded_unit") == "mcg/kg/min").then(pl.lit("direct"))
            .when(pl.col("embedded_unit") == "mcg/min").then(pl.lit("weight_normalized"))
            .when(pl.col("embedded_unit") == "mg/kg/min").then(pl.lit("mass_converted"))
            .when(pl.col("embedded_unit") == "mg/min").then(pl.lit("mass_and_weight_converted"))
            .when(pl.col("embedded_unit") == "mcg/hr").then(pl.lit("time_and_weight_normalized"))
            .when(pl.col("embedded_unit") == "ml/hr").then(pl.lit("concentration_assumed"))
            .otherwise(None)
        ).alias("conversion_method")
    )

    # 3. Export Unprocessable Records
    df_unprocessable = df_converted.filter(pl.col("standardized_rate").is_null())
    df_unprocessable.select(["stay_id", "drugname_lower", "drug_type", "raw_rate", "embedded_unit"]).write_csv(unconvertible_file)
    
    print(f"\n    [CONVERSION PATHWAYS SUMMARY]")
    pathways = df_converted.filter(pl.col("standardized_rate").is_not_null()).group_by("conversion_method").len().sort("len", descending=True)
    for row in pathways.iter_rows():
        print(f"       - {row[0]:<30}: {row[1]:,}")
    print(f"       - EXPORTED UNPROCESSABLE         : {df_unprocessable.height:,} (Saved to {unconvertible_file.name})")

    # 4. Physiological Bounds Tracking
    print("\n    [PHYSIOLOGICAL BOUNDS CLIPPING]")
    df_clean = df_converted.filter(pl.col("standardized_rate").is_not_null())
    
    valid_dfs = []
    for drug, max_val in UPPER_BOUNDS.items():
        df_drug = df_clean.filter(pl.col("drug_type") == drug)
        if df_drug.height == 0:
            continue
        
        df_valid = df_drug.filter(pl.col("standardized_rate") <= max_val)
        removed = df_drug.height - df_valid.height
        print(f"       - {drug.capitalize():<15} | Upper Bound: {max_val:<5} | Records Removed: {removed:,}")
        valid_dfs.append(df_valid)
        
    df_final = pl.concat(valid_dfs).select([
        "stay_id", "event_time", "weight_source", "conversion_method", 
        "drug_type", "standardized_rate"
    ])

    # 5. Validation / Distribution Verification
    print("\n    [DISTRIBUTION VERIFICATION (Post-Standardization)]")
    dist = df_final.group_by("drug_type").agg([
        pl.col("standardized_rate").median().round(3).alias("Median"),
        pl.col("standardized_rate").quantile(0.25).round(3).alias("Q25"),
        pl.col("standardized_rate").quantile(0.75).round(3).alias("Q75"),
        pl.col("standardized_rate").quantile(0.99).round(3).alias("99th_Pct"),
        pl.col("standardized_rate").max().round(3).alias("Max")
    ])
    
    # Simple table print formatting
    header = f"       | {'Drug Type':<15} | {'Median':<8} | {'IQR (Q25-Q75)':<15} | {'99th Pct':<8} | {'Max':<8} |"
    print("       " + "-" * (len(header) - 7))
    print(header)
    print("       " + "-" * (len(header) - 7))
    for row in dist.iter_rows():
        iqr = f"{row[2]}-{row[3]}"
        print(f"       | {row[0].capitalize():<15} | {row[1]:<8} | {iqr:<15} | {row[4]:<8} | {row[5]:<8} |")
    print("       " + "-" * (len(header) - 7))

    print("\n    -> Exporting strictly standardized pressor rates...")
    df_final.write_parquet(out_file)
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! Standardized {df_final.height:,} records in {elapsed:.2f} seconds.")
    print(f"    -> Saved to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    standardize_units()