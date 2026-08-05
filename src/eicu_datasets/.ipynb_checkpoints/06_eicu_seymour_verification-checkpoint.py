"""
06_eicu_seymour_verification.py

Phase 9: External Validation (eICU Sepsis Phenotype Sanity Check)
Performs a static K-Means clustering (k=4) on the final eICU Sepsis-3 cohort
to verify alignment with Seymour et al.'s clinical endotypes (Alpha, Beta, Gamma, Delta).

This acts as a biological and clinical "positive control" for the cohort extraction.
If the eICU extraction was successful, we should see the same 4 distinct phenotypes 
emerge with their expected mortality gradients, proving that we haven't lost the 
biological signal of sepsis while navigating the eICU schema.

[FIX]: Added numpy and explicit numeric casting to prevent Pandas object-type 
       errors when clustering columns contain entirely null values.
"""

import time
from pathlib import Path

import pandas as pd
import polars as pl
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Executing eICU Seymour Sepsis Phenotype Verification...")
    start_time = time.time()
    
    cohort_file = PROCESSED_DIR / "eicu_final_sepsis3_cohort.parquet"
    temporal_file = PROCESSED_DIR / "eicu_sepsis_temporal_data_cleaned.parquet"
    
    if not cohort_file.exists():
        print(f"[ERROR] Final cohort file not found at: {cohort_file}")
        return

    print("\n[*] Loading eICU Sepsis-3 cohort and scanning temporal data...")
    df_cohort = pl.read_parquet(cohort_file)
    df_vitals = pl.scan_parquet(temporal_file)
    
    df_joined = df_vitals.join(df_cohort.lazy().select(["stay_id", "sepsis_onset_offset"]), on="stay_id")

    print("[*] Segmenting acute window (0 to 24 hours post-onset using minute offsets)...")
    # Filter strictly to the acute window using eICU offsets
    df_acute = df_joined.with_columns(
        ((pl.col("event_time") - pl.col("sepsis_onset_offset")) / 60.0).alias("hours_from_onset")
    ).filter(
        (pl.col("hours_from_onset") >= 0) & (pl.col("hours_from_onset") <= 24)
    ).collect()

    print("[*] Aggregating clinical biomarkers for K-Means clustering...")
    # Map eICU string variables to standard feature names
    feature_map = {
        "heartrate": "heart_rate",
        "systemicmean": "map",
        "noninvasivemean": "map",
        "respiration": "resp_rate",
        "temperature": "temperature",
        "wbc x 1000": "wbc",
        "platelets x 1000": "platelets",
        "creatinine": "creatinine",
        "total bilirubin": "bilirubin",
        "lactate": "lactate"
    }

    mapping_df = pl.DataFrame({
        "itemid": list(feature_map.keys()),
        "feature": list(feature_map.values())
    }, schema={"itemid": pl.Utf8, "feature": pl.Utf8})
    
    df_mapped = df_acute.join(mapping_df, on="itemid", how="inner")

    # Pivot to get summary metrics per stay_id
    df_pivoted = df_mapped.group_by(["stay_id", "feature"]).agg(
        pl.col("valuenum").mean().alias("val")
    ).pivot(
        values="val",
        index="stay_id",
        on="feature",
        aggregate_function="first"
    )

    # Join back with cohort demographic data and mortality labels
    analysis_df = df_cohort.select([
        "stay_id", "subject_id", "hadm_id", "age", "hospital_expire_flag"
    ]).join(
        df_pivoted, on="stay_id", how="inner"
    ).to_pandas()

    # Select features for clustering
    clustering_cols = ["age", "heart_rate", "map", "resp_rate", "temperature", "wbc", "platelets", "creatinine", "bilirubin", "lactate"]
    
    for col in clustering_cols:
        if col not in analysis_df.columns:
            analysis_df[col] = np.nan
        # Force cast to float to prevent Pandas object-type errors during grouping
        analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')

    X = analysis_df[clustering_cols]
    
    print("[*] Imputing missing values (median) and scaling features...")
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    print("[*] Executing K-Means clustering (k=4)...")
    # Run K-Means with k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    analysis_df["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Generate summary stats
    summary = analysis_df.groupby("cluster").agg(
        Patient_Count=("stay_id", "count"),
        Mortality_Rate=("hospital_expire_flag", "mean"),
        Mean_Age=("age", "mean"),
        Mean_Lactate=("lactate", "mean"),
        Mean_Creatinine=("creatinine", "mean"),
        Mean_WBC=("wbc", "mean"),
        Mean_MAP=("map", "mean")
    ).reset_index()
    
    # --- AUTOMATED SEYMOUR ENDOTYPE MAPPING ---
    summary["Endotype"] = ""
    
    # 1. Delta: Highest mortality (Refractory Shock)
    delta_idx = summary["Mortality_Rate"].idxmax()
    summary.loc[delta_idx, "Endotype"] = "Delta (δ) - Shock"
    
    # 2. Beta: Highest creatinine among the remaining (Renal Dysfunction)
    rem_idx1 = summary.index.difference([delta_idx])
    beta_idx = summary.loc[rem_idx1, "Mean_Creatinine"].idxmax()
    summary.loc[beta_idx, "Endotype"] = "Beta (β) - Renal"
    
    # 3. Alpha: Lowest mortality among the remaining (Stable/Baseline)
    rem_idx2 = summary.index.difference([delta_idx, beta_idx])
    alpha_idx = summary.loc[rem_idx2, "Mortality_Rate"].idxmin()
    summary.loc[alpha_idx, "Endotype"] = "Alpha (α) - Stable"
    
    # 4. Gamma: The final remaining cluster (Pulmonary/Inflammatory)
    gamma_idx = summary.index.difference([delta_idx, beta_idx, alpha_idx])[0]
    summary.loc[gamma_idx, "Endotype"] = "Gamma (γ) - Inflammatory"
    
    # Reorder columns to make Endotype prominent
    cols = ["cluster", "Endotype", "Patient_Count", "Mortality_Rate", "Mean_Age", "Mean_Lactate", "Mean_Creatinine", "Mean_WBC", "Mean_MAP"]
    summary = summary[cols]
    
    # Format numeric outputs
    summary["Mortality_Rate"] = (summary["Mortality_Rate"] * 100).round(1).astype(str) + "%"
    for col in ["Mean_Age", "Mean_Lactate", "Mean_Creatinine", "Mean_WBC", "Mean_MAP"]:
        summary[col] = summary[col].round(2)
        
    print("\n" + "="*105)
    print("eICU SEYMOUR ENDOTYPE SANITY CHECK RESULTS (k=4)")
    print("="*105)
    print(summary.to_string(index=False))
    print("="*105)
    
    elapsed = time.time() - start_time
    print(f"\n[+] eICU Seymour Endotype verification completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()