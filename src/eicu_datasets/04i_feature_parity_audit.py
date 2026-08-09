"""
04i_feature_parity_audit.py

Canonical Feature Parity Audit.
Audits the exact 30 features the XGBoost model expects.
[FIX]: Now accurately concatenates the 4 eICU data streams (Vitals, GCS, FiO2, Pressors)
       and uses the corrected eICU string mappings.
"""

import time
import polars as pl
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

MIMIC_TEMP = BASE_DIR / "data" / "processed" / "mimiciv" / "sepsis_temporal_data_cleaned.parquet"
MIMIC_COHORT = BASE_DIR / "data" / "processed" / "mimiciv" / "final_sepsis3_cohort.parquet"
EICU_COHORT = PROCESSED_DIR / "eicu_final_sepsis3_cohort.parquet"

# The 4 eICU Streams
EICU_VITALS = PROCESSED_DIR / "eicu_sepsis_temporal_data_cleaned.parquet"
EICU_GCS = PROCESSED_DIR / "eicu_gcs_timeline.parquet"
EICU_FIO2 = PROCESSED_DIR / "eicu_fio2_timeline.parquet"
EICU_PRESSORS = PROCESSED_DIR / "eicu_standardized_pressors.parquet"

CANONICAL_FEATURES = [
    "hr", "map", "rr", "temp_c", "spo2", 
    "gcs_eye", "gcs_verbal", "gcs_motor", 
    "pao2", "fio2", "pf_ratio", "paco2", 
    "lactate", "creatinine", "bun", "bilirubin", 
    "platelets", "wbc", "hemoglobin", 
    "ph", "pt", "aptt", "albumin", 
    "potassium", "sodium", "glucose", "chloride", 
    "urine_output", "neq", "vent"
]

MIMIC_MAP = {
    220045: "hr", 220181: "map", 220052: "map", 220210: "rr", 
    223762: "temp_c", 223761: "temp_c", 220277: "spo2",
    51301: "wbc", 51300: "wbc", 51265: "platelets", 227457: "platelets", 
    51222: "hemoglobin", 50912: "creatinine", 220615: "creatinine", 
    51006: "bun", 50885: "bilirubin", 225664: "bilirubin", 
    50820: "ph", 50813: "lactate", 227442: "lactate", 
    51274: "pt", 51275: "aptt", 50862: "albumin", 220862: "albumin", 
    50971: "potassium", 50822: "potassium", 50983: "sodium", 
    50824: "sodium", 50931: "glucose", 50809: "glucose", 
    50902: "chloride", 50806: "chloride", 
    50818: "paco2", 50821: "pao2", 223835: "fio2", 
    220739: "gcs_eye", 223900: "gcs_verbal", 223901: "gcs_motor",
    226559: "urine_output", 226560: "urine_output", 226561: "urine_output", 
    226584: "urine_output", 226563: "urine_output", 226564: "urine_output", 
    226565: "urine_output", 226567: "urine_output", 226557: "urine_output", 
    226558: "urine_output"
}

EICU_MAP = {
    "heartrate": "hr", "systemicmean": "map", "noninvasivemean": "map", "respiration": "rr", 
    "temperature": "temp_c", "sao2": "spo2",
    "wbc x 1000": "wbc", "platelets x 1000": "platelets", "hgb": "hemoglobin", 
    "creatinine": "creatinine", "bun": "bun", "total bilirubin": "bilirubin", 
    "ph": "ph", "lactate": "lactate", 
    "pt": "pt", "ptt": "aptt", "albumin": "albumin", 
    "potassium": "potassium", "sodium": "sodium", "glucose": "glucose", "chloride": "chloride", 
    "pao2": "pao2", "paco2": "paco2",
    "fio2": "fio2", "gcs_eye": "gcs_eye", "gcs_verbal": "gcs_verbal", "gcs_motor": "gcs_motor",
    "urine_output": "urine_output", "vent": "vent", 
    "norepinephrine": "neq", "epinephrine": "neq", "vasopressin": "neq", 
    "dopamine": "neq", "phenylephrine": "neq"
}

def load_eicu_temporal():
    dfs = []
    if EICU_VITALS.exists():
        dfs.append(pl.read_parquet(EICU_VITALS).select(["stay_id", "itemid", "valuenum"]))
    if EICU_GCS.exists():
        dfs.append(pl.read_parquet(EICU_GCS).select(["stay_id", "itemid", "valuenum"]))
    if EICU_FIO2.exists():
        dfs.append(pl.read_parquet(EICU_FIO2).select(["stay_id", "itemid", "valuenum"]))
    if EICU_PRESSORS.exists():
        dfs.append(pl.read_parquet(EICU_PRESSORS).select([
            "stay_id", pl.col("drug_type").alias("itemid"), pl.col("standardized_rate").alias("valuenum")
        ]))
    return pl.concat(dfs) if dfs else pl.DataFrame()

def get_mimic_stats(total_pts):
    df = pl.read_parquet(MIMIC_TEMP).select(["stay_id", "itemid", "valuenum"]).with_columns(
        pl.col("itemid").cast(pl.Int64, strict=False)
    )
    map_df = pl.DataFrame({"itemid": list(MIMIC_MAP.keys()), "feature": list(MIMIC_MAP.values())}, schema={"itemid": pl.Int64, "feature": pl.Utf8})
    df_mapped = df.join(map_df, on="itemid", how="inner")
    
    return df_mapped.group_by("feature").agg([
        pl.col("valuenum").count().alias("MIMIC_N"),
        (pl.col("stay_id").n_unique() / total_pts * 100).round(1).alias("MIMIC_Cov%"),
    ])

def get_eicu_stats(total_pts):
    df = load_eicu_temporal().with_columns(pl.col("itemid").cast(pl.Utf8, strict=False))
    if df.is_empty(): return pl.DataFrame()
    
    map_df = pl.DataFrame({"itemid": list(EICU_MAP.keys()), "feature": list(EICU_MAP.values())}, schema={"itemid": pl.Utf8, "feature": pl.Utf8})
    df_mapped = df.join(map_df, on="itemid", how="inner")
    
    return df_mapped.group_by("feature").agg([
        pl.col("valuenum").count().alias("eICU_N"),
        (pl.col("stay_id").n_unique() / total_pts * 100).round(1).alias("eICU_Cov%"),
    ])

def run_audit():
    print("[*] Building Synchronized Canonical Feature Parity Matrix...")
    m_pts = pl.read_parquet(MIMIC_COHORT).height
    e_pts = pl.read_parquet(EICU_COHORT).height

    m_stats = get_mimic_stats(m_pts)
    e_stats = get_eicu_stats(e_pts)

    base = pl.DataFrame({"feature": CANONICAL_FEATURES})
    comp = base.join(m_stats, on="feature", how="left").join(e_stats, on="feature", how="left").fill_null(0.0)

    print("\n=========================================================================================")
    print(" TRUE CANONICAL FEATURE COVERAGE AUDIT (All 4 Streams Combined)")
    print("=========================================================================================")
    print(f"{'Feature':<15} | {'MIMIC N':<10} | {'eICU N':<10} | {'M_Cov%':<7} | {'e_Cov%':<7} | {'Δ Miss%':<8}")
    print("-" * 80)

    for row in comp.iter_rows(named=True):
        f = row["feature"].upper()
        m_n, e_n = int(row["MIMIC_N"]), int(row["eICU_N"])
        m_cov, e_cov = row["MIMIC_Cov%"], row["eICU_Cov%"]
        
        d_miss = round(abs(m_cov - e_cov), 1)
        flag = "🚨" if e_cov == 0.0 or d_miss > 40.0 else ""
        
        print(f"{f:<15} | {m_n:<10,} | {e_n:<10,} | {m_cov:>5}% | {e_cov:>5}% | {d_miss:>5}% {flag}")

if __name__ == "__main__":
    run_audit()