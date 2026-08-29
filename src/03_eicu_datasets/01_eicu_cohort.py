"""
Build the eICU-CRD adult ICU base cohort.

Mirrors the MIMIC-IV inclusion criteria exactly so the external cohort is not
defined more or less strictly than the development cohort, and maps eICU column
names onto the MIMIC-IV schema used throughout the project.

Inclusion:
    first ICU stay per patient, ICU length of stay >= 24 h, age >= 18 years
Exclusion:
    elective surgery, taken from the APACHE predictor table

Two eICU-specific handling rules:
    ages recorded as "> 89" become 91, following the de-identification scheme
    stay order is proxied by ascending stay identifier, because eICU carries no
    absolute admission timestamp; this is a limitation of the database, not a
    choice, and is stated in the manuscript

Length of stay is derived from the unit discharge offset, which eICU records in
minutes.

Reads:
    data/raw/eicu-crd/2.0/{patient, apachePredVar}
Writes:
    eicu_base_cohort.parquet
"""

import time
from pathlib import Path
import duckdb

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
EICU_DIR = BASE_DIR / "data" / "raw" / "eicu-crd" / "2.0"
OUT_DIR = BASE_DIR / "data" / "processed" / "eicu"

# --- Main Execution ------------------------------------------------------
def main():
    print("[*] Executing eICU base cohort extraction pipeline...")
    start_time = time.time()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Output correctly prefixed with eicu_
    out_file = OUT_DIR / "eicu_base_cohort.parquet"
    
    print("    -> Initializing in-memory DuckDB...")
    con = duckdb.connect(database=':memory:')
    
    # We use sample_size=-1 in read_csv_auto because eICU mixes strings ('> 89') 
    # and integers in the age column, which can cause duckdb type-inference errors.
    query = f"""
    WITH patient_data AS (
        SELECT 
            uniquepid AS subject_id, 
            patienthealthsystemstayid AS hadm_id, 
            patientunitstayid AS stay_id, 
            CASE 
                WHEN gender = 'Male' THEN 'M' 
                WHEN gender = 'Female' THEN 'F' 
                ELSE NULL 
            END AS gender,
            -- Handle eICU's > 89 anonymization string
            CASE 
                WHEN age = '> 89' THEN 91 
                WHEN age = '' THEN NULL 
                ELSE CAST(age AS NUMERIC) 
            END AS age,
            ethnicity AS race,
            unittype AS first_careunit,
            -- eICU stores LOS in minutes (offset)
            unitdischargeoffset / 1440.0 AS icu_los_days,
            -- Map discharge status to mortality flag
            CASE 
                WHEN hospitaldischargestatus = 'Expired' THEN 1 
                ELSE 0 
            END AS hospital_expire_flag,
            -- Proxies chronological order of ICU stays using ascending ID assignment
            ROW_NUMBER() OVER (PARTITION BY uniquepid ORDER BY patientunitstayid ASC) as icu_seq
        FROM read_csv_auto('{EICU_DIR}/patient.csv.gz', sample_size=-1)
    ),
    apache_data AS (
        SELECT patientunitstayid, electivesurgery
        FROM read_csv_auto('{EICU_DIR}/apachePredVar.csv.gz', sample_size=-1)
    )
    SELECT 
        p.subject_id,
        p.hadm_id,
        p.stay_id,
        p.gender,
        p.age,
        p.race,
        p.first_careunit,
        p.icu_los_days,
        p.hospital_expire_flag
    FROM patient_data p
    LEFT JOIN apache_data a 
      ON p.stay_id = a.patientunitstayid
    WHERE p.icu_seq = 1 
      AND p.age >= 18
      AND p.icu_los_days >= 1.0
      AND (a.electivesurgery IS NULL OR a.electivesurgery = 0)
    """
    
    print("\n    -> Executing streaming joins and applying cohort filters...")
    print("       - Aligning columns to MIMIC-IV schema")
    print("       - Retaining first ICU stay only")
    print("       - Filtering for age >= 18")
    print("       - Filtering for ICU Length of Stay >= 24 hours")
    print("       - Excluding elective surgical admissions via APACHE variables")
    
    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! eICU base cohort established in {elapsed:.2f} seconds.")
    print(f"    -> Total Unique Adult Patients (Non-Elective): {count}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
