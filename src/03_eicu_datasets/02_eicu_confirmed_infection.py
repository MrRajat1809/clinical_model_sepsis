"""
Identify suspected infection from recorded diagnoses.

eICU cannot support the culture-plus-antibiotic coupling used in MIMIC-IV: its
microbiology and medication tables are too sparsely populated for the temporal
rules to be applied reliably. Infection is therefore taken from active and
admission diagnoses recorded within 24 hours either side of ICU admission, and
the suspected infection time is the earliest such entry.

Qualifying terms cover sepsis and septic presentations, pneumonia, generic
infection, peritonitis, meningitis, cholangitis, endocarditis and bacteraemia.

This asymmetry with MIMIC-IV is the largest structural difference between the
two cohorts and belongs in the limitations. The elective-surgery exclusion
already applied in 01 does the work of keeping prophylactic surgical cases out.

Reads:
    eicu_base_cohort.parquet
    data/raw/eicu-crd/2.0/{diagnosis, admissionDx}
Writes:
    eicu_infection_cohort.parquet, carrying sit_offset in minutes
"""

import time
from pathlib import Path
import duckdb

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
EICU_DIR = BASE_DIR / "data" / "raw" / "eicu-crd" / "2.0"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

# --- Main Execution ------------------------------------------------------
def main():
    print("[*] Executing eICU active diagnosis infection filter pipeline...")
    start_time = time.time()
    
    base_cohort_file = PROCESSED_DIR / "eicu_base_cohort.parquet"
    out_file = PROCESSED_DIR / "eicu_infection_cohort.parquet"
    
    if not base_cohort_file.exists():
        print(f"[ERROR] Base cohort not found at: {base_cohort_file}")
        return
        
    print("\n    -> Initializing in-memory DuckDB...")
    con = duckdb.connect(database=':memory:')
    
    # --- Extract Active Infection Diagnoses ------------------------------
    print("    -> Scanning `diagnosis` and `admissionDx` tables for active infections...")
    
    query = f"""
    WITH infectious_dx AS (
        -- Source 1: Active ICU Diagnoses
        SELECT 
            patientunitstayid AS stay_id, 
            diagnosisoffset AS dx_offset, 
            diagnosisstring AS dx_string
        FROM read_csv_auto('{EICU_DIR}/diagnosis.csv.gz', sample_size=-1)
        WHERE (
            lower(diagnosisstring) LIKE '%sepsis%' OR
            lower(diagnosisstring) LIKE '%septic%' OR
            lower(diagnosisstring) LIKE '%infection%' OR
            lower(diagnosisstring) LIKE '%pneumonia%' OR
            lower(diagnosisstring) LIKE '%peritonitis%' OR
            lower(diagnosisstring) LIKE '%meningitis%' OR
            lower(diagnosisstring) LIKE '%cholangitis%' OR
            lower(diagnosisstring) LIKE '%endocarditis%' OR
            lower(diagnosisstring) LIKE '%bacteremia%'
        )
        AND diagnosisoffset >= -1440 AND diagnosisoffset <= 1440

        UNION ALL

        -- Source 2: Admission Diagnoses
        SELECT 
            patientunitstayid AS stay_id, 
            admitdxenteredoffset AS dx_offset, 
            admitdxpath AS dx_string
        FROM read_csv_auto('{EICU_DIR}/admissionDx.csv.gz', sample_size=-1)
        WHERE (
            lower(admitdxpath) LIKE '%sepsis%' OR
            lower(admitdxpath) LIKE '%septic%' OR
            lower(admitdxpath) LIKE '%infection%' OR
            lower(admitdxpath) LIKE '%pneumonia%' OR
            lower(admitdxpath) LIKE '%peritonitis%' OR
            lower(admitdxpath) LIKE '%meningitis%' OR
            lower(admitdxpath) LIKE '%cholangitis%' OR
            lower(admitdxpath) LIKE '%endocarditis%' OR
            lower(admitdxpath) LIKE '%bacteremia%'
        )
        AND admitdxenteredoffset >= -1440 AND admitdxenteredoffset <= 1440
    ),
    first_dx AS (
        -- Take the earliest diagnosis timestamp (Suspected Infection Time)
        SELECT 
            stay_id, 
            MIN(dx_offset) as sit_offset
        FROM infectious_dx
        GROUP BY stay_id
    )
    SELECT 
        b.*, 
        f.sit_offset
    FROM '{base_cohort_file}' b
    JOIN first_dx f ON b.stay_id = f.stay_id
    """
    
    print("    -> Joining identified infections with the base cohort...")
    
    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! eICU infection cohort established in {elapsed:.2f} seconds.")
    print(f"    -> Total True Positive Infection Patients: {count:,}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
