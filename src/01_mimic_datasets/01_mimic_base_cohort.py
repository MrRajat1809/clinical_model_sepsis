"""
01_mimic_base_cohort.py

Extracts the foundational unique adult ICU cohort from MIMIC-IV.
Filters: First ICU stay only, Age >= 18, ICU Length of Stay >= 24 hours.
Excludes: Elective surgical admissions to reduce prophylactic antibiotic false positives.

Features included:
- Mortality labels (hospital_expire_flag, dod).
- ICU context (first_careunit).
- Exact age calculation at admission.

Exports to a highly compressed Parquet file for downstream steps.
"""

import time
from pathlib import Path
import duckdb

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
MIMIC_DIR = BASE_DIR / "data" / "raw" / "mimiciv" / "3.1"
OUT_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Executing MIMIC-IV base ICU cohort extraction pipeline...")
    start_time = time.time()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "mimic_base_cohort.parquet"
    
    print("\n[*] Initializing in-memory DuckDB...")
    con = duckdb.connect(database=':memory:')
    
    query = f"""
    WITH ranked_icu AS (
        SELECT 
            subject_id, 
            hadm_id, 
            stay_id, 
            intime, 
            outtime, 
            los,
            first_careunit,
            ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime ASC) as icu_seq
        FROM read_csv_auto('{MIMIC_DIR}/icu/icustays.csv.gz')
    ),
    base_cohort AS (
        SELECT 
            r.subject_id,
            r.hadm_id,
            r.stay_id,
            p.gender,
            p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year) AS age,
            a.race,
            a.admission_type,
            r.first_careunit,
            r.intime as icu_intime,
            r.outtime as icu_outtime,
            r.los as icu_los_days,
            a.hospital_expire_flag,
            p.dod
        FROM ranked_icu r
        JOIN read_csv_auto('{MIMIC_DIR}/hosp/patients.csv.gz') p 
          ON r.subject_id = p.subject_id
        JOIN read_csv_auto('{MIMIC_DIR}/hosp/admissions.csv.gz') a 
          ON r.hadm_id = a.hadm_id
        WHERE r.icu_seq = 1 
          AND r.los >= 1.0
          AND a.admission_type != 'ELECTIVE'
    )
    SELECT * FROM base_cohort 
    WHERE age >= 18
    """
    
    print("[*] Executing streaming joins and applying cohort filters...")
    print("    - Retaining first ICU stay only")
    print("    - Filtering for age >= 18")
    print("    - Filtering for ICU Length of Stay >= 24 hours")
    print("    - Excluding elective surgical admissions")
    
    # Execute and write directly to Parquet to save RAM
    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    # Verify the output
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! Base cohort established in {elapsed:.2f} seconds.")
    print(f"    -> Total Unique Adult Patients (Non-Elective): {count}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()