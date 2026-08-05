"""
02_mimic_confirmed_infection.py

Applies a strict "Confirmed Infection" filter to the base cohort based on Sepsis-3 guidelines.
Requires a microbiological culture and an IV antibiotic order that meet specific temporal rules:
- If culture first: Antibiotics must follow within 72 hours.
- If antibiotics first: Culture must follow within 24 hours.

Uses a Gaps and Islands SQL approach to calculate cumulative treatment episodes, 
ensuring continuous antibiotic therapies (even across drug switches) are accurately tracked.

[FIX APPLIED]: Updated 'admission_age' to 'age' to match upstream changes.
[FIX APPLIED]: Passed through new demographics, ICU context, and mortality labels (race, 
admission_type, first_careunit, hospital_expire_flag, dod) to prevent data loss.
"""

import time
from pathlib import Path

import duckdb

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
MIMIC_DIR = BASE_DIR / "data" / "raw" / "mimiciv" / "3.1"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Executing MIMIC-IV confirmed infection filter pipeline...")
    start_time = time.time()
    
    base_cohort_file = PROCESSED_DIR / "base_cohort.parquet"
    out_file = PROCESSED_DIR / "infection_cohort.parquet"
    
    if not base_cohort_file.exists():
        print(f"[ERROR] Base cohort not found at: {base_cohort_file}")
        return
        
    print("\n[*] Initializing in-memory DuckDB...")
    con = duckdb.connect(database=':memory:')
    
    query = f"""
    WITH cultures AS (
        SELECT 
            subject_id, 
            hadm_id, 
            COALESCE(charttime, chartdate) AS culture_time,
            spec_type_desc
        FROM read_csv_auto('{MIMIC_DIR}/hosp/microbiologyevents.csv.gz')
        WHERE spec_itemid IS NOT NULL 
    ),
    raw_abx AS (
        SELECT 
            subject_id, 
            hadm_id, 
            starttime,
            stoptime,
            drug
        FROM read_csv_auto('{MIMIC_DIR}/hosp/prescriptions.csv.gz')
        WHERE route IN ('IV', 'IV DRIP', 'IV PIGGYBACK', 'IV PUSH')
          AND regexp_matches(lower(drug), '.*(vanco|cef|peni|mycin|floxacin|bactam|cillin|mero|doxy|azithro|linezolid|tazo).*')
    ),
    abx_lag AS (
        -- Get the stop time of the previous antibiotic for this patient/admission
        SELECT 
            subject_id, 
            hadm_id, 
            starttime, 
            stoptime,
            LAG(stoptime) OVER (PARTITION BY subject_id, hadm_id ORDER BY starttime) as prev_stop
        FROM raw_abx
    ),
    abx_islands AS (
        -- If the gap between the previous stop and current start is > 24 hours, it's a new treatment island
        SELECT 
            subject_id, 
            hadm_id, 
            starttime, 
            stoptime,
            CASE 
                WHEN prev_stop IS NULL THEN 1 
                WHEN starttime > prev_stop + INTERVAL 24 HOUR THEN 1 
                ELSE 0 
            END as is_new_island
        FROM abx_lag
    ),
    abx_groups AS (
        -- Create a unique ID for each continuous treatment episode
        SELECT 
            subject_id, 
            hadm_id, 
            starttime, 
            stoptime,
            SUM(is_new_island) OVER (PARTITION BY subject_id, hadm_id ORDER BY starttime) as episode_id
        FROM abx_islands
    ),
    iv_antibiotic_episodes AS (
        -- Collapse the episodes into start, stop, and total duration
        SELECT 
            subject_id, 
            hadm_id, 
            MIN(starttime) AS abx_start_time,
            MAX(stoptime) AS abx_stop_time,
            EXTRACT(EPOCH FROM (MAX(stoptime) - MIN(starttime))) / 3600.0 AS abx_duration_hours
        FROM abx_groups
        GROUP BY subject_id, hadm_id, episode_id
    ),
    coupled_events AS (
        SELECT 
            b.subject_id,
            b.hadm_id,
            b.stay_id,
            b.gender,
            b.age,                      -- [FIXED] Updated from admission_age
            b.race,                     -- [ADDED] Pass-through
            b.admission_type,           -- [ADDED] Pass-through
            b.first_careunit,           -- [ADDED] Pass-through
            b.icu_intime,
            b.icu_outtime,
            b.icu_los_days,
            b.hospital_expire_flag,     -- [ADDED] Pass-through labels
            b.dod,                      -- [ADDED] Pass-through labels
            c.culture_time,
            a.abx_start_time,
            a.abx_stop_time,
            a.abx_duration_hours,
            -- Determine the Suspected Infection Time (SIT)
            CASE 
                WHEN c.culture_time <= a.abx_start_time THEN c.culture_time
                ELSE a.abx_start_time
            END AS suspected_infection_time
        FROM '{base_cohort_file}' b
        JOIN cultures c ON b.subject_id = c.subject_id AND b.hadm_id = c.hadm_id
        JOIN iv_antibiotic_episodes a ON b.subject_id = a.subject_id AND b.hadm_id = a.hadm_id
        WHERE 
            -- Sepsis-3 Temporal Logic:
            (
                (c.culture_time <= a.abx_start_time AND a.abx_start_time <= (c.culture_time + INTERVAL 72 HOUR))
                OR
                (a.abx_start_time < c.culture_time AND c.culture_time <= (a.abx_start_time + INTERVAL 24 HOUR))
            )
            -- 72-Hour Cumulative Treatment Rule
            AND (
                a.abx_duration_hours >= 72.0 
                OR (b.icu_los_days < 3.0) 
            )
    ),
    first_infections AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY suspected_infection_time ASC) as infection_seq
        FROM coupled_events
        WHERE 
            -- Restrict SIT to the immediate ICU presentation window
            suspected_infection_time >= (icu_intime - INTERVAL 24 HOUR)
            AND suspected_infection_time <= (icu_intime + INTERVAL 24 HOUR)
    )
    SELECT * EXCLUDE (infection_seq)
    FROM first_infections
    WHERE infection_seq = 1
    """
    
    print("[*] Executing gaps-and-islands temporal logic and applying Sepsis-3 filters...")
    print("    - Coupling cultures with IV antibiotics")
    print("    - Enforcing Sepsis-3 temporal rules (72h / 24h window)")
    print("    - Calculating 72-hour cumulative continuous treatment duration")
    print("    - Restricting Suspected Infection Time (SIT) to ICU presentation window")
    
    # Execute and write directly to Parquet
    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    # Verify the output
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! Confirmed infection cohort established in {elapsed:.2f} seconds.")
    print(f"    -> Total True Positive Infection Patients: {count}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()