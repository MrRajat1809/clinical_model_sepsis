"""
04a_eicu_temporal_slice.py

Phase 9: External Validation (eICU Temporal Slice)
Extracts the critical temporal window (-48 to +48 hours) around the Suspected Infection Time (SIT).
- Maps the +/- 48 hour window to +/- 2880 minutes using eICU's relative offsets.
- Extracts vitals, labs, vasopressors, urine output, and mechanical ventilation status.
- Melts eICU's wide-format `vitalPeriodic` table into a long format using DuckDB UNPIVOT 
  to match the exact schema output of the MIMIC-IV pipeline.
"""

import time
from pathlib import Path

import duckdb

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
EICU_DIR = BASE_DIR / "data" / "raw" / "eicu-crd" / "2.0"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Executing eICU temporal extraction pipeline...")
    start_time = time.time()
    
    cohort_file = PROCESSED_DIR / "eicu_sepsis_phenotype_cohort.parquet"
    out_file = PROCESSED_DIR / "eicu_sepsis_temporal_data.parquet"
    
    if not cohort_file.exists():
        print(f"[ERROR] Cohort file not found at: {cohort_file}")
        return
        
    print("\n[*] Initializing in-memory DuckDB...")
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA threads=4;")
    
    query = f"""
    WITH cohort AS (
        SELECT stay_id, sit_offset 
        FROM '{cohort_file}'
    ),
    sliced_vitals_periodic AS (
        -- eICU stores continuous vitals in a wide format; we unpivot it to match MIMIC
        SELECT stay_id, 'vital' AS data_type, observationoffset AS event_time, metric AS itemid, valuenum
        FROM (
            SELECT 
                c.stay_id, 
                v.observationoffset, 
                v.heartrate, 
                v.sao2, 
                v.temperature, 
                v.systemicsystolic, 
                v.systemicdiastolic, 
                v.systemicmean, 
                v.respiration
            FROM read_csv_auto('{EICU_DIR}/vitalPeriodic.csv.gz', sample_size=-1) v
            INNER JOIN cohort c ON v.patientunitstayid = c.stay_id
            WHERE v.observationoffset >= (c.sit_offset - 2880)
              AND v.observationoffset <= (c.sit_offset + 2880)
        )
        UNPIVOT (
            valuenum FOR metric IN (heartrate, sao2, temperature, systemicsystolic, systemicdiastolic, systemicmean, respiration)
        )
        WHERE valuenum IS NOT NULL
    ),
    sliced_vitals_aperiodic AS (
        -- Non-invasive blood pressures are stored separately
        SELECT stay_id, 'vital' AS data_type, observationoffset AS event_time, metric AS itemid, valuenum
        FROM (
            SELECT 
                c.stay_id, 
                v.observationoffset, 
                v.noninvasivesystolic, 
                v.noninvasivediastolic, 
                v.noninvasivemean
            FROM read_csv_auto('{EICU_DIR}/vitalAperiodic.csv.gz', sample_size=-1) v
            INNER JOIN cohort c ON v.patientunitstayid = c.stay_id
            WHERE v.observationoffset >= (c.sit_offset - 2880)
              AND v.observationoffset <= (c.sit_offset + 2880)
        )
        UNPIVOT (
            valuenum FOR metric IN (noninvasivesystolic, noninvasivediastolic, noninvasivemean)
        )
        WHERE valuenum IS NOT NULL
    ),
    sliced_labs AS (
        SELECT 
            c.stay_id,
            'lab' AS data_type,
            l.labresultoffset AS event_time,
            lower(l.labname) AS itemid,
            l.labresult AS valuenum
        FROM read_csv_auto('{EICU_DIR}/lab.csv.gz', sample_size=-1) l
        INNER JOIN cohort c ON l.patientunitstayid = c.stay_id
        WHERE l.labresultoffset >= (c.sit_offset - 2880)
          AND l.labresultoffset <= (c.sit_offset + 2880)
          AND l.labresult IS NOT NULL
          AND lower(l.labname) IN (
              'pao2', 'total bilirubin', 'creatinine', 'platelets x 1000', 
              'wbc x 1000', 'hgb', 'bun', 'ph', 'lactate', 'pt', 'ptt', 
              'albumin', 'potassium', 'sodium', 'glucose', 'chloride', 'paco2'
          )
    ),
    sliced_vasos AS (
        SELECT 
            c.stay_id,
            'vaso' AS data_type,
            i.infusionoffset AS event_time,
            'vasopressor' AS itemid,
            TRY_CAST(i.drugrate AS FLOAT) AS valuenum
        FROM read_csv_auto('{EICU_DIR}/infusionDrug.csv.gz', sample_size=-1) i
        INNER JOIN cohort c ON i.patientunitstayid = c.stay_id
        WHERE i.infusionoffset >= (c.sit_offset - 2880)
          AND i.infusionoffset <= (c.sit_offset + 2880)
          AND i.drugrate IS NOT NULL
          AND (
              lower(i.drugname) LIKE '%norepinephrine%' OR 
              lower(i.drugname) LIKE '%epinephrine%' OR 
              lower(i.drugname) LIKE '%dopamine%' OR 
              lower(i.drugname) LIKE '%dobutamine%' OR 
              lower(i.drugname) LIKE '%vasopressin%' OR 
              lower(i.drugname) LIKE '%phenylephrine%'
          )
    ),
    sliced_uo AS (
        SELECT 
            c.stay_id,
            'uo' AS data_type,
            o.intakeoutputoffset AS event_time,
            'urine_output' AS itemid,
            o.cellvaluenumeric AS valuenum
        FROM read_csv_auto('{EICU_DIR}/intakeOutput.csv.gz', sample_size=-1) o
        INNER JOIN cohort c ON o.patientunitstayid = c.stay_id
        WHERE o.intakeoutputoffset >= (c.sit_offset - 2880)
          AND o.intakeoutputoffset <= (c.sit_offset + 2880)
          AND lower(o.cellpath) LIKE '%urine%'
          AND o.cellvaluenumeric IS NOT NULL
    ),
    sliced_vents AS (
        SELECT 
            c.stay_id,
            'vent' AS data_type,
            t.treatmentoffset AS event_time,
            '888888' AS itemid, -- Standardized Ventilation ID matching MIMIC
            1.0 AS valuenum
        FROM read_csv_auto('{EICU_DIR}/treatment.csv.gz', sample_size=-1) t
        INNER JOIN cohort c ON t.patientunitstayid = c.stay_id
        WHERE t.treatmentoffset >= (c.sit_offset - 2880)
          AND t.treatmentoffset <= (c.sit_offset + 2880)
          AND (
              lower(t.treatmentstring) LIKE '%mechanical ventilation%' OR 
              lower(t.treatmentstring) LIKE '%ventilator%' OR
              lower(t.treatmentstring) LIKE '%intubation%'
          )
    )
    SELECT * FROM sliced_vitals_periodic
    UNION ALL
    SELECT * FROM sliced_vitals_aperiodic
    UNION ALL
    SELECT * FROM sliced_labs
    UNION ALL
    SELECT * FROM sliced_vasos
    UNION ALL
    SELECT * FROM sliced_uo
    UNION ALL
    SELECT * FROM sliced_vents
    """
    
    print("[*] Streaming multi-table temporal slice (-2880m to +2880m from SIT)...")
    print("    - Extracting and unpivoting vitals (vitalPeriodic, vitalAperiodic)")
    print("    - Extracting targeted labs (lab)")
    print("    - Extracting raw vasopressor therapies (infusionDrug)")
    print("    - Extracting urine output (intakeOutput)")
    print("    - Extracting mechanical ventilation statuses (treatment)")
    
    # Execute and write directly to Parquet
    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    # Verify the output
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! eICU temporal slice extracted in {elapsed:.2f} seconds.")
    print(f"    -> Total Time-Series Datapoints Retrieved: {count:,}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()