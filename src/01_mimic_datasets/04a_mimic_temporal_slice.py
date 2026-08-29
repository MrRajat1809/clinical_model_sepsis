"""
Extract the raw clinical time series around the suspected infection time.

Pulls every observation from -48 h to +48 h relative to SIT across five source
tables and emits them in one long-format table keyed by stay, timestamp and
item. Two normalisations are applied at extraction because they are cheaper here
than downstream:

    temperature charted in Fahrenheit is converted to Celsius
    vasopressor rates are standardised to mcg/kg/min, or units/min for
    vasopressin, using recorded infusion weight with an 80 kg fallback

Every dose unit in use is mapped explicitly and anything unrecognised becomes
NULL, so an unconverted rate can never reach the tensor. Ventilation is taken
from both procedure records and charted ventilator parameters, since either
alone misses a substantial share of ventilated stays.

Reads:
    mimic_sepsis_phenotype_cohort.parquet
    data/raw/mimiciv/3.1/{icu/chartevents, hosp/labevents, icu/inputevents,
                          icu/outputevents, icu/procedureevents}
Writes:
    mimic_sepsis_temporal_data.parquet
"""

import time
from pathlib import Path
import duckdb

# --- Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MIMIC_DIR = BASE_DIR / "data" / "raw" / "mimiciv" / "3.1"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

# --- Main Execution ------------------------------------------------------
def main():
    print("Executing MIMIC-IV temporal extraction pipeline...")
    start_time = time.time()
    
    cohort_file = PROCESSED_DIR / "mimic_sepsis_phenotype_cohort.parquet"
    out_file = PROCESSED_DIR / "mimic_sepsis_temporal_data.parquet"
    
    if not cohort_file.exists():
        print(f"[ERROR] Cohort file not found at: {cohort_file}")
        return
        
    print("\n[*] Initializing in-memory DuckDB...")
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA threads=4;")
    
    # ItemID Definitions
    vital_itemids = "220045, 220210, 220181, 220052, 223761, 223762, 220739, 223900, 223901, 223835, 220277"
    
    lab_itemids = (
        "50821, "        # PaO2
        "50885, "        # tBil
        "50912, "        # Scr
        "51265, "        # Platelets
        "51301, 51300, " # WBC
        "51222, "        # Hemoglobin
        "51006, "        # BUN
        "50820, "        # pH
        "50813, "        # Lactate
        "51274, "        # PT
        "51275, "        # APTT
        "50862, "        # Albumin
        "50971, 50822, " # Potassium
        "50983, 50824, " # Sodium
        "50931, 50809, " # Glucose
        "50902, 50806, " # Chloride
        "50818"          # PaCO2
    )
    
    # Norepi(221906), Epi(221289), Dopa(221662), Dobutamine(221653), Vaso(222315), Phenylephrine(221749)
    vaso_itemids = "221906, 221289, 221662, 221653, 222315, 221749"
    vent_proc_itemids = "225792, 225794"
    vent_chart_itemids = "220339, 223849, 223848" # PEEP, Vent Mode, Vent Type
    uo_itemids = "226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558"
    
    query = f"""
    WITH cohort AS (
        SELECT subject_id, hadm_id, stay_id, suspected_infection_time 
        FROM '{cohort_file}'
    ),
    sliced_vitals AS (
        SELECT 
            c.stay_id,
            'vital' AS data_type,
            c.charttime AS event_time,
            c.itemid,
            -- Inline standardization: Fahrenheit to Celsius
            CASE 
                WHEN c.itemid = 223761 THEN (c.valuenum - 32.0) * 5.0 / 9.0
                ELSE c.valuenum
            END AS valuenum
        FROM read_csv_auto('{MIMIC_DIR}/icu/chartevents.csv.gz') c
        INNER JOIN cohort p ON c.stay_id = p.stay_id
        WHERE c.charttime >= (p.suspected_infection_time - INTERVAL 48 HOUR)
          AND c.charttime <= (p.suspected_infection_time + INTERVAL 48 HOUR)
          AND c.valuenum IS NOT NULL
          AND c.itemid IN ({vital_itemids})
    ),
    sliced_labs AS (
        SELECT 
            p.stay_id,
            'lab' AS data_type,
            l.charttime AS event_time,
            l.itemid,
            l.valuenum
        FROM read_csv_auto('{MIMIC_DIR}/hosp/labevents.csv.gz') l
        INNER JOIN cohort p ON l.subject_id = p.subject_id AND l.hadm_id = p.hadm_id
        WHERE l.charttime >= (p.suspected_infection_time - INTERVAL 48 HOUR)
          AND l.charttime <= (p.suspected_infection_time + INTERVAL 48 HOUR)
          AND l.valuenum IS NOT NULL
          AND l.itemid IN ({lab_itemids})
    ),
    sliced_vasos AS (
        SELECT 
            p.stay_id,
            'vaso' AS data_type,
            i.starttime AS event_time,
            i.itemid,
            -- Standardise every dose unit in use; anything unrecognised
            -- becomes NULL rather than passing through unconverted.
            CASE 
                -- Convert hourly rates to minute rates
                WHEN lower(i.rateuom) = 'units/hour' THEN (i.rate / 60.0)
                
                -- Convert mg to mcg
                WHEN lower(i.rateuom) = 'mg/kg/min' THEN (i.rate * 1000.0)
                WHEN lower(i.rateuom) = 'mg/min' THEN 
                    CASE WHEN i.patientweight > 0 THEN ((i.rate * 1000.0) / i.patientweight) ELSE ((i.rate * 1000.0) / 80.0) END
                
                -- Normalize non-weight based mcg rates using 80kg fallback
                WHEN lower(i.rateuom) = 'mcg/min' THEN 
                    CASE WHEN i.patientweight > 0 THEN (i.rate / i.patientweight) ELSE (i.rate / 80.0) END
                
                -- Passthrough standard units
                WHEN lower(i.rateuom) = 'mcg/kg/min' THEN i.rate
                WHEN lower(i.rateuom) = 'units/min' THEN i.rate
                
                -- Drop any unrecognized units to prevent silent garbage
                ELSE NULL
            END AS valuenum
        FROM read_csv_auto('{MIMIC_DIR}/icu/inputevents.csv.gz') i
        INNER JOIN cohort p ON i.stay_id = p.stay_id
        WHERE i.starttime >= (p.suspected_infection_time - INTERVAL 48 HOUR)
          AND i.starttime <= (p.suspected_infection_time + INTERVAL 48 HOUR)
          AND i.rate IS NOT NULL
          AND i.rate > 0
          AND i.itemid IN ({vaso_itemids})
    ),
    sliced_uo AS (
        SELECT 
            o.stay_id,
            'uo' AS data_type,
            o.charttime AS event_time,
            o.itemid,
            o.value AS valuenum
        FROM read_csv_auto('{MIMIC_DIR}/icu/outputevents.csv.gz') o
        INNER JOIN cohort p ON o.stay_id = p.stay_id
        WHERE o.charttime >= (p.suspected_infection_time - INTERVAL 48 HOUR)
          AND o.charttime <= (p.suspected_infection_time + INTERVAL 48 HOUR)
          AND o.value IS NOT NULL
          AND o.itemid IN ({uo_itemids})
    ),
    sliced_vents AS (
        -- Source 1: Procedure Events
        SELECT 
            p.stay_id,
            'vent' AS data_type,
            CASE 
                WHEN v.starttime < (p.suspected_infection_time - INTERVAL 48 HOUR) 
                THEN (p.suspected_infection_time - INTERVAL 48 HOUR)
                ELSE v.starttime
            END AS event_time,
            888888 AS itemid, -- Standardized Ventilation ID
            1.0 AS valuenum
        FROM read_csv_auto('{MIMIC_DIR}/icu/procedureevents.csv.gz') v
        INNER JOIN cohort p ON v.stay_id = p.stay_id
        WHERE v.starttime <= (p.suspected_infection_time + INTERVAL 48 HOUR)
          AND v.endtime >= (p.suspected_infection_time - INTERVAL 48 HOUR)
          AND v.itemid IN ({vent_proc_itemids})
          
        UNION ALL
        
        -- Source 2: Chart Events (PEEP, Vent Modes)
        SELECT 
            c.stay_id,
            'vent' AS data_type,
            c.charttime AS event_time,
            888888 AS itemid, -- Standardized Ventilation ID
            1.0 AS valuenum
        FROM read_csv_auto('{MIMIC_DIR}/icu/chartevents.csv.gz') c
        INNER JOIN cohort p ON c.stay_id = p.stay_id
        WHERE c.charttime >= (p.suspected_infection_time - INTERVAL 48 HOUR)
          AND c.charttime <= (p.suspected_infection_time + INTERVAL 48 HOUR)
          AND c.itemid IN ({vent_chart_itemids})
          AND c.valuenum IS NOT NULL
    )
    SELECT * FROM sliced_vitals
    UNION ALL
    SELECT * FROM sliced_labs
    UNION ALL
    SELECT * FROM sliced_vasos
    -- Unrecognised dose units became NULL above; drop those rows so they
    -- never reach the tensor.
    WHERE valuenum IS NOT NULL
    UNION ALL
    SELECT * FROM sliced_uo
    UNION ALL
    SELECT * FROM sliced_vents
    """
    
    print("[*] Streaming multi-table temporal slice (-48h to +48h from SIT)...")
    print("    - Extracting targeted vitals (chartevents)")
    print("    - Extracting targeted labs (labevents)")
    print("    - Extracting raw vasopressor therapies (inputevents) with strict unit normalization")
    print("    - Extracting urine output (outputevents)")
    print("    - Extracting mechanical ventilation statuses (procedureevents + chartevents)")
    
    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! Temporal slice extracted in {elapsed:.2f} seconds.")
    print(f"    -> Total Time-Series Datapoints Retrieved: {count}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
