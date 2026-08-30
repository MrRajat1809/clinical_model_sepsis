"""
Identify suspected infection and derive the suspected infection time (SIT).

Couples microbiological cultures to intravenous antibiotic administration using
the Sepsis-3 temporal rules. Antibiotic orders are first collapsed into
continuous treatment episodes with a gaps-and-islands pass, so that consecutive
courses and drug switches count as one episode rather than several; a gap of
more than 24 h between the previous stop and the next start opens a new episode.

A culture and an episode are coupled when either:
    the culture precedes antibiotics by no more than 72 h, or
    antibiotics precede the culture by no more than 24 h

An episode must reach 72 h of cumulative treatment -- the union of its
prescription intervals, so concurrent agents count once and a gap between
courses counts not at all -- waived when the ICU stay is
shorter than three days. SIT is the earlier of the culture time and the
antibiotic start time, restricted to +/- 24 h around ICU admission. The earliest
qualifying event per patient is kept.

Reads:
    mimic_base_cohort.parquet
    data/raw/mimiciv/3.1/hosp/{microbiologyevents, prescriptions}
Writes:
    mimic_infection_cohort.parquet
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
    print("Executing MIMIC-IV confirmed infection filter pipeline...")
    start_time = time.time()
    
    base_cohort_file = PROCESSED_DIR / "mimic_base_cohort.parquet"
    out_file = PROCESSED_DIR / "mimic_infection_cohort.parquet"
    
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
        -- Get the stop time of the previous antibiotic for this patient/admission.
        --
        -- starttime alone is not a unique ordering: concurrent IV antibiotics
        -- are prescribed at the same timestamp routinely and MIMIC records
        -- those times coarsely, so a patient-admission carries many ties.
        -- Ordering on it alone leaves the row order among ties to the engine,
        -- and a different DuckDB build or thread count then returns a
        -- different prev_stop, which moves episode boundaries and changes who
        -- clears the 72-hour rule below. (stoptime, drug) completes the key;
        -- rows tied on all three are indistinguishable in every column this
        -- query uses.
        SELECT 
            subject_id, 
            hadm_id, 
            starttime, 
            stoptime,
            drug,
            LAG(stoptime) OVER (
                PARTITION BY subject_id, hadm_id
                ORDER BY starttime, stoptime, drug
            ) as prev_stop
        FROM raw_abx
    ),
    abx_islands AS (
        -- If the gap between the previous stop and current start is > 24 hours, it's a new treatment island
        SELECT 
            subject_id, 
            hadm_id, 
            starttime, 
            stoptime,
            drug,
            CASE 
                WHEN prev_stop IS NULL THEN 1 
                WHEN starttime > prev_stop + INTERVAL 24 HOUR THEN 1 
                ELSE 0 
            END as is_new_island
        FROM abx_lag
    ),
    abx_groups AS (
        -- Create a unique ID for each continuous treatment episode. The
        -- ordering matches abx_lag exactly: the island flags were computed
        -- under that order, so summing them under any other order would count
        -- boundaries the flags were not describing. The frame is stated rather
        -- than defaulted -- the default RANGE frame makes tied rows peers and
        -- hands them one shared episode id, which is not a running count.
        SELECT 
            subject_id, 
            hadm_id, 
            starttime, 
            stoptime,
            drug,
            SUM(is_new_island) OVER (
                PARTITION BY subject_id, hadm_id
                ORDER BY starttime, stoptime, drug
                ROWS UNBOUNDED PRECEDING
            ) as episode_id
        FROM abx_islands
    ),
    abx_coverage AS (
        -- Inside an episode, prescriptions overlap (concurrent agents) and leave
        -- gaps (up to the 24 h the island rule tolerates). Carry the furthest
        -- stop time reached by anything ordered earlier, so the two can be told
        -- apart.
        SELECT
            subject_id,
            hadm_id,
            episode_id,
            starttime,
            stoptime,
            drug,
            MAX(stoptime) OVER (
                PARTITION BY subject_id, hadm_id, episode_id
                ORDER BY starttime, stoptime, drug
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS covered_until
        FROM abx_groups
    ),
    abx_blocks AS (
        -- A prescription starting after everything before it has stopped opens a
        -- new block of continuous cover. One starting at or before that point
        -- extends the block already running.
        SELECT
            subject_id,
            hadm_id,
            episode_id,
            starttime,
            stoptime,
            SUM(CASE WHEN covered_until IS NULL OR starttime > covered_until
                     THEN 1 ELSE 0 END) OVER (
                PARTITION BY subject_id, hadm_id, episode_id
                ORDER BY starttime, stoptime, drug
                ROWS UNBOUNDED PRECEDING
            ) AS block_id
        FROM abx_coverage
    ),
    abx_block_spans AS (
        -- Every prescription in a block overlaps the cover running before it, so
        -- the block's union is exactly its first start to its last stop.
        SELECT
            subject_id,
            hadm_id,
            episode_id,
            MIN(starttime) AS block_start,
            MAX(stoptime) AS block_stop
        FROM abx_blocks
        GROUP BY subject_id, hadm_id, episode_id, block_id
    ),
    iv_antibiotic_episodes AS (
        -- Collapse the episode into start, stop, and the time treatment actually
        -- covered.
        --
        -- The duration is the union of the prescription intervals, not the span
        -- from first start to last stop. The span counted the gaps between
        -- courses as treatment, which is what let a patient clear the 72-hour
        -- rule below on the strength of a gap. Concurrent agents count once, not
        -- once each: two drugs running together for a day are one day of
        -- treatment. abx_start_time and abx_stop_time are the same values the
        -- span produced -- only the duration changes.
        SELECT
            subject_id,
            hadm_id,
            MIN(block_start) AS abx_start_time,
            MAX(block_stop) AS abx_stop_time,
            SUM(EXTRACT(EPOCH FROM (block_stop - block_start))) / 3600.0 AS abx_duration_hours
        FROM abx_block_spans
        GROUP BY subject_id, hadm_id, episode_id
    ),
    coupled_events AS (
        SELECT 
            b.subject_id,
            b.hadm_id,
            b.stay_id,
            b.gender,
            b.age,
            b.race,
            b.admission_type,
            b.first_careunit,
            b.icu_intime,
            b.icu_outtime,
            b.icu_los_days,
            b.hospital_expire_flag,
            b.dod,
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
               -- suspected_infection_time ties whenever several cultures on the
               -- admission couple to the same antibiotic episode. Left untied,
               -- the surviving row -- and with it the culture and antibiotic
               -- times that set the +/-48h window for every downstream stage --
               -- is whichever the engine happened to emit first. Rows tied on
               -- all four keys are identical in every column selected here.
               ROW_NUMBER() OVER (
                   PARTITION BY subject_id
                   ORDER BY suspected_infection_time ASC,
                            culture_time ASC,
                            abx_start_time ASC,
                            abx_stop_time ASC
               ) as infection_seq
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
    
    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! Confirmed infection cohort established in {elapsed:.2f} seconds.")
    print(f"    -> Total True Positive Infection Patients: {count}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
