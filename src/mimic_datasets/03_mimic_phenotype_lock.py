"""
03_mimic_phenotype_lock.py

Applies a Sepsis Mimic Exclusion filter to the confirmed infection cohort.
Excludes patients whose primary or secondary diagnosis is a classic sepsis mimic
(e.g., Acute Myocardial Infarction, Pulmonary Embolism, Acute Pancreatitis, Trauma/Burns).

[FIX APPLIED]: Added a Charlson Comorbidity Index (CCI) calculator to extract baseline 
chronic illness burden directly from the diagnoses_icd table. The score is appended to 
the cohort for downstream static feature vector generation.
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
    print("Executing MIMIC-IV sepsis phenotype lock & comorbidity pipeline...")
    start_time = time.time()
    
    infection_file = PROCESSED_DIR / "infection_cohort.parquet"
    out_file = PROCESSED_DIR / "sepsis_phenotype_cohort.parquet"
    
    if not infection_file.exists():
        print(f"[ERROR] Infection cohort not found at: {infection_file}")
        return
        
    print("\n[*] Initializing in-memory DuckDB...")
    con = duckdb.connect(database=':memory:')
    
    query = f"""
    WITH sepsis_mimics_icd AS (
        SELECT DISTINCT subject_id, hadm_id
        FROM read_csv_auto('{MIMIC_DIR}/hosp/diagnoses_icd.csv.gz')
        WHERE seq_num <= 2 -- Target the primary and secondary diagnoses
          AND (
            -- 1. Acute Myocardial Infarction (AMI) / Cardiogenic Shock
            (icd_version = 9 AND icd_code LIKE '410%') OR
            (icd_version = 10 AND (icd_code LIKE 'I21%' OR icd_code LIKE 'I22%')) OR
            
            -- 2. Pulmonary Embolism (PE)
            (icd_version = 9 AND icd_code LIKE '4151%') OR
            (icd_version = 10 AND icd_code LIKE 'I26%') OR
            
            -- 3. Acute Pancreatitis
            (icd_version = 9 AND icd_code LIKE '5770%') OR
            (icd_version = 10 AND icd_code LIKE 'K85%') OR
            
            -- 4. Trauma and Burns
            (icd_version = 9 AND (
                (icd_code >= '800' AND icd_code < '960') OR -- General Injury/Trauma
                (icd_code >= '940' AND icd_code < '950')    -- Burns
            )) OR
            (icd_version = 10 AND (
                icd_code LIKE 'S%' OR -- Injuries, poisoning
                -- T codes up to T32 cover burns and frostbite
                (icd_code LIKE 'T%' AND CAST(SUBSTRING(icd_code, 2, 2) AS INT) <= 32)
            ))
          )
    ),
    charlson_flags AS (
        -- Calculate Charlson Comorbidity components per admission
        SELECT 
            hadm_id,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '410%') OR (icd_version = 10 AND icd_code LIKE 'I21%') THEN 1 ELSE 0 END) AS mi,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '428%') OR (icd_version = 10 AND icd_code LIKE 'I50%') THEN 1 ELSE 0 END) AS chf,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '4439%') OR (icd_version = 10 AND icd_code LIKE 'I739%') THEN 1 ELSE 0 END) AS pvd,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '430%') OR (icd_version = 10 AND icd_code LIKE 'I60%') THEN 1 ELSE 0 END) AS cevd,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '290%') OR (icd_version = 10 AND icd_code LIKE 'F03%') THEN 1 ELSE 0 END) AS dementia,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '490%') OR (icd_version = 10 AND icd_code LIKE 'J44%') THEN 1 ELSE 0 END) AS cpd,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '710%') OR (icd_version = 10 AND icd_code LIKE 'M32%') THEN 1 ELSE 0 END) AS rheum,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '531%') OR (icd_version = 10 AND icd_code LIKE 'K25%') THEN 1 ELSE 0 END) AS pud,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '5712%') OR (icd_version = 10 AND icd_code LIKE 'K703%') THEN 1 ELSE 0 END) AS mild_liver,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '2500%') OR (icd_version = 10 AND icd_code LIKE 'E119%') THEN 1 ELSE 0 END) AS diab,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '2504%') OR (icd_version = 10 AND icd_code LIKE 'E112%') THEN 1 ELSE 0 END) AS diab_comp,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '342%') OR (icd_version = 10 AND icd_code LIKE 'G81%') THEN 1 ELSE 0 END) AS paraplegia,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '585%') OR (icd_version = 10 AND icd_code LIKE 'N18%') THEN 1 ELSE 0 END) AS renal,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '140%') OR (icd_version = 10 AND icd_code LIKE 'C00%') THEN 1 ELSE 0 END) AS cancer,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '5722%') OR (icd_version = 10 AND icd_code LIKE 'K721%') THEN 1 ELSE 0 END) AS sev_liver,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '196%') OR (icd_version = 10 AND icd_code LIKE 'C77%') THEN 1 ELSE 0 END) AS mets,
            MAX(CASE WHEN (icd_version = 9 AND icd_code LIKE '042%') OR (icd_version = 10 AND icd_code LIKE 'B20%') THEN 1 ELSE 0 END) AS hiv
        FROM read_csv_auto('{MIMIC_DIR}/hosp/diagnoses_icd.csv.gz')
        GROUP BY hadm_id
    ),
    charlson_scores AS (
        -- Sum the standard weights for the Charlson Comorbidity Index
        SELECT 
            hadm_id,
            (mi * 1 + chf * 1 + pvd * 1 + cevd * 1 + dementia * 1 + cpd * 1 + rheum * 1 + pud * 1 + mild_liver * 1 + 
             diab * 1 + diab_comp * 2 + paraplegia * 2 + renal * 2 + cancer * 2 + sev_liver * 3 + mets * 6 + hiv * 6) AS charlson_comorbidity_index
        FROM charlson_flags
    )
    SELECT 
        i.*,
        COALESCE(c.charlson_comorbidity_index, 0) AS charlson_comorbidity_index
    FROM '{infection_file}' i
    LEFT JOIN sepsis_mimics_icd m ON i.subject_id = m.subject_id AND i.hadm_id = m.hadm_id
    LEFT JOIN charlson_scores c ON i.hadm_id = c.hadm_id
    WHERE m.subject_id IS NULL -- Keep ONLY patients who do NOT match the mimic list
    """
    
    print("[*] Executing ICD-based exclusion for classic sepsis mimics...")
    print("    - Evaluating primary and secondary diagnoses (seq_num <= 2)")
    print("    - Excluding Acute Myocardial Infarction (AMI), PE, Pancreatitis, Trauma/Burns")
    print("[*] Generating Charlson Comorbidity Index (CCI)...")
    print("    - Mapping historical ICD-9/10 codes to 17 morbidity categories")
    print("    - Applying prognostic weighting and appending to cohort")
    
    # Execute and write directly to Parquet
    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    # Verify the output
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! Phenotype lock and Comorbidity scoring applied in {elapsed:.2f} seconds.")
    print(f"    -> Total Verified Sepsis Patients (Mimics Excluded): {count}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()