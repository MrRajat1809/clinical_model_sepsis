"""
03_eicu_phenotype_lock.py

Phase 9: External Validation (eICU Sepsis Phenotype Lock)
Applies a Sepsis Mimic Exclusion filter to the eICU confirmed infection cohort.
Excludes patients whose primary or major diagnosis is a classic sepsis mimic
(e.g., Acute Myocardial Infarction, Pulmonary Embolism, Acute Pancreatitis, Trauma/Burns).
This strictly avoids the surgical/mimic inclusion bias found in poorly filtered eICU studies.

[FIX]: Calculates the Charlson Comorbidity Index (CCI) by parsing both eICU's comma-separated 
       `icd9code` column and the text-based `diagnosisstring` to ensure maximum capture of 
       historical morbidity for the static feature vector.
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
    print("[*] Executing eICU sepsis phenotype lock & comorbidity pipeline...")
    start_time = time.time()
    
    infection_file = PROCESSED_DIR / "eicu_infection_cohort.parquet"
    out_file = PROCESSED_DIR / "eicu_sepsis_phenotype_cohort.parquet"
    
    if not infection_file.exists():
        print(f"[ERROR] Infection cohort not found at: {infection_file}")
        return
        
    print("\n    -> Initializing in-memory DuckDB...")
    con = duckdb.connect(database=':memory:')
    
    query = f"""
    WITH sepsis_mimics AS (
        SELECT DISTINCT patientunitstayid AS stay_id
        FROM read_csv_auto('{EICU_DIR}/diagnosis.csv.gz', sample_size=-1)
        WHERE diagnosispriority IN ('Primary', 'Major') -- Target primary and major clinical drivers
          AND (
            -- 1. Acute Myocardial Infarction (AMI) / Cardiogenic Shock
            icd9code LIKE '%410%' OR icd9code LIKE '%I21%' OR icd9code LIKE '%I22%' OR
            lower(diagnosisstring) LIKE '%myocardial infarction%' OR lower(diagnosisstring) LIKE '%cardiogenic shock%' OR
            
            -- 2. Pulmonary Embolism (PE)
            icd9code LIKE '%415.1%' OR icd9code LIKE '%I26%' OR
            lower(diagnosisstring) LIKE '%pulmonary embolism%' OR
            
            -- 3. Acute Pancreatitis
            icd9code LIKE '%577.0%' OR icd9code LIKE '%K85%' OR
            lower(diagnosisstring) LIKE '%pancreatitis%' OR
            
            -- 4. Trauma and Burns
            lower(diagnosisstring) LIKE '%trauma%' OR lower(diagnosisstring) LIKE '%burn %' OR 
            lower(diagnosisstring) LIKE '% burn%' OR lower(diagnosisstring) LIKE '%injury%'
          )
    ),
    charlson_flags AS (
        -- Calculate Charlson Comorbidity components per stay using both ICD and String matching
        SELECT 
            patientunitstayid AS stay_id,
            MAX(CASE WHEN icd9code LIKE '%410%' OR icd9code LIKE '%I21%' OR lower(diagnosisstring) LIKE '%myocardial infarction%' THEN 1 ELSE 0 END) AS mi,
            MAX(CASE WHEN icd9code LIKE '%428%' OR icd9code LIKE '%I50%' OR lower(diagnosisstring) LIKE '%congestive heart failure%' THEN 1 ELSE 0 END) AS chf,
            MAX(CASE WHEN icd9code LIKE '%443.9%' OR lower(diagnosisstring) LIKE '%peripheral vascular%' THEN 1 ELSE 0 END) AS pvd,
            MAX(CASE WHEN icd9code LIKE '%430%' OR icd9code LIKE '%I60%' OR lower(diagnosisstring) LIKE '%stroke%' OR lower(diagnosisstring) LIKE '%tia%' THEN 1 ELSE 0 END) AS cevd,
            MAX(CASE WHEN icd9code LIKE '%290%' OR icd9code LIKE '%F03%' OR lower(diagnosisstring) LIKE '%dementia%' THEN 1 ELSE 0 END) AS dementia,
            MAX(CASE WHEN icd9code LIKE '%490%' OR icd9code LIKE '%J44%' OR lower(diagnosisstring) LIKE '%copd%' OR lower(diagnosisstring) LIKE '%chronic obstructive%' THEN 1 ELSE 0 END) AS cpd,
            MAX(CASE WHEN icd9code LIKE '%710%' OR icd9code LIKE '%M32%' OR lower(diagnosisstring) LIKE '%rheumatoid%' OR lower(diagnosisstring) LIKE '%lupus%' THEN 1 ELSE 0 END) AS rheum,
            MAX(CASE WHEN icd9code LIKE '%531%' OR icd9code LIKE '%K25%' OR lower(diagnosisstring) LIKE '%peptic ulcer%' THEN 1 ELSE 0 END) AS pud,
            MAX(CASE WHEN icd9code LIKE '%571.2%' OR icd9code LIKE '%K70.3%' OR lower(diagnosisstring) LIKE '%cirrhosis%' THEN 1 ELSE 0 END) AS mild_liver,
            MAX(CASE WHEN icd9code LIKE '%250.0%' OR icd9code LIKE '%E11.9%' OR lower(diagnosisstring) LIKE '%diabetes%' THEN 1 ELSE 0 END) AS diab,
            MAX(CASE WHEN icd9code LIKE '%250.4%' OR icd9code LIKE '%E11.2%' OR lower(diagnosisstring) LIKE '%diabetic%' THEN 1 ELSE 0 END) AS diab_comp,
            MAX(CASE WHEN icd9code LIKE '%342%' OR icd9code LIKE '%G81%' OR lower(diagnosisstring) LIKE '%paraplegia%' OR lower(diagnosisstring) LIKE '%hemiplegia%' THEN 1 ELSE 0 END) AS paraplegia,
            MAX(CASE WHEN icd9code LIKE '%585%' OR icd9code LIKE '%N18%' OR lower(diagnosisstring) LIKE '%chronic kidney disease%' OR lower(diagnosisstring) LIKE '%renal failure%' THEN 1 ELSE 0 END) AS renal,
            MAX(CASE WHEN icd9code LIKE '%140%' OR icd9code LIKE '%C00%' OR lower(diagnosisstring) LIKE '%cancer%' OR lower(diagnosisstring) LIKE '%carcinoma%' OR lower(diagnosisstring) LIKE '%malignancy%' THEN 1 ELSE 0 END) AS cancer,
            MAX(CASE WHEN icd9code LIKE '%572.2%' OR icd9code LIKE '%K72.1%' OR lower(diagnosisstring) LIKE '%hepatic failure%' THEN 1 ELSE 0 END) AS sev_liver,
            MAX(CASE WHEN icd9code LIKE '%196%' OR icd9code LIKE '%C77%' OR lower(diagnosisstring) LIKE '%metastatic%' THEN 1 ELSE 0 END) AS mets,
            MAX(CASE WHEN icd9code LIKE '%042%' OR icd9code LIKE '%B20%' OR lower(diagnosisstring) LIKE '%hiv%' OR lower(diagnosisstring) LIKE '%aids%' THEN 1 ELSE 0 END) AS hiv
        FROM read_csv_auto('{EICU_DIR}/diagnosis.csv.gz', sample_size=-1)
        GROUP BY patientunitstayid
    ),
    charlson_scores AS (
        -- Sum the standard weights for the Charlson Comorbidity Index
        SELECT 
            stay_id,
            (mi * 1 + chf * 1 + pvd * 1 + cevd * 1 + dementia * 1 + cpd * 1 + rheum * 1 + pud * 1 + mild_liver * 1 + 
             diab * 1 + diab_comp * 2 + paraplegia * 2 + renal * 2 + cancer * 2 + sev_liver * 3 + mets * 6 + hiv * 6) AS charlson_comorbidity_index
        FROM charlson_flags
    )
    SELECT 
        i.*,
        COALESCE(c.charlson_comorbidity_index, 0) AS charlson_comorbidity_index
    FROM '{infection_file}' i
    LEFT JOIN sepsis_mimics m ON i.stay_id = m.stay_id
    LEFT JOIN charlson_scores c ON i.stay_id = c.stay_id
    WHERE m.stay_id IS NULL -- Keep ONLY patients who do NOT match the mimic list
    """
    
    print("    -> Executing hybrid ICD & text-based exclusion for classic sepsis mimics...")
    print("       - Evaluating primary and major diagnoses")
    print("       - Excluding Acute Myocardial Infarction (AMI), PE, Pancreatitis, Trauma/Burns")
    print("    -> Generating Charlson Comorbidity Index (CCI)...")
    print("       - Mapping historical codes/strings to 17 morbidity categories")
    print("       - Applying prognostic weighting and appending to cohort")
    
    # Execute and write directly to Parquet
    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    # Verify the output
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! eICU Phenotype lock and Comorbidity scoring applied in {elapsed:.2f} seconds.")
    print(f"    -> Total Verified Sepsis Patients (Mimics Excluded): {count:,}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()