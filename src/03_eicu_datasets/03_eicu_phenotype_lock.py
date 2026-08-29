"""
Exclude sepsis mimics and compute the Charlson Comorbidity Index for eICU.

The phenotype lock screens primary and major diagnoses for the same four mimic
categories as MIMIC-IV, using coded and textual diagnosis information. The
trauma criterion explicitly excludes organ-injury phrases (kidney, lung,
myocardial, hepatic, brain), because eICU's vocabulary spells acute kidney
injury as an "injury" and a plain substring match would remove septic acute
kidney injury from the validation cohort while leaving it in the development
cohort.

Comorbidity uses the enhanced ICD-9-CM and ICD-10 algorithms of Quan et al.
(2005), matching MIMIC-IV. eICU records a large share of diagnoses as free text
with no code, so a narrow textual fallback is consulted only for entries that
carry no usable code; entries with a code are scored from the code alone in both
databases. The fallback terms name each comorbidity explicitly so that acute
presentations are not counted as chronic disease.

Codes are matched at a comma boundary after whitespace is stripped, because the
icd9code column holds a comma-separated ICD-9 and ICD-10 pair and anchoring to
the start of the string would only ever reach the first of the two.

Reads:
    eicu_infection_cohort.parquet, data/raw/eicu-crd/2.0/diagnosis
Writes:
    eicu_sepsis_phenotype_cohort.parquet
    outputs/metrics/eicu_charlson_coverage.json

The coverage audit reports what share of diagnosis rows fall back to text and
the resulting index distribution, so the residual gap against MIMIC-IV can be
quantified rather than assumed.
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
            lower(diagnosisstring) LIKE '%trauma%' OR 
            lower(diagnosisstring) LIKE '%thermal burn%' OR 
            lower(diagnosisstring) LIKE '%chemical burn%' OR 
            lower(diagnosisstring) LIKE '%electrical burn%' OR 
            (
              lower(diagnosisstring) LIKE '%injury%' 
              AND lower(diagnosisstring) NOT LIKE '%kidney injury%' 
              AND lower(diagnosisstring) NOT LIKE '%renal injury%' 
              AND lower(diagnosisstring) NOT LIKE '%lung injury%'
              AND lower(diagnosisstring) NOT LIKE '%myocardial injury%'
              AND lower(diagnosisstring) NOT LIKE '%brain injury%'
              AND lower(diagnosisstring) NOT LIKE '%liver injury%'
            )
          )
    ),
    charlson_flags AS (
        -- Charlson: Quan 2005 coded mapping when the row carries a usable
        -- icd9code, and a tightened diagnosisstring fallback only when it does
        -- not. The fallback is deliberately narrower than the original text
        -- matching: '%tia%' matched 'essential hypertension' and '%renal
        -- failure%' matched 'acute renal failure', both of which inflated CCI.
        SELECT 
            patientunitstayid AS stay_id,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(410|412|I21|I22|I252)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%myocardial infarction%'))
                     THEN 1 ELSE 0 END) AS mi,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(39891|40201|40211|40291|40401|40403|40411|40413|40491|40493|425[4-9]|428|I099|I110|I130|I132|I255|I420|I42[5-9]|I43|I50|P290)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%congestive heart failure%'))
                     THEN 1 ELSE 0 END) AS chf,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(0930|4373|440|441|443[1-9]|4471|5571|5579|V434|I70|I71|I731|I738|I739|I771|I790|I792|K551|K558|K559|Z958|Z959)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%peripheral vascular%'))
                     THEN 1 ELSE 0 END) AS pvd,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(36234|43[0-8]|G45|G46|H340|I6[0-9])'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%stroke%' OR lower(diagnosisstring) LIKE '%transient ischemic%'))
                     THEN 1 ELSE 0 END) AS cevd,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(290|2941|3312|F0[0-3]|F051|G30|G311)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%dementia%'))
                     THEN 1 ELSE 0 END) AS dementia,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(4168|4169|49[0-9]|50[0-5]|5064|5081|5088|I278|I279|J4[0-7]|J6[0-7]|J684|J701|J703)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%copd%' OR lower(diagnosisstring) LIKE '%chronic obstructive%'))
                     THEN 1 ELSE 0 END) AS cpd,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(4465|710[0-4]|714[0-2]|7148|725|M05|M06|M315|M32|M33|M34|M351|M353|M360)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%rheumatoid%' OR lower(diagnosisstring) LIKE '%lupus%'))
                     THEN 1 ELSE 0 END) AS rheum,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(53[1-4]|K2[5-8])'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%peptic ulcer%'))
                     THEN 1 ELSE 0 END) AS pud,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(07022|07023|07032|07033|07044|07054|0706|0709|570|571|5733|5734|5738|5739|V427|B18|K70[0-3]|K709|K71[3-5]|K717|K73|K74|K760|K76[2-4]|K768|K769|Z944)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%cirrhosis%'))
                     THEN 1 ELSE 0 END) AS mild_liver,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(250[0-3]|2508|2509|E10[01689]|E11[01689]|E12[01689]|E13[01689]|E14[01689])'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%diabetes mellitus%'))
                     THEN 1 ELSE 0 END) AS diab,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(250[4-7]|E10[2-57]|E11[2-57]|E12[2-57]|E13[2-57]|E14[2-57])'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%diabetic%'))
                     THEN 1 ELSE 0 END) AS diab_comp,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(3341|342|343|344[0-6]|3449|G041|G114|G80[12]|G81|G82|G83[0-4]|G839)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%paraplegia%' OR lower(diagnosisstring) LIKE '%hemiplegia%'))
                     THEN 1 ELSE 0 END) AS paraplegia,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(40301|40311|40391|40402|40403|40412|40413|40492|40493|582|583[0-7]|585|586|5880|V420|V451|V56|I120|I131|N03[2-7]|N05[2-7]|N18|N19|N250|Z49[0-2]|Z940|Z992)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%chronic kidney disease%' OR lower(diagnosisstring) LIKE '%chronic renal failure%' OR lower(diagnosisstring) LIKE '%end stage renal%'))
                     THEN 1 ELSE 0 END) AS renal,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(14[0-9]|15[0-9]|16[0-9]|17[0-2]|17[4-9]|18[0-9]|19[0-5]|20[0-8]|2386|C0[0-9]|C1[0-9]|C2[0-6]|C3[0-4]|C3[7-9]|C4[013]|C4[5-9]|C5[0-9]|C6[0-9]|C7[0-6]|C8[1-58]|C9[0-7])'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%cancer%' OR lower(diagnosisstring) LIKE '%carcinoma%' OR lower(diagnosisstring) LIKE '%malignancy%'))
                     THEN 1 ELSE 0 END) AS cancer,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(456[0-2]|572[2-8]|I850|I859|I864|I982|K704|K711|K721|K729|K76[5-7])'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%hepatic failure%'))
                     THEN 1 ELSE 0 END) AS sev_liver,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(19[6-9]|C7[7-9]|C80)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%metastatic%'))
                     THEN 1 ELSE 0 END) AS mets,
            MAX(CASE WHEN (icd9code IS NOT NULL AND TRIM(icd9code) <> '' AND REGEXP_MATCHES(REPLACE(REPLACE(icd9code, '.', ''), ' ', ''), '(^|,)(04[2-4]|B2[0-2]|B24)'))
                          OR ((icd9code IS NULL OR TRIM(icd9code) = '') AND (lower(diagnosisstring) LIKE '%hiv%' OR lower(diagnosisstring) LIKE '%aids%'))
                     THEN 1 ELSE 0 END) AS hiv
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
    print("       - Mapping historical codes to 17 morbidity categories via Quan 2005 definition")
    print("       - Applying fallback text string extraction to recover absent ICD-9 fields")
    print("       - Applying prognostic weighting and appending to cohort")
    
    # Charlson coverage audit. Kept so the final run documents how much of the
    # eICU CCI comes from coded data versus the text fallback, and how far the
    # eICU distribution still sits from MIMIC. Both numbers belong in the
    # limitations.
    print("\n    [CHARLSON CODE COVERAGE AUDIT]")
    cov = con.execute(f"""
        WITH cohort AS (SELECT stay_id FROM '{infection_file}'),
        dx AS (
            SELECT d.patientunitstayid AS stay_id, d.icd9code
            FROM read_csv_auto('{EICU_DIR}/diagnosis.csv.gz', sample_size=-1) d
            INNER JOIN cohort c ON d.patientunitstayid = c.stay_id
        )
        SELECT
            COUNT(*) AS dx_rows,
            SUM(CASE WHEN icd9code IS NULL OR TRIM(icd9code) = '' THEN 1 ELSE 0 END) AS dx_rows_uncoded,
            COUNT(DISTINCT stay_id) AS stays_with_any_dx,
            COUNT(DISTINCT CASE WHEN icd9code IS NOT NULL AND TRIM(icd9code) <> ''
                                THEN stay_id END) AS stays_with_any_code
        FROM dx
    """).fetchone()

    dx_rows, dx_uncoded, stays_any_dx, stays_any_code = cov
    n_cohort = con.execute(f"SELECT COUNT(*) FROM '{infection_file}'").fetchone()[0]

    print(f"       - Cohort stays                       : {n_cohort:,}")
    print(f"       - Diagnosis rows for the cohort      : {dx_rows:,}")
    if dx_rows:
        print(f"       - Rows with no usable icd9code       : {dx_uncoded:,} "
              f"({dx_uncoded / dx_rows * 100:.1f}%)  <- these use the text fallback")
    print(f"       - Stays with >=1 diagnosis row       : {stays_any_dx:,}")
    if n_cohort:
        print(f"       - Stays with >=1 coded diagnosis     : {stays_any_code:,} "
              f"({stays_any_code / n_cohort * 100:.1f}% of cohort)")

    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    print(f"\n[+] Success! eICU Phenotype lock and Comorbidity scoring applied in {elapsed:.2f} seconds.")
    print(f"    -> Total Verified Sepsis Patients (Mimics Excluded): {count:,}")
    cci = con.execute(f"""
        SELECT
            MEDIAN(charlson_comorbidity_index) AS med,
            QUANTILE_CONT(charlson_comorbidity_index, 0.25) AS q25,
            QUANTILE_CONT(charlson_comorbidity_index, 0.75) AS q75,
            AVG(CASE WHEN charlson_comorbidity_index = 0 THEN 1.0 ELSE 0.0 END) * 100 AS pct_zero
        FROM '{out_file}'
    """).fetchone()
    import json as _json
    _metrics = BASE_DIR / "outputs" / "metrics"
    _metrics.mkdir(parents=True, exist_ok=True)
    with open(_metrics / "eicu_charlson_coverage.json", "w") as _f:
        _json.dump({"cohort_stays": int(n_cohort),
                    "diagnosis_rows": int(dx_rows),
                    "rows_without_code": int(dx_uncoded),
                    "pct_rows_using_text_fallback": round(dx_uncoded / dx_rows * 100, 2) if dx_rows else None,
                    "stays_with_any_diagnosis": int(stays_any_dx),
                    "stays_with_any_code": int(stays_any_code),
                    "cci_median": float(cci[0]), "cci_q25": float(cci[1]),
                    "cci_q75": float(cci[2]), "cci_pct_zero": round(float(cci[3]), 2)}, _f, indent=4)

    print("\n    [CHARLSON DISTRIBUTION] compare against the MIMIC cohort")
    print(f"       - CCI median (IQR) : {cci[0]:.1f} ({cci[1]:.1f} - {cci[2]:.1f})")
    print(f"       - CCI == 0         : {cci[3]:.1f}% of stays")
    print("       - Residual gap versus MIMIC is a real limitation, not a bug to")
    print("         chase with looser text patterns. Report it.")

    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
