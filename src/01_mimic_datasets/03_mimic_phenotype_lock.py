"""
Exclude sepsis mimics and compute the Charlson Comorbidity Index.

Two independent jobs against the same diagnosis table. The phenotype lock drops
admissions whose primary or secondary diagnosis (sequence number <= 2) is a
condition that reproduces sepsis physiology without infection: acute myocardial
infarction, pulmonary embolism, acute pancreatitis, and trauma or burns. The
comorbidity index is computed over all recorded diagnoses for the admission
using the enhanced ICD-9-CM and ICD-10 algorithms of Quan et al. (2005) with the
standard category weights, matching the definition applied to eICU so the two
cohorts remain comparable.

ICD-9 trauma is matched numerically on the three-digit category (800-959), which
subsumes the burn block; TRY_CAST yields NULL for V and E codes, excluding those
chapters as intended.

Reads:
    mimic_infection_cohort.parquet
    data/raw/mimiciv/3.1/hosp/diagnoses_icd
Writes:
    mimic_sepsis_phenotype_cohort.parquet
    outputs/metrics/mimic_phenotype_lock_attrition.json

The attrition report gives the cohort entering the lock and the number matched
by each rule, for the cohort flow diagram. Rules overlap, so counts do not sum.
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
    print("Executing MIMIC-IV sepsis phenotype lock & comorbidity pipeline...")
    start_time = time.time()
    
    infection_file = PROCESSED_DIR / "mimic_infection_cohort.parquet"
    out_file = PROCESSED_DIR / "mimic_sepsis_phenotype_cohort.parquet"
    
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
            -- Numeric comparison on the three-digit ICD-9 category. A VARCHAR
            -- range would sort short codes unpredictably ('85' sorts after
            -- '800'). TRY_CAST yields NULL for V and E codes, excluding those
            -- chapters. 800-959 subsumes the 940-949 burn block.
            (icd_version = 9 AND
                TRY_CAST(SUBSTRING(TRIM(icd_code), 1, 3) AS INTEGER) BETWEEN 800 AND 959
            ) OR
            (icd_version = 10 AND (
                TRIM(icd_code) LIKE 'S%' OR -- Injuries, poisoning
                -- T codes up to T32 cover burns and frostbite
                (TRIM(icd_code) LIKE 'T%' AND
                 TRY_CAST(SUBSTRING(TRIM(icd_code), 2, 2) AS INTEGER) <= 32)
            ))
          )
    ),
    charlson_flags AS (
        -- Calculate Charlson Comorbidity components per admission using Quan 2005 mappings
        SELECT 
            hadm_id,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(410|412)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(I21|I22|I252)')) THEN 1 ELSE 0 END) AS mi,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(39891|40201|40211|40291|40401|40403|40411|40413|40491|40493|425[4-9]|428)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(I099|I110|I130|I132|I255|I420|I42[5-9]|I43|I50|P290)')) THEN 1 ELSE 0 END) AS chf,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(0930|4373|440|441|443[1-9]|4471|5571|5579|V434)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(I70|I71|I731|I738|I739|I771|I790|I792|K551|K558|K559|Z958|Z959)')) THEN 1 ELSE 0 END) AS pvd,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(36234|43[0-8])')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(G45|G46|H340|I6[0-9])')) THEN 1 ELSE 0 END) AS cevd,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(290|2941|3312)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(F0[0-3]|F051|G30|G311)')) THEN 1 ELSE 0 END) AS dementia,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(4168|4169|49[0-9]|50[0-5]|5064|5081|5088)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(I278|I279|J4[0-7]|J6[0-7]|J684|J701|J703)')) THEN 1 ELSE 0 END) AS cpd,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(4465|710[0-4]|714[0-2]|7148|725)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(M05|M06|M315|M32|M33|M34|M351|M353|M360)')) THEN 1 ELSE 0 END) AS rheum,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(53[1-4])')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(K2[5-8])')) THEN 1 ELSE 0 END) AS pud,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(07022|07023|07032|07033|07044|07054|0706|0709|570|571|5733|5734|5738|5739|V427)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(B18|K70[0-3]|K709|K71[3-5]|K717|K73|K74|K760|K76[2-4]|K768|K769|Z944)')) THEN 1 ELSE 0 END) AS mild_liver,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(250[0-3]|2508|2509)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(E10[01689]|E11[01689]|E12[01689]|E13[01689]|E14[01689])')) THEN 1 ELSE 0 END) AS diab,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(250[4-7])')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(E10[2-57]|E11[2-57]|E12[2-57]|E13[2-57]|E14[2-57])')) THEN 1 ELSE 0 END) AS diab_comp,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(3341|342|343|344[0-6]|3449)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(G041|G114|G80[12]|G81|G82|G83[0-4]|G839)')) THEN 1 ELSE 0 END) AS paraplegia,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(40301|40311|40391|40402|40403|40412|40413|40492|40493|582|583[0-7]|585|586|5880|V420|V451|V56)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(I120|I131|N03[2-7]|N05[2-7]|N18|N19|N250|Z49[0-2]|Z940|Z992)')) THEN 1 ELSE 0 END) AS renal,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(14[0-9]|15[0-9]|16[0-9]|17[0-2]|17[4-9]|18[0-9]|19[0-5]|20[0-8]|2386)')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(C0[0-9]|C1[0-9]|C2[0-6]|C3[0-4]|C3[7-9]|C4[013]|C4[5-9]|C5[0-9]|C6[0-9]|C7[0-6]|C8[1-58]|C9[0-7])')) THEN 1 ELSE 0 END) AS cancer,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(456[0-2]|572[2-8])')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(I850|I859|I864|I982|K704|K711|K721|K729|K76[5-7])')) THEN 1 ELSE 0 END) AS sev_liver,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(19[6-9])')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(C7[7-9]|C80)')) THEN 1 ELSE 0 END) AS mets,
            MAX(CASE WHEN (icd_version = 9 AND REGEXP_MATCHES(TRIM(icd_code), '^(04[2-4])')) OR 
                          (icd_version = 10 AND REGEXP_MATCHES(TRIM(icd_code), '^(B2[0-2]|B24)')) THEN 1 ELSE 0 END) AS hiv
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
    print("    - Mapping historical ICD-9/10 codes to 17 morbidity categories via Quan 2005 definition")
    print("    - Applying prognostic weighting and appending to cohort")
    
    # --- Per-Rule Attrition, for the Cohort Flow Diagram ------------------
    rules = {
        "Acute myocardial infarction":
            "(icd_version = 9 AND TRIM(icd_code) LIKE '410%') OR "
            "(icd_version = 10 AND (TRIM(icd_code) LIKE 'I21%' OR TRIM(icd_code) LIKE 'I22%'))",
        "Pulmonary embolism":
            "(icd_version = 9 AND TRIM(icd_code) LIKE '4151%') OR "
            "(icd_version = 10 AND TRIM(icd_code) LIKE 'I26%')",
        "Acute pancreatitis":
            "(icd_version = 9 AND TRIM(icd_code) LIKE '5770%') OR "
            "(icd_version = 10 AND TRIM(icd_code) LIKE 'K85%')",
        "Trauma and burns":
            "(icd_version = 9 AND TRY_CAST(SUBSTRING(TRIM(icd_code), 1, 3) AS INTEGER) "
            "BETWEEN 800 AND 959) OR "
            "(icd_version = 10 AND (TRIM(icd_code) LIKE 'S%' OR "
            "(TRIM(icd_code) LIKE 'T%' AND "
            "TRY_CAST(SUBSTRING(TRIM(icd_code), 2, 2) AS INTEGER) <= 32)))",
    }

    n_in = con.execute(f"SELECT COUNT(*) FROM '{infection_file}'").fetchone()[0]
    attrition_rows = []
    print("\n    [PHENOTYPE LOCK ATTRITION]")
    print(f"       - Confirmed-infection cohort entering the lock : {n_in:,}")
    print("       - Patients matched by each rule (rules overlap, so these do not sum):")
    for label, predicate in rules.items():
        n_hit = con.execute(f"""
            SELECT COUNT(*) FROM '{infection_file}' i
            WHERE EXISTS (
                SELECT 1 FROM read_csv_auto('{MIMIC_DIR}/hosp/diagnoses_icd.csv.gz') d
                WHERE d.subject_id = i.subject_id AND d.hadm_id = i.hadm_id
                  AND d.seq_num <= 2 AND ({predicate})
            )
        """).fetchone()[0]
        pct = (n_hit / n_in * 100) if n_in else 0.0
        attrition_rows.append({"rule": label, "n_matched": n_hit, "pct_of_infection_cohort": round(pct, 2)})
        print(f"           {label:<32}: {n_hit:>7,} ({pct:5.2f}%)")

    con.execute(f"COPY ({query}) TO '{out_file}' (FORMAT PARQUET)")
    
    count = con.execute(f"SELECT COUNT(*) FROM '{out_file}'").fetchone()[0]
    elapsed = time.time() - start_time
    
    # Persist for the cohort flow diagram
    import json as _json
    _metrics = BASE_DIR / "outputs" / "metrics"
    _metrics.mkdir(parents=True, exist_ok=True)
    with open(_metrics / "mimic_phenotype_lock_attrition.json", "w") as _f:
        _json.dump({"infection_cohort_in": int(n_in),
                    "phenotype_cohort_out": int(count),
                    "removed_total": int(n_in - count),
                    "rules_overlap": True,
                    "per_rule": attrition_rows}, _f, indent=4)

    print(f"\n[+] Success! Phenotype lock and Comorbidity scoring applied in {elapsed:.2f} seconds.")
    print(f"    -> Total Verified Sepsis Patients (Mimics Excluded): {count}")
    print(f"    -> Output saved successfully to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()
