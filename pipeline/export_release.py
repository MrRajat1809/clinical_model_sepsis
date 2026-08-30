"""
export_release.py

Builds the ClinicalTensorSepsis release package from artifacts the pipeline has
already written. Read-only with respect to the pipeline: nothing under src/,
data/processed/ or outputs/ is modified.

Format decisions follow PhysioNet's author guidelines:
  gzip-compressed CSV, matching what MIMIC-IV and eICU-CRD themselves ship
  wide layout, one observation per row, per the tidy-data requirement
  strings not label codes, read from the cohort tables rather than the
    static arrays, whose category codes are assigned per database and are
    meaningless across them
  integer offsets in minutes, never absolute dates
  no pickle-format file is ever selected

    python pipeline/export_release.py
    python pipeline/export_release.py --outdir release
"""

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
PROC_M = BASE / "data" / "processed" / "mimiciv"
PROC_E = BASE / "data" / "processed" / "eicu"
PROC_A = BASE / "data" / "processed" / "atlas"
METRICS = BASE / "outputs" / "metrics"
FEATURES = BASE / "outputs" / "features"
MODELS = BASE / "outputs" / "models"

MIMIC = "MIMIC-IV"
EICU = "eICU-CRD"

# Documentation for the thirty hourly variables. Held here rather than derived,
# because unit and provenance are knowledge in the extraction scripts, not values
# in any artifact.
VARIABLE_DOC = {
    "hr":            ("beats/min",   "chartevents 220045",              "vitalPeriodic heartrate",        "max"),
    "map":           ("mmHg",        "chartevents 220181/220052",       "vitalPeriodic/Aperiodic mean",   "min"),
    "rr":            ("breaths/min", "chartevents 220210",              "vitalPeriodic respiration",      "max"),
    "temp_c":        ("degC",        "chartevents 223762/223761",       "vitalPeriodic temperature",      "mean"),
    "spo2":          ("percent",     "chartevents 220277",              "vitalPeriodic sao2",             "min"),
    "gcs_eye":       ("1-4",         "chartevents 220739",              "nurseCharting eye",              "min"),
    "gcs_verbal":    ("1-5",         "chartevents 223900",              "nurseCharting verbal",           "min"),
    "gcs_motor":     ("1-6",         "chartevents 223901",              "nurseCharting motor",            "min"),
    "pao2":          ("mmHg",        "labevents 50821",                 "lab pao2",                       "max"),
    "fio2":          ("fraction",    "chartevents 223835",              "respiratoryCharting fio2",       "max"),
    "pf_ratio":      ("mmHg",        "engineered pao2/fio2",            "engineered pao2/fio2",           "engineered"),
    "paco2":         ("mmHg",        "labevents 50818",                 "lab paco2",                      "max"),
    "lactate":       ("mmol/L",      "labevents 50813",                 "lab lactate",                    "max"),
    "creatinine":    ("mg/dL",       "labevents 50912",                 "lab creatinine",                 "max"),
    "bun":           ("mg/dL",       "labevents 51006",                 "lab bun",                        "max"),
    "bilirubin":     ("mg/dL",       "labevents 50885",                 "lab total bilirubin",            "max"),
    "platelets":     ("K/uL",        "labevents 51265",                 "lab platelets x 1000",           "min"),
    "wbc":           ("K/uL",        "labevents 51301/51300",           "lab wbc x 1000",                 "max"),
    "hemoglobin":    ("g/dL",        "labevents 51222",                 "lab hgb",                        "min"),
    "ph":            ("pH units",    "labevents 50820",                 "lab ph",                         "min"),
    "pt":            ("seconds",     "labevents 51274",                 "lab pt",                         "max"),
    "aptt":          ("seconds",     "labevents 51275",                 "lab ptt",                        "max"),
    "albumin":       ("g/dL",        "labevents 50862",                 "lab albumin",                    "min"),
    "potassium":     ("mEq/L",       "labevents 50971/50822",           "lab potassium",                  "mean"),
    "sodium":        ("mEq/L",       "labevents 50983/50824",           "lab sodium",                     "mean"),
    "glucose":       ("mg/dL",       "labevents 50931/50809",           "lab glucose",                    "max"),
    "chloride":      ("mEq/L",       "labevents 50902/50806",           "lab chloride",                   "mean"),
    "urine_output":  ("mL",          "outputevents urine itemids",      "intakeOutput urine cellpaths",   "sum"),
    "neq":           ("mcg/kg/min",  "engineered from inputevents",     "engineered from infusionDrug",   "engineered"),
    "vent":          ("0/1",         "procedureevents + chartevents",   "treatment strings",              "max"),
}

COHORT_DOC = {
    "source_db":                  "Source database for this row",
    "stay_id":                    "Stay identifier in the source database; MIMIC-IV icustays.stay_id or eICU-CRD patientunitstayid",
    "subject_id":                 "Patient identifier in the source database",
    "hadm_id":                    "Hospital admission identifier; MIMIC-IV only, blank for eICU-CRD",
    "age":                        "Age in years at admission; values above 89 are capped per source de-identification",
    "sex":                        "Recorded sex, M or F",
    "race":                       "Recorded race or ethnicity, verbatim from the source; vocabularies differ between databases",
    "first_careunit":             "Type of intensive care unit at admission, verbatim from the source",
    "icu_los_days":               "Intensive care length of stay in days",
    "sit_offset_min":             "Suspected infection time, minutes from intensive care admission",
    "sepsis_onset_offset_min":    "Sepsis onset, minutes from intensive care admission; the origin of the hourly grid",
    "baseline_sofa":              "Total SOFA score in the baseline window, 48 h before suspected infection up to that time",
    "baseline_pf_ratio":          "PaO2/FiO2 ratio in the baseline window",
    "charlson_comorbidity_index": "Charlson index, Quan 2005 algorithm. See Usage Notes before treating as pre-admission burden",
    "hospital_expire_flag":       "In-hospital mortality, 1 if the patient died during the admission",
}

missing_inputs: list[str] = []


def need(path: Path):
    if not path.exists():
        missing_inputs.append(str(path.relative_to(BASE)))
        return None
    return path


def load_npy(path: Path, **kw):
    p = need(path)
    return None if p is None else np.load(p, **kw)


def first_col(df: pd.DataFrame, *names):
    """Return the first of `names` present in df, else a column of NA."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([pd.NA] * len(df), index=df.index)


def minutes_from(series, origin):
    """Offset in whole minutes; accepts datetimes or numeric offsets."""
    if series is None:
        return pd.Series(dtype="Float64")
    s = pd.Series(series)
    if pd.api.types.is_datetime64_any_dtype(s):
        return ((s - pd.Series(origin)).dt.total_seconds() / 60.0).round()
    return pd.to_numeric(s, errors="coerce").round()


# ------------------------------------------------------------------ tier 1
def build_cohort() -> pd.DataFrame | None:
    fm, fe = need(PROC_M / "mimic_final_sepsis3_cohort.parquet"), need(PROC_E / "eicu_final_sepsis3_cohort.parquet")
    if fm is None or fe is None:
        return None
    m, e = pd.read_parquet(fm), pd.read_parquet(fe)

    intime = pd.to_datetime(first_col(m, "icu_intime"))
    mm = pd.DataFrame({
        "source_db": MIMIC,
        "stay_id": m["stay_id"],
        "subject_id": first_col(m, "subject_id"),
        "hadm_id": first_col(m, "hadm_id"),
        "age": first_col(m, "age"),
        "sex": first_col(m, "gender"),
        "race": first_col(m, "race"),
        "first_careunit": first_col(m, "first_careunit"),
        "icu_los_days": first_col(m, "icu_los_days"),
        "sit_offset_min": minutes_from(pd.to_datetime(first_col(m, "suspected_infection_time")), intime),
        "sepsis_onset_offset_min": minutes_from(pd.to_datetime(first_col(m, "sepsis_onset_time")), intime),
        "baseline_sofa": first_col(m, "baseline_sofa"),
        "baseline_pf_ratio": first_col(m, "baseline_pf_ratio"),
        "charlson_comorbidity_index": first_col(m, "charlson_comorbidity_index"),
        "hospital_expire_flag": first_col(m, "hospital_expire_flag"),
    })

    ee = pd.DataFrame({
        "source_db": EICU,
        "stay_id": e["stay_id"],
        "subject_id": first_col(e, "subject_id"),
        "hadm_id": pd.NA,
        "age": first_col(e, "age"),
        "sex": first_col(e, "gender"),
        "race": first_col(e, "race"),
        "first_careunit": first_col(e, "first_careunit"),
        "icu_los_days": first_col(e, "icu_los_days"),
        "sit_offset_min": minutes_from(first_col(e, "sit_offset"), None),
        "sepsis_onset_offset_min": minutes_from(first_col(e, "sepsis_onset_offset"), None),
        "baseline_sofa": first_col(e, "baseline_sofa"),
        "baseline_pf_ratio": first_col(e, "baseline_pf_ratio"),
        "charlson_comorbidity_index": first_col(e, "charlson_comorbidity_index"),
        "hospital_expire_flag": first_col(e, "hospital_expire_flag"),
    })

    out = pd.concat([mm, ee], ignore_index=True)
    out["sex"] = out["sex"].astype(str).str.upper().str[0].replace({"N": pd.NA})
    return out


# ------------------------------------------------------------------ tier 2
def tensor_to_wide(tensor, stay_ids, feat_names, source) -> pd.DataFrame:
    n, hours, k = tensor.shape
    flat = tensor.reshape(n * hours, k)
    df = pd.DataFrame(flat, columns=[str(f) for f in feat_names])
    df.insert(0, "hour", np.tile(np.arange(hours), n))
    df.insert(0, "stay_id", np.repeat(np.asarray(stay_ids), hours))
    df.insert(0, "source_db", source)
    return df


def build_hourly(kind: str) -> pd.DataFrame | None:
    """kind is 'raw' (observed, blanks preserved) or 'imputed'."""
    stem = "sepsis_tensor_raw" if kind == "raw" else "sepsis_imputed_tensor"
    tm = load_npy(PROC_M / f"mimic_{stem}.npy")
    te = load_npy(PROC_E / f"eicu_{stem}.npy")
    im = load_npy(PROC_M / "mimic_sepsis_tensor_stay_ids.npy", allow_pickle=True)
    ie = load_npy(PROC_E / "eicu_sepsis_tensor_stay_ids.npy", allow_pickle=True)
    fm = load_npy(PROC_M / "mimic_sepsis_tensor_features.npy", allow_pickle=True)
    if any(x is None for x in (tm, te, im, ie, fm)):
        return None
    return pd.concat([tensor_to_wide(tm, im, fm, MIMIC),
                      tensor_to_wide(te, ie, fm, EICU)], ignore_index=True)


# ------------------------------------------------------------------ tier 3
def summary_matrix(tensor, statics_df, stay_ids, feat_names):
    blocks = [np.mean(tensor, axis=1), np.min(tensor, axis=1),
              np.max(tensor, axis=1), np.std(tensor, axis=1)]
    cols = []
    for stat in ("Mean", "Min", "Max", "Std"):
        cols += [f"{f}_{stat}" for f in feat_names]
    temporal = pd.DataFrame(np.concatenate(blocks, axis=1), columns=cols)
    stat = statics_df.reset_index(drop=True)
    out = pd.concat([stat, temporal], axis=1)
    out.insert(0, "stay_id", np.asarray(stay_ids))
    return out


def build_features_model(cohort: pd.DataFrame) -> pd.DataFrame | None:
    tm = load_npy(PROC_M / "mimic_sepsis_imputed_tensor.npy")
    te = load_npy(PROC_E / "eicu_sepsis_imputed_tensor.npy")
    im = load_npy(PROC_M / "mimic_sepsis_tensor_stay_ids.npy", allow_pickle=True)
    ie = load_npy(PROC_E / "eicu_sepsis_tensor_stay_ids.npy", allow_pickle=True)
    fm = load_npy(PROC_M / "mimic_sepsis_tensor_features.npy", allow_pickle=True)
    if any(x is None for x in (tm, te, im, ie, fm)):
        return None
    fnames = [str(f) for f in fm]

    def statics_for(src, ids):
        c = cohort[cohort["source_db"] == src].set_index("stay_id")
        c = c.reindex(np.asarray(ids))
        return pd.DataFrame({"age": c["age"].values, "baseline_sofa": c["baseline_sofa"].values})

    mm = summary_matrix(tm, statics_for(MIMIC, im), im, fnames); mm.insert(0, "source_db", MIMIC)
    ee = summary_matrix(te, statics_for(EICU, ie), ie, fnames); ee.insert(0, "source_db", EICU)
    out = pd.concat([mm, ee], ignore_index=True)

    # partition assignment, so the associated analysis is reproducible exactly
    tr = load_npy(MODELS / "mimic_train_indices.npy")
    te_idx = load_npy(MODELS / "mimic_test_set_indices.npy")
    part = pd.Series("external", index=out.index, dtype=object)
    if tr is not None and te_idx is not None:
        ids = np.asarray(im)
        lookup = {}
        for i in tr:
            lookup[ids[i]] = "development"
        for i in te_idx:
            lookup[ids[i]] = "held_out_test"
        mask = out["source_db"] == MIMIC
        part[mask] = out.loc[mask, "stay_id"].map(lookup).fillna("unassigned")
    out.insert(2, "partition", part)
    return out


# ------------------------------------------------------------------ tier 4
def build_features_transported(model_cols: list[str]) -> pd.DataFrame | None:
    atlas = None
    for name in ("atlas_sepsis_features.npy", "atlas_sepsis_features_124.npy"):
        if (PROC_A / name).exists():
            atlas = np.load(PROC_A / name)
            break
    meta = need(PROC_A / "atlas_metadata.parquet")
    if atlas is None or meta is None:
        missing_inputs.append("data/processed/atlas/atlas_sepsis_features_*.npy")
        return None
    md = pd.read_parquet(meta)
    fm = load_npy(PROC_M / "mimic_sepsis_tensor_features.npy", allow_pickle=True)
    if fm is None:
        return None
    fnames = [str(f) for f in fm]

    # 01a writes [120 temporal, then static in MODEL_STATICS order]
    statics = ["age", "baseline_sofa"]
    n_static = len(statics)
    n_temporal = atlas.shape[1] - n_static
    cols = []
    for stat in ("Mean", "Min", "Max", "Std"):
        cols += [f"{f}_{stat}" for f in fnames]
    cols = cols[:n_temporal] + statics

    df = pd.DataFrame(atlas, columns=cols)
    df.insert(0, "is_transported", (md["cohort_source"].astype(str).str.contains("eICU")).values)
    df.insert(0, "stay_id", md["stay_id"].values if "stay_id" in md.columns else md["atlas_id"].values)
    df.insert(0, "source_db", np.where(df["is_transported"], EICU, MIMIC))

    ordered = ["source_db", "stay_id", "is_transported"] + [c for c in model_cols if c in df.columns]
    return df[ordered + [c for c in df.columns if c not in ordered]]


# ------------------------------------------------------------------ docs
def build_dictionary(hourly_cols, model_cols) -> pd.DataFrame:
    rows = []
    for col, desc in COHORT_DOC.items():
        rows.append({"file": "cohort.csv.gz", "variable": col, "unit": "",
                     "mimic_source": "", "eicu_source": "", "hourly_aggregation": "",
                     "in_model": "", "description": desc})

    parity = {}
    pf = METRICS / "feature_parity_density.csv"
    if pf.exists():
        pdf = pd.read_csv(pf)
        for _, r in pdf.iterrows():
            parity[str(r["feature"]).lower()] = (r.get("mimic_density_pct"), r.get("eicu_density_pct"))

    for v in hourly_cols:
        unit, msrc, esrc, agg = VARIABLE_DOC.get(v, ("", "", "", ""))
        dm, de = parity.get(v.lower(), ("", ""))
        rows.append({"file": "hourly_observed.csv.gz / hourly_imputed.csv.gz",
                     "variable": v, "unit": unit, "mimic_source": msrc, "eicu_source": esrc,
                     "hourly_aggregation": agg, "in_model": "summarised",
                     "description": f"Observed density MIMIC-IV {dm}% eICU-CRD {de}%" if dm != "" else ""})

    for c in model_cols:
        rows.append({"file": "features_model.csv.gz / features_transported.csv.gz",
                     "variable": c, "unit": "", "mimic_source": "", "eicu_source": "",
                     "hourly_aggregation": "", "in_model": "yes",
                     "description": "Static variable" if "_" not in c else
                                    f"{c.rsplit('_', 1)[1]} of {c.rsplit('_', 1)[0]} across the 24 hour window"})
    return pd.DataFrame(rows)


def build_cohort_flow() -> pd.DataFrame:
    rows = []
    att = METRICS / "mimic_phenotype_lock_attrition.json"
    if att.exists():
        a = json.loads(att.read_text(encoding="utf-8"))
        rows.append({"source_db": MIMIC, "step": "confirmed infection cohort",
                     "n_stays": a.get("infection_cohort_in"), "note": ""})
        for r in a.get("per_rule", []):
            rows.append({"source_db": MIMIC, "step": f"matched mimic rule: {r['rule']}",
                         "n_stays": r["n_matched"], "note": "rules overlap, do not sum"})
        rows.append({"source_db": MIMIC, "step": "after phenotype lock",
                     "n_stays": a.get("phenotype_cohort_out"), "note": ""})
    cov = METRICS / "eicu_charlson_coverage.json"
    if cov.exists():
        c = json.loads(cov.read_text(encoding="utf-8"))
        rows.append({"source_db": EICU, "step": "confirmed infection cohort",
                     "n_stays": c.get("cohort_stays"), "note": ""})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ writing
def write_csv_gz(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False, compression="gzip")
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


README = """ClinicalTensorSepsis

Harmonised Sepsis-3 cohorts with hourly physiology derived from MIMIC-IV and the
eICU Collaborative Research Database. Both cohorts were constructed by a single
specification so that a difference in model performance between them can be
attributed to the data rather than to differing cohort definitions.

Files
  cohort.csv.gz                 one row per intensive care stay
  hourly_observed.csv.gz        measured values, one row per stay-hour.
                                A BLANK CELL MEANS NO OBSERVATION IN THAT HOUR.
  hourly_imputed.csv.gz         the same grid after reconstruction by a model
                                trained on the MIMIC-IV cohort only. Complete.
  features_model.csv.gz         summary representation used for modelling,
                                with the development and held-out assignment
  features_transported.csv.gz   the same representation after optimal transport.
                                ROWS FROM eICU-CRD CONTAIN MAPPED VALUES, NOT
                                MEASUREMENTS. Do not read them as observations.
  data_dictionary.csv           per-variable documentation, including how often
                                each variable is actually recorded in each cohort
  cohort_flow.csv               cohort attrition at each construction step
  sha256sums.txt                checksums for every file above

All times are integer offsets in minutes from intensive care admission. No
absolute dates appear anywhere in this package.

Recording density differs substantially between the two databases. Read
data_dictionary.csv before comparing them.

See the project description on PhysioNet for full methods, usage notes and
known limitations.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="release")
    args = ap.parse_args()
    out = BASE / args.outdir
    out.mkdir(parents=True, exist_ok=True)

    print(f"ClinicalTensorSepsis release export -> {out.relative_to(BASE)}/\n")
    written = []

    cohort = build_cohort()
    if cohort is not None:
        written.append(write_csv_gz(cohort, out / "cohort.csv.gz"))
        n_m = int((cohort["source_db"] == MIMIC).sum())
        n_e = int((cohort["source_db"] == EICU).sum())
        print(f"  cohort.csv.gz                 {len(cohort):>9,} rows   "
              f"MIMIC-IV {n_m:,} / eICU-CRD {n_e:,}")

    hourly_cols = []
    for kind, fname in (("raw", "hourly_observed.csv.gz"), ("imputed", "hourly_imputed.csv.gz")):
        h = build_hourly(kind)
        if h is None:
            continue
        hourly_cols = [c for c in h.columns if c not in ("source_db", "stay_id", "hour")]
        written.append(write_csv_gz(h, out / fname))
        blanks = int(h[hourly_cols].isna().sum().sum())
        print(f"  {fname:<29} {len(h):>9,} rows   {blanks:,} unobserved cells")

    model_cols = []
    fmdl = build_features_model(cohort) if cohort is not None else None
    if fmdl is not None:
        model_cols = [c for c in fmdl.columns if c not in ("source_db", "stay_id", "partition")]
        written.append(write_csv_gz(fmdl, out / "features_model.csv.gz"))
        print(f"  features_model.csv.gz         {len(fmdl):>9,} rows   {len(model_cols)} features")

    ftr = build_features_transported(model_cols)
    if ftr is not None:
        written.append(write_csv_gz(ftr, out / "features_transported.csv.gz"))
        print(f"  features_transported.csv.gz   {len(ftr):>9,} rows   "
              f"{int(ftr['is_transported'].sum()):,} transported")

    d = build_dictionary(hourly_cols, model_cols)
    d.to_csv(out / "data_dictionary.csv", index=False); written.append(out / "data_dictionary.csv")
    print(f"  data_dictionary.csv           {len(d):>9,} rows")

    cf = build_cohort_flow()
    cf.to_csv(out / "cohort_flow.csv", index=False); written.append(out / "cohort_flow.csv")
    print(f"  cohort_flow.csv               {len(cf):>9,} rows")

    (out / "readme.txt").write_text(README, encoding="utf-8")
    written.append(out / "readme.txt")

    lines = [f"{sha256(p)}  {p.name}" for p in sorted(written, key=lambda x: x.name)]
    (out / "sha256sums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  readme.txt, sha256sums.txt")

    print(f"\n  generated {datetime.now():%Y-%m-%d %H:%M}")

    if missing_inputs:
        print(f"\n[!] {len(set(missing_inputs))} input(s) missing; package is incomplete:")
        for m in sorted(set(missing_inputs)):
            print(f"      {m}")
        print("    Run pipeline/build_dataset.sh first.")
        sys.exit(1)

    print("\n  Next: physionet validate " + str(out.relative_to(BASE)))
    print("        croissant-baker --input " + str(out.relative_to(BASE)))


if __name__ == "__main__":
    main()
