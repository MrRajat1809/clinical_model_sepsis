"""
collect_results.py

Assembles every headline number the pipeline produces into one document,
organised to match the Results subsections of main.tex.

Reads the persisted artifacts under outputs/, never the run log, so it is
reproducible and can be re-run at any time without re-running the pipeline.
Anything missing is reported as missing rather than silently skipped, so a
half-finished run is obvious.

    python pipeline/collect_results.py                 # writes outputs/RESULTS_DIGEST.md
    python pipeline/collect_results.py --stdout        # also print to the terminal
"""

import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs"
METRICS = OUT / "metrics"
FEATURES = OUT / "features"
ANALYSIS = OUT / "analysis"
PREDS = OUT / "predictions"
PROC_M = BASE / "data" / "processed" / "mimiciv"
PROC_E = BASE / "data" / "processed" / "eicu"

missing: list[str] = []
lines: list[str] = []


def w(s: str = "") -> None:
    lines.append(s)


def load_json(path: Path):
    if not path.exists():
        missing.append(str(path.relative_to(BASE)))
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{path.relative_to(BASE)} (unreadable: {exc})")
        return None


def load_csv(path: Path):
    if not path.exists():
        missing.append(str(path.relative_to(BASE)))
        return None
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{path.relative_to(BASE)} (unreadable: {exc})")
        return None


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def ci(metrics: dict, key: str, nd=3) -> str:
    """Render 'value [lo - hi]' from the AUROC / AUROC_95CI convention."""
    if not metrics or key not in metrics:
        return "n/a"
    val = fmt(metrics[key], nd)
    lo_hi = metrics.get(f"{key}_95CI") or metrics.get(f"{key}_CI")
    if isinstance(lo_hi, (list, tuple)) and len(lo_hi) == 2:
        return f"{val} [{fmt(lo_hi[0], nd)} - {fmt(lo_hi[1], nd)}]"
    return val


def cohort_size(parquet: Path):
    if not parquet.exists():
        missing.append(str(parquet.relative_to(BASE)))
        return None, None
    try:
        import polars as pl
        df = pl.read_parquet(parquet, columns=["hospital_expire_flag"])
        return df.height, int(df["hospital_expire_flag"].sum())
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{parquet.relative_to(BASE)} (unreadable: {exc})")
        return None, None
    try:
        import pandas as pd
        df = pd.read_parquet(parquet, columns=["hospital_expire_flag"])
        return len(df), int(df["hospital_expire_flag"].sum())
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{parquet.relative_to(BASE)} (unreadable: {exc})")
        return None, None


def table(headers, rows):
    w("| " + " | ".join(headers) + " |")
    w("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w()


# ============================================================ 1. COHORT
def section_cohort():
    w("## 1. Cohort characterization and model development")
    w()

    n_m, d_m = cohort_size(PROC_M / "mimic_final_sepsis3_cohort.parquet")
    n_e, d_e = cohort_size(PROC_E / "eicu_final_sepsis3_cohort.parquet")
    rows = []
    for name, n, d in [("MIMIC-IV (development)", n_m, d_m), ("eICU-CRD (external)", n_e, d_e)]:
        rate = f"{d / n * 100:.1f}%" if n and d is not None else "n/a"
        rows.append([name, fmt(n), fmt(d), rate])
    w("**Final Sepsis-3 cohorts**")
    w()
    table(["Cohort", "N stays", "Deaths", "Hospital mortality"], rows)

    att = load_json(METRICS / "mimic_phenotype_lock_attrition.json")
    if att:
        w("**Phenotype lock attrition, MIMIC-IV** (rules overlap, so they do not sum)")
        w()
        w(f"Entering the lock: {fmt(att.get('infection_cohort_in'))} -> "
          f"leaving: {fmt(att.get('phenotype_cohort_out'))} "
          f"(removed {fmt(att.get('removed_total'))})")
        w()
        table(["Mimic rule", "N matched", "% of infection cohort"],
              [[r["rule"], fmt(r["n_matched"]), f"{r['pct_of_infection_cohort']}%"]
               for r in att.get("per_rule", [])])

    cov = load_json(METRICS / "eicu_charlson_coverage.json")
    if cov:
        w("**Comorbidity capture** (the residual MIMIC/eICU gap belongs in the limitations)")
        w()
        w(f"- eICU diagnosis rows without a usable ICD code: "
          f"{fmt(cov.get('rows_without_code'))} of {fmt(cov.get('diagnosis_rows'))} "
          f"({fmt(cov.get('pct_rows_using_text_fallback'), 1)}%), scored via the text fallback")
        w(f"- eICU stays with at least one coded diagnosis: {fmt(cov.get('stays_with_any_code'))} "
          f"of {fmt(cov.get('cohort_stays'))}")
        w(f"- eICU Charlson median (IQR): {fmt(cov.get('cci_median'), 1)} "
          f"({fmt(cov.get('cci_q25'), 1)} - {fmt(cov.get('cci_q75'), 1)}), "
          f"zero-CCI {fmt(cov.get('cci_pct_zero'), 1)}%")
        w()

    par = load_csv(METRICS / "feature_parity_density.csv")
    if par is not None:
        ov = par[par["feature"] == "OVERALL"]
        if len(ov):
            r = ov.iloc[0]
            w(f"**Tensor density before imputation** — MIMIC {r['mimic_density_pct']}%, "
              f"eICU {r['eicu_density_pct']}% "
              f"(absolute gap {r['abs_density_diff_pct']} points)")
            w()
        worst = par[par["feature"] != "OVERALL"].nlargest(5, "abs_density_diff_pct")
        table(["Largest density gaps", "MIMIC %", "eICU %", "gap"],
              [[x["feature"], x["mimic_density_pct"], x["eicu_density_pct"], x["abs_density_diff_pct"]]
               for _, x in worst.iterrows()])

    w("**Internal model comparison, MIMIC-IV held-out test set**")
    w()
    rows = []
    clin = load_json(METRICS / "mimic_clinical_baseline_metrics.json")
    if clin:
        for key, label in [("SOFA_Only", "SOFA alone"), ("Age_SOFA", "Age + SOFA")]:
            m = clin.get(key, {}).get("metrics", {})
            rows.append([label, fmt(m.get("AUROC")), fmt(m.get("AUPRC")), fmt(m.get("Brier"))])
    base = load_json(METRICS / "mimic_detailed_baseline_metrics.json")
    if base:
        for name, m in base.items():
            rows.append([name.replace("_", " "), ci(m, "AUROC"), ci(m, "AUPRC"), ci(m, "Brier")])
    for f, label in [("mimic_static_mlp_metrics.json", "Static MLP"),
                     ("mimic_temporal_bigru_metrics.json", "Temporal BiGRU"),
                     ("mimic_multimodal_bigru_metrics.json", "Multimodal BiGRU"),
                     ("mimic_attention_multimodal_bigru_metrics.json", "Attention BiGRU")]:
        m = load_json(METRICS / f)
        if m:
            rows.append([label, fmt(m.get("AUROC")), fmt(m.get("AUPRC")), fmt(m.get("Brier"))])
    champ = load_json(METRICS / "mimic_champion_metrics.json")
    if champ:
        m = champ.get("metrics", {})
        rows.append(["**XGBoost (primary)**", ci(m, "AUROC"), ci(m, "AUPRC"), ci(m, "Brier")])
    lr = load_json(METRICS / "mimic_champion_lr_metrics.json")
    if lr:
        m = lr.get("metrics", {})
        rows.append(["Elastic-net LR", ci(m, "AUROC"), ci(m, "AUPRC"), ci(m, "Brier")])
    table(["Model", "AUROC [95% CI]", "AUPRC [95% CI]", "Brier [95% CI]"], rows)

    if champ:
        hp = champ.get("hyperparameters", {})
        w("Selected hyperparameters: " + ", ".join(f"{k}={v}" for k, v in hp.items()))
        w()

    dl = load_json(ANALYSIS / "delong_significance_results.json")
    if dl:
        w(f"**DeLong**, primary XGBoost {fmt(dl.get('Champion_AUROC'))} vs logistic regression "
          f"{fmt(dl.get('Baseline_AUROC'))} on the same patients: P = {dl.get('p_value'):.3g}"
          + ("" if dl.get("Significant") else "  (not significant)"))
        w()

    abl = load_csv(METRICS / "mimic_modality_ablation_results.csv")
    if abl is not None:
        w("**Modality ablation** (paired bootstrap against the primary model)")
        w()
        table(["Modality", "Features", "AUROC", "delta vs primary", "P"],
              [[r["Modality"], int(r["Features"]), fmt(r["AUROC"]),
                f"{r['Δ_AUROC_from_Champion']:+.3f}", f"{r['P_Value_vs_Champion']:.3g}"]
               for _, r in abl.iterrows()])

    for f, label in [("mimic_seymour_endotype_summary.csv", "MIMIC-IV"),
                     ("eicu_seymour_endotype_summary.csv", "eICU-CRD")]:
        s = load_csv(METRICS / f)
        if s is not None:
            w(f"**Seymour endotype check, {label}**")
            w()
            cols = [c for c in ["Endotype", "Patient_Count", "Mortality_Rate", "Mean_Lactate",
                                "Mean_Creatinine", "Mean_MAP"] if c in s.columns]
            table(cols, [[_pct(r[c]) if c == "Mortality_Rate"
                          else (fmt(r[c], 2) if c != "Endotype" else r[c])
                          for c in cols] for _, r in s.iterrows()])


# ============================================================ 2. TRANSPORTABILITY
def _pct(v):
    """Mortality as a percentage, whether the CSV holds 0.437 or the string "43.7%"."""
    s = str(v).strip()
    if s.endswith("%"):
        return s
    try:
        return f"{float(s) * 100:.1f}%"
    except ValueError:
        return s


def section_transportability():
    w("## 2. External transportability")
    w()
    rows = []
    full = load_json(METRICS / "eicu_champion_metrics.json")
    if full:
        m = full.get("metrics", {})
        rows.append(["Full model, eICU", ci(m, "AUROC"), ci(m, "AUPRC"), ci(m, "Brier"),
                     fmt(m.get("Calibration_Slope")), fmt(m.get("Calibration_Intercept"))])
    pruned = load_json(METRICS / "eicu_pruned_metrics.json")
    if pruned:
        m = pruned.get("metrics", {})
        nf = pruned.get("n_features")
        label = f"Reduced model ({nf} features), eICU" if nf else "Reduced model, eICU"
        rows.append([label, ci(m, "AUROC"), ci(m, "AUPRC"), ci(m, "Brier"),
                     fmt(m.get("Calibration_Slope")), fmt(m.get("Calibration_Intercept"))])
    table(["Model", "AUROC [95% CI]", "AUPRC [95% CI]", "Brier [95% CI]", "Cal. slope", "Cal. intercept"], rows)

    champ = load_json(METRICS / "mimic_champion_metrics.json")
    if champ and full:
        a_in = champ["metrics"].get("AUROC")
        a_ex = full["metrics"].get("AUROC")
        if a_in and a_ex and a_in > 0.5:
            ret = (a_ex - 0.5) / (a_in - 0.5) * 100
            w(f"**Portability ratio for the model as a whole**: internal AUROC {fmt(a_in)} -> "
              f"external {fmt(a_ex)}, i.e. {ret:.1f}% of above-chance discrimination retained.")
            w()


# ============================================================ 3. CALIBRATION
def section_calibration():
    w("## 3. Calibration drift")
    w()
    cal = load_json(METRICS / "mimic_calibration_summary.json")
    if cal:
        w("**Internal calibration, MIMIC-IV held-out test set** (calibrators fitted out-of-fold)")
        w()
        table(["Representation", "AUROC", "Brier", "Slope", "Intercept", "ECE", "Threshold"],
              [[k, fmt(v.get("AUROC")), fmt(v.get("Brier")), fmt(v.get("Slope")),
                fmt(v.get("Intercept")), fmt(v.get("ECE")), fmt(v.get("Optimal_Threshold"))]
               for k, v in cal.get("Metrics", {}).items()])

    for f, label in [("eicu_calibration_metrics.json", "Full model"),
                     ("eicu_pruned_calibration_metrics.json", "Reduced model")]:
        ext = load_json(METRICS / f)
        if ext:
            w(f"**External recalibration on eICU-CRD, {label}** (80% calibration / 20% held-out test)")
            w()
            table(["Representation", "AUROC", "AUPRC", "Brier", "Threshold", "Sens", "Spec", "PPV", "F1"],
                  [[k, fmt(v.get("AUROC")), fmt(v.get("AUPRC")), fmt(v.get("Brier")),
                    fmt(v.get("Optimal_Threshold")), fmt(v.get("Sensitivity")),
                    fmt(v.get("Specificity")), fmt(v.get("PPV")), fmt(v.get("F1"))]
                   for k, v in ext.items()])

    dca = load_csv(METRICS / "eicu_dca_summary.csv")
    if dca is not None:
        w("**Decision curve analysis, net benefit at clinical thresholds**")
        w()
        cols = list(dca.columns)
        table(cols, [[fmt(r[c], 4) if c != "Threshold" else r[c] for c in cols]
                     for _, r in dca.iterrows()])


# ============================================================ 4. TEMPORAL
def section_temporal():
    w("## 4. Temporal portability")
    w()
    for f, label in [("mimic_temporal_early_warning_metrics.json", "Internal (MIMIC-IV test set)"),
                     ("eicu_temporal_early_warning_metrics.json", "External (eICU-CRD)")]:
        d = load_json(METRICS / f)
        if d:
            key = next(iter(d))
            w(f"**{label}**")
            w()
            table(["Window (h)", "AUROC", "AUPRC", "Brier"],
                  [[r["Hours"], fmt(r["AUROC"]), fmt(r["AUPRC"]), fmt(r["Brier"])] for r in d[key]])


# ============================================================ 5. OPTIMAL TRANSPORT
def section_ot():
    w("## 5. Optimal transport")
    w()
    h = load_json(METRICS / "atlas_ot_harmonization_metrics.json")
    if h:
        w(f"- Atlas size: {fmt(h.get('Total_Patients'))} patients")
        w(f"- Centroid MSE before OT {fmt(h.get('Pre_OT_Centroid_MSE'), 4)}, "
          f"after {fmt(h.get('Post_OT_Centroid_MSE'), 4)} "
          f"({fmt(h.get('Centroid_Alignment_Improvement_Pct'), 1)}% closer)")
        w()
    mix = load_json(METRICS / "atlas_manifold_mixing_metrics.json")
    if mix:
        w(f"- Joint manifold silhouette by cohort {fmt(mix.get('Cohort_Silhouette_Score'), 4)} "
          f"(0 = fully mixed), by mortality {fmt(mix.get('Mortality_Silhouette_Score'), 4)}")
        w()

    rows = []
    raw = load_json(METRICS / "eicu_champion_metrics.json")
    if raw:
        rows.append(["Unaligned eICU", ci(raw["metrics"], "AUROC"), ci(raw["metrics"], "AUPRC")])
    ot = load_json(METRICS / "eicu_OT_domain_adapted_metrics.json")
    if ot:
        rows.append(["OT-harmonized eICU", ci(ot["metrics"], "AUROC"), ci(ot["metrics"], "AUPRC")])
    otp = load_json(METRICS / "eicu_pruned_ot_metrics.json")
    if otp:
        rows.append(["OT-harmonized, reduced model", fmt(otp.get("AUROC")), fmt(otp.get("AUPRC"))])
    if rows:
        w("**Effect of alignment on external discrimination**")
        w()
        table(["Representation", "AUROC", "AUPRC"], rows)

    sw = load_json(METRICS / "atlas_ot_epsilon_sweep.json")
    if sw:
        w(f"**Regularization sensitivity** (baseline unharmonized AUROC "
          f"{fmt(sw.get('baseline_raw_eicu_auroc'))}, "
          f"{fmt(sw.get('sample_size_per_cohort'))} patients per cohort for the fit)")
        w()
        table(["epsilon", "OT AUROC", "shift from unaligned"],
              [[r["epsilon"], fmt(r["ot_auroc"]), f"{r['shift_from_raw']:+.4f}" if r["shift_from_raw"] is not None else "failed"]
               for r in sw.get("sweep", [])])

    cv = load_json(METRICS / "atlas_ot_constrained_variants.json")
    if cv:
        w("**Constrained transport variants** (Spearman is against the unaligned predictions, "
          "so it measures how far patient ranking survived)")
        w()
        table(["Variant", "AUROC", "Spearman vs unaligned"],
              [["Unaligned", fmt(cv.get("raw_eicu_auroc")), "1.000"],
               ["Standard Sinkhorn OT", fmt(cv.get("standard_ot_auroc")),
                fmt(cv.get("spearman_raw_vs_standard_ot"))],
               ["Post-OT interval projection", fmt(cv.get("interval_projection_auroc")),
                fmt(cv.get("spearman_raw_vs_interval_projection"))],
               [f"Feature-gated OT ({cv.get('n_features_protected_by_gate')} of "
                f"{cv.get('n_features_total')} protected)", fmt(cv.get("bio_gated_auroc")),
                fmt(cv.get("spearman_raw_vs_bio_gated"))]])

    mc = load_csv(ANALYSIS / "eicu_multicenter_variance_report.csv")
    if mc is not None and len(mc):
        sd_pre = mc["Pre_OT_AUROC"].std()
        sd_post = mc["Post_OT_AUROC"].std()
        w(f"**Between-hospital dispersion** across the {len(mc)} largest eICU hospitals: "
          f"AUROC SD {sd_pre:.4f} unaligned -> {sd_post:.4f} aligned")
        w()
        table(["Hospital", "N", "Mortality %", "AUROC unaligned", "AUROC aligned", "delta"],
              [[r["Hospital_ID"], int(r["N_Patients"]), fmt(r["Mortality_Rate_%"], 1),
                fmt(r["Pre_OT_AUROC"]), fmt(r["Post_OT_AUROC"]), f"{r['Delta_AUROC']:+.3f}"]
               for _, r in mc.iterrows()])

    uni = load_csv(METRICS / "atlas_univariate_auroc_shifts.csv")
    if uni is not None:
        w(f"**Per-feature effect of transport**: mean univariate AUROC shift "
          f"{uni['AUROC_Diff'].mean():+.4f} across {len(uni)} features; "
          f"{int((uni['AUROC_Diff'] < 0).sum())} features lost signal, "
          f"{int((uni['AUROC_Diff'] > 0).sum())} gained.")
        w()
        worst = uni.nsmallest(8, "AUROC_Diff")
        table(["Most degraded by OT", "pre-OT AUROC", "post-OT AUROC", "change"],
              [[r["Feature"], fmt(r["Pre_OT_AUROC"]), fmt(r["Post_OT_AUROC"]), f"{r['AUROC_Diff']:+.4f}"]
               for _, r in worst.iterrows()])


# ============================================================ 6. FEATURE STABILITY
def section_stability():
    w("## 6. Feature stability")
    w()
    stable = load_json(FEATURES / "mimic_stable_optimal_features.json")
    rfecv = load_csv(METRICS / "mimic_rfecv_100_iteration_stability.csv")
    if stable is not None:
        w(f"- Features retained in at least 80% of 100 RFECV iterations: **{len(stable)}** "
          f"of {len(rfecv) if rfecv is not None else 'n/a'}")
        w()
    if rfecv is not None:
        top = rfecv.head(15)
        table(["Most stable feature", "selection frequency"],
              [[r["Feature"], f"{r['Selection_Frequency_Pct']:.0f}%"] for _, r in top.iterrows()])

    sp = load_json(ANALYSIS / "consensus_feature_correlation.json")
    if sp:
        w(f"- Spearman correlation between RFECV selection stability and consensus SHAP rank: "
          f"rho = {fmt(sp.get('spearman_rho'))}, P = {sp.get('p_value'):.3g}, "
          f"over {fmt(sp.get('n_features_matched'))} matched features")
        w()

    pruned = load_json(METRICS / "eicu_pruned_metrics.json")
    champ = load_json(METRICS / "mimic_champion_metrics.json")
    full_ext = load_json(METRICS / "eicu_champion_metrics.json")
    if pruned and full_ext:
        w(f"- Reduced model retains external AUROC {fmt(pruned['metrics'].get('AUROC'))} "
          f"versus {fmt(full_ext['metrics'].get('AUROC'))} for the full representation")
        w()


# ============================================================ 7. CLINICAL DRIVERS
def section_drivers():
    w("## 7. Clinical drivers of prognostic prediction")
    w()
    top = load_json(FEATURES / "mimic_top_20_consensus_features.json")
    if top:
        w("**Consensus SHAP, 50 seeds, MIMIC-IV held-out test set**")
        w()
        table(["Rank", "Feature", "Mean |SHAP|", "95% CI across seeds"],
              [[i + 1, r["Feature"], fmt(r["Mean_Abs_SHAP"], 4),
                f"{fmt(r['Lower_95CI'], 4)} - {fmt(r['Upper_95CI'], 4)}"]
               for i, r in enumerate(top[:15])])

    fair = load_csv(ANALYSIS / "algorithmic_fairness_report.csv")
    if fair is not None:
        w("**Subgroup discrimination** (prespecified margin: absolute difference <= 0.02)")
        w()
        rows = []
        for (ds, sub), grp in fair.groupby(["Dataset", "Subgroup"], sort=False):
            if len(grp) == 2:
                a, b = grp.iloc[0], grp.iloc[1]
                diff = abs(a["AUROC"] - b["AUROC"])
                rows.append([ds, sub,
                             f"{a['Category']} {a['AUROC']:.3f} (n={int(a['N'])})",
                             f"{b['Category']} {b['AUROC']:.3f} (n={int(b['N'])})",
                             f"{diff:.3f}", "within" if diff <= 0.02 else "EXCEEDS"])
        table(["Cohort", "Subgroup", "Group A", "Group B", "|difference|", "margin"], rows)

    wa = load_csv(ANALYSIS / "wong_audit_a_thresholds.csv")
    wb = load_csv(ANALYSIS / "wong_audit_b_report.csv")
    if wa is not None:
        near = wa.iloc[(wa["Threshold"] - 0.225).abs().argsort()[:1]]
        if len(near):
            r = near.iloc[0]
            w(f"**Alert burden at the prespecified 0.225 threshold**: PPV {r['PPV']:.3f}, "
              f"number needed to evaluate {r['NNE']:.2f}, alert rate {r['Alert_Rate_Pct']:.1f}%")
            w()
    if wb is not None and len(wb):
        r = wb.iloc[0]
        w("**First recorded vasopressor relative to sepsis onset, among true-positive alerts** "
          "(descriptive only; this is not evidence the model predicted treatment need)")
        w()
        tot = r.get("Total_True_Positive_Alerts", 0)
        rows = []
        for col, label in [("TP_Pressor_Before_Onset", "before onset"),
                           ("TP_Pressor_Within_24h", "0-24 h after onset"),
                           ("TP_Pressor_After_24h", ">24 h after onset"),
                           ("TP_No_Post_Onset_Pressor", "none recorded")]:
            if col in wb.columns:
                v = int(r[col])
                rows.append([label, fmt(v), f"{v / tot * 100:.1f}%" if tot else "n/a"])
        table(["Timing", "N", "% of true positives"], rows)
        if "Lead_Time_Median_Hrs" in wb.columns and r.notna().get("Lead_Time_Median_Hrs", False):
            w(f"For the >24 h group, median interval beyond the 24-hour horizon: "
              f"{fmt(r['Lead_Time_Median_Hrs'], 1)} h "
              f"(IQR {fmt(r.get('Lead_Time_IQR_25'), 1)} - {fmt(r.get('Lead_Time_IQR_75'), 1)})")
            w()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="also print the digest")
    args = ap.parse_args()

    import datetime
    w("# Results digest")
    w()
    w(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from the artifacts under `outputs/`.")
    w("Section numbering matches the Results subsections of `main.tex`.")
    w()
    w("Every number here is read from a persisted artifact, not from the run log, "
      "so this file can be regenerated at any time without re-running anything.")
    w()
    w("---")
    w()

    for fn in (section_cohort, section_transportability, section_calibration,
               section_temporal, section_ot, section_stability, section_drivers):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            w(f"> **collector error in {fn.__name__}: {exc}**")
            w()
        w("---")
        w()

    if missing:
        w("## Artifacts not found")
        w()
        w("These sections are incomplete. Either the stage that writes them has not "
          "run, or it failed.")
        w()
        for m in sorted(set(missing)):
            w(f"- `{m}`")
        w()

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "RESULTS_DIGEST.md"
    dest.write_text("\n".join(lines), encoding="utf-8")

    print(f"[+] wrote {dest.relative_to(BASE)}  ({len(lines)} lines)")
    if missing:
        print(f"[!] {len(set(missing))} artifact(s) missing - digest is partial:")
        for m in sorted(set(missing))[:10]:
            print(f"      {m}")
        if len(set(missing)) > 10:
            print(f"      ... and {len(set(missing)) - 10} more")
    else:
        print("[+] all expected artifacts present")

    if args.stdout:
        print()
        print("\n".join(lines))


if __name__ == "__main__":
    main()
