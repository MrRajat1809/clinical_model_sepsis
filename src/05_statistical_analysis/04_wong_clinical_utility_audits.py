"""
04_wong_clinical_utility_audits.py

Executes clinical utility and temporal context audits inspired by Wong et al. 

Interpretation note:
Audit B is a descriptive temporal-context analysis. It measures whether
the model's fixed 24-hour prediction horizon precedes the first recorded
post-onset vasopressor administration. It does NOT establish that the
model predicted vasopressor need, caused earlier intervention, or provided
clinical benefit.
"""

import time
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

PREDS_FILE = BASE_DIR / "outputs" / "predictions" / "mimic_champion_predictions.csv"
TEMPORAL_FILE = BASE_DIR / "data" / "processed" / "mimiciv" / "mimic_sepsis_temporal_data_cleaned.parquet"
COHORT_FILE = BASE_DIR / "data" / "processed" / "mimiciv" / "mimic_final_sepsis3_cohort.parquet"

OUT_ANALYSIS = BASE_DIR / "outputs" / "analysis"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"
OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

AUDIT_A_REPORT = OUT_ANALYSIS / "wong_audit_a_thresholds.csv"
AUDIT_B_REPORT = OUT_ANALYSIS / "wong_audit_b_report.csv"
AUDIT_B_PATIENTS = OUT_ANALYSIS / "wong_audit_b_patient_timing.parquet"
PLOT_FILE = OUT_FIGURES / "wong_clinical_utility_audits.png"

TARGET_THRESHOLD = 0.225 

def main():
    print("[*] Initiating Wong et al. Clinical Utility Audits...")
    start_time = time.time()

    # ---------------------------------------------------------
    # 1. LOAD PREDICTIONS
    # ---------------------------------------------------------
    if not PREDS_FILE.exists():
        print(f"[ERROR] Predictions file missing: {PREDS_FILE}")
        return
        
    df_preds = pd.read_csv(PREDS_FILE)
    prob_col = "pred_probability" if "pred_probability" in df_preds.columns else "probability"
    label_col = "true_label" if "true_label" in df_preds.columns else "label"
    
    y_true = df_preds[label_col].values
    y_prob = df_preds[prob_col].values

    # ---------------------------------------------------------
    # AUDIT A: ALERT BURDEN & NNE
    # ---------------------------------------------------------
    print("\n[AUDIT A]: Alert Burden and Number Needed to Evaluate (NNE)")
    
    # Exact prespecified threshold
    y_pred_target = (y_prob >= TARGET_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_target, labels=[0, 1]).ravel()
    
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    nne = 1 / ppv if ppv > 0 else np.nan
    
    alert_rate = (tp + fp) / len(y_true)
    false_alert_rate = fp / len(y_true)
    
    print(f"    - Threshold       : {TARGET_THRESHOLD:.3f}")
    print(f"    - PPV             : {ppv*100:.1f}%")
    print(f"    - NNE             : {nne:.2f}")
    print(f"    - Sensitivity     : {sensitivity*100:.1f}%")
    print(f"    - Specificity     : {specificity*100:.1f}%")
    print(f"    - Alert rate      : {alert_rate*100:.1f}%")
    print(f"    - False-alert rate: {false_alert_rate*100:.1f}%")
    
    # Save full threshold sweep for the plot
    thresholds = np.linspace(0.05, 0.95, 100)
    audit_a_results = []
    for thresh in thresholds:
        yp = (y_prob >= thresh).astype(int)
        tn_i, fp_i, fn_i, tp_i = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
        p_i = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0
        audit_a_results.append({
            "Threshold": thresh,
            "PPV": p_i,
            "NNE": 1 / p_i if p_i > 0 else np.nan,
            "Alert_Rate_Pct": (tp_i + fp_i) / len(y_true) * 100
        })
    df_audit_a = pd.DataFrame(audit_a_results)
    df_audit_a.to_csv(AUDIT_A_REPORT, index=False)

    # ---------------------------------------------------------
    # AUDIT B: USUAL CARE / INTERVENTION TIMING
    # ---------------------------------------------------------
    print("\n[AUDIT B]: Clinical Intervention Timing (Contextual)")
    print("    -> Measuring absolute first recorded vasopressor relative to sepsis onset...")

    mimic_pressor_ids = ["221906", "221289", "221662", "221653", "222315", "221749"]

    df_temporal = (
        pl.read_parquet(TEMPORAL_FILE)
        .select(["stay_id", "event_time", "itemid", "valuenum"])
        .to_pandas()
    )
    df_temporal["itemid"] = df_temporal["itemid"].astype(str)
    df_temporal["event_time"] = pd.to_datetime(df_temporal["event_time"])

    df_pressors = df_temporal[
        df_temporal["itemid"].isin(mimic_pressor_ids) & (df_temporal["valuenum"] > 0)
    ].copy()

    if df_pressors.empty:
        print("    [WARNING] No positive vasopressor administrations found.")
        return

    df_cohort = (
        pl.read_parquet(COHORT_FILE)
        .select(["stay_id", "sepsis_onset_time"])
        .to_pandas()
    )
    df_cohort["sepsis_onset_time"] = pd.to_datetime(df_cohort["sepsis_onset_time"])
    
    df_pressors = df_pressors.merge(df_cohort, on="stay_id", how="inner")
    
    df_pressors["hours_from_onset"] = (
        df_pressors["event_time"] - df_pressors["sepsis_onset_time"]
    ).dt.total_seconds() / 3600.0

    # Find the ABSOLUTE FIRST pressor across the entire record
    idx = df_pressors.groupby("stay_id")["event_time"].idxmin()
    absolute_first_pressor = (
        df_pressors.loc[idx, ["stay_id", "event_time", "itemid", "hours_from_onset"]]
        .rename(columns={
            "event_time": "first_pressor_time",
            "itemid": "first_pressor_itemid",
            "hours_from_onset": "first_pressor_hour"
        })
    )

    df_tp = df_preds[(df_preds[label_col] == 1) & (df_preds[prob_col] >= TARGET_THRESHOLD)].copy()
    n_tp = len(df_tp)

    df_tp_timing = df_tp.merge(absolute_first_pressor, on="stay_id", how="left")

    # Mutually Exclusive Categories
    has_pressor = df_tp_timing["first_pressor_hour"].notna()
    
    pre_onset = has_pressor & (df_tp_timing["first_pressor_hour"] < 0)
    within_24h = has_pressor & (df_tp_timing["first_pressor_hour"] >= 0) & (df_tp_timing["first_pressor_hour"] <= 24)
    after_24h = has_pressor & (df_tp_timing["first_pressor_hour"] > 24)
    no_pressor = ~has_pressor

    n_pre = int(pre_onset.sum())
    n_within24 = int(within_24h.sum())
    n_after24 = int(after_24h.sum())
    n_no_post = int(no_pressor.sum())
    
    # [FIX] Define the denominator for the plotting legend
    n_with_post = n_within24 + n_after24

    print(f"\n    - Alert threshold                     : {TARGET_THRESHOLD:.3f}")
    print(f"    - True-positive alerts                : {n_tp:,}")

    print("\n    [ABSOLUTE FIRST PRESSOR TIMING AMONG TRUE-POSITIVE ALERTS]")
    print(f"    - First pressor before sepsis onset   : {n_pre:,} ({n_pre/n_tp*100:.1f}%)")
    print(f"    - First pressor 0-24 h after onset    : {n_within24:,} ({n_within24/n_tp*100:.1f}%)")
    print(f"    - First pressor >24 h after onset     : {n_after24:,} ({n_after24/n_tp*100:.1f}%)")
    print(f"    - No pressor recorded                 : {n_no_post:,} ({n_no_post/n_tp*100:.1f}%)")

    audit_b_results = {
        "Total_True_Positive_Alerts": n_tp,
        "TP_Pressor_Before_Onset": n_pre,
        "TP_Pressor_Within_24h": n_within24,
        "TP_Pressor_After_24h": n_after24,
        "TP_No_Post_Onset_Pressor": n_no_post
    }

    if n_with_post > 0:
        post_onset_mask = within_24h | after_24h
        median_latency = df_tp_timing.loc[post_onset_mask, "first_pressor_hour"].median()
        print(f"\n    - Median post-onset pressor timing    : {median_latency:.1f} h")
        
        if n_after24 > 0:
            lead_times = df_tp_timing.loc[after_24h, "first_pressor_hour"] - 24.0
            print(f"\n    [TEMPORAL PRECEDENCE FOR >24H GROUP]")
            print(f"    - Median Precedence: {lead_times.median():.1f} hours (IQR: {lead_times.quantile(0.25):.1f} - {lead_times.quantile(0.75):.1f})")
            
            audit_b_results["Lead_Time_Median_Hrs"] = lead_times.median()
            audit_b_results["Lead_Time_IQR_25"] = lead_times.quantile(0.25)
            audit_b_results["Lead_Time_IQR_75"] = lead_times.quantile(0.75)

    # Save outputs
    pd.DataFrame([audit_b_results]).to_csv(AUDIT_B_REPORT, index=False)
    df_tp_timing.to_parquet(AUDIT_B_PATIENTS, index=False)

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    print("\n    -> Generating Wong Audit Visualization Dashboard...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')

    # Panel A: Alert Fatigue Curve
    ax1 = axes[0]
    ax1.plot(df_audit_a["Threshold"], df_audit_a["NNE"], color="#4C72B0", lw=3)
    ax1.axvline(TARGET_THRESHOLD, color="#C44E52", linestyle="--", label=f"Clinical Cutoff (NNE={nne:.1f})")
    ax1.set_ylim(0, 15)
    ax1.set_title("A) Alert Burden: Number Needed to Evaluate (NNE)", fontweight="bold", fontsize=14)
    ax1.set_xlabel("AI Risk Score Threshold", weight="bold")
    ax1.set_ylabel("Patients Evaluated per 1 True Positive", weight="bold")
    ax1.legend(frameon=True)
    ax1.set_facecolor("#f8f9fa")
    sns.despine(ax=ax1)

    # Panel B: Latency Distribution
    ax2 = axes[1]
    plot_data = df_tp_timing[df_tp_timing["first_pressor_hour"].notna() & (df_tp_timing["first_pressor_hour"] <= 72)]
    
    sns.histplot(plot_data["first_pressor_hour"], bins=30, color="#C44E52", kde=True, ax=ax2)
    ax2.axvline(24, color="black", linestyle="--", lw=2, label="Prediction Available (Hour 24)")
    ax2.set_title("B) Temporal Precedence: First Post-Onset Vasopressor", fontweight="bold", fontsize=14)
    ax2.set_xlabel("Hours from Sepsis Onset Time", weight="bold")
    ax2.set_ylabel("Number of True Positive Patients", weight="bold")
    
    precedence_pct = (n_after24 / n_with_post * 100) if n_with_post > 0 else 0
    ax2.axvspan(24, plot_data["first_pressor_hour"].max(), color="#74c476", alpha=0.2, 
                label=f"Prediction precedes recorded pressor ({precedence_pct:.1f}%)")
    ax2.legend(frameon=True)
    ax2.set_facecolor("#f8f9fa")
    sns.despine(ax=ax2)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n[+] Success! Wong Audits completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()