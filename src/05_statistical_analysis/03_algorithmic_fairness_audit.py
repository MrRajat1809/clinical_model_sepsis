"""
03_algorithmic_fairness_audit.py

Evaluates algorithmic fairness across demographic subgroups (Gender and Age).
Utilizes 1000 bootstrap iterations to generate 95% Confidence Intervals.
Runs across MIMIC (Internal), eICU (External), and Atlas (Global) datasets.
"""

import time
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]

# Paths
MIMIC_PREDS = BASE_DIR / "outputs" / "predictions" / "mimic_champion_predictions.csv"
MIMIC_COHORT = BASE_DIR / "data" / "processed" / "mimiciv" / "mimic_final_sepsis3_cohort.parquet"

EICU_PREDS = BASE_DIR / "outputs" / "metrics" / "eicu_champion_predictions.csv"
EICU_COHORT = BASE_DIR / "data" / "processed" / "eicu" / "eicu_final_sepsis3_cohort.parquet"

REPORT_FILE = BASE_DIR / "outputs" / "analysis" / "algorithmic_fairness_report.csv"

def bootstrap_auc(y_true, y_prob, n_iterations=1000):
    """Calculates AUROC with 95% CI using bootstrapping."""
    aucs = []
    for _ in range(n_iterations):
        idx = resample(np.arange(len(y_true)))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return np.nan, np.nan, np.nan
    return np.mean(aucs), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

def evaluate_fairness(df_eval, dataset_name):
    print(f"\n========================================================")
    print(f" FAIRNESS AUDIT: {dataset_name.upper()} (N={len(df_eval)})")
    print(f"========================================================")
    
    label_col = "true_label" if "true_label" in df_eval.columns else "label"
    prob_col = "pred_probability" if "pred_probability" in df_eval.columns else "probability"

    results = []

    # --- 1. GENDER FAIRNESS ---
    print("    -> Evaluating Gender Fairness...")
    male_mask = df_eval["gender"] == "M"
    female_mask = df_eval["gender"] == "F"
    
    auc_m, ci_lower_m, ci_upper_m = bootstrap_auc(df_eval.loc[male_mask, label_col].values, df_eval.loc[male_mask, prob_col].values)
    auc_f, ci_lower_f, ci_upper_f = bootstrap_auc(df_eval.loc[female_mask, label_col].values, df_eval.loc[female_mask, prob_col].values)
    
    delta_gender = abs(auc_m - auc_f)
    print(f"       Male AUROC   : {auc_m:.4f} (95% CI: {ci_lower_m:.4f}-{ci_upper_m:.4f}) | N={male_mask.sum()}")
    print(f"       Female AUROC : {auc_f:.4f} (95% CI: {ci_lower_f:.4f}-{ci_upper_f:.4f}) | N={female_mask.sum()}")
    print(f"       Difference   : {delta_gender:.4f}")
    if delta_gender <= 0.02:
        print("       [PASS] Gender fairness margin (<=0.02) achieved.")
    else:
        print("       [FAIL] Gender fairness margin exceeded.")

    results.extend([
        {"Dataset": dataset_name, "Subgroup": "Gender", "Category": "Male", "N": male_mask.sum(), "AUROC": auc_m, "CI_Lower": ci_lower_m, "CI_Upper": ci_upper_m},
        {"Dataset": dataset_name, "Subgroup": "Gender", "Category": "Female", "N": female_mask.sum(), "AUROC": auc_f, "CI_Lower": ci_lower_f, "CI_Upper": ci_upper_f}
    ])

    # --- 2. AGE FAIRNESS ---
    print("\n    -> Evaluating Age Fairness (<60 vs >=60)...")
    young_mask = df_eval["age"] < 60
    old_mask = df_eval["age"] >= 60
    
    auc_y, ci_lower_y, ci_upper_y = bootstrap_auc(df_eval.loc[young_mask, label_col].values, df_eval.loc[young_mask, prob_col].values)
    auc_o, ci_lower_o, ci_upper_o = bootstrap_auc(df_eval.loc[old_mask, label_col].values, df_eval.loc[old_mask, prob_col].values)
    
    delta_age = abs(auc_y - auc_o)
    print(f"       Age < 60 AUROC  : {auc_y:.4f} (95% CI: {ci_lower_y:.4f}-{ci_upper_y:.4f}) | N={young_mask.sum()}")
    print(f"       Age >= 60 AUROC : {auc_o:.4f} (95% CI: {ci_lower_o:.4f}-{ci_upper_o:.4f}) | N={old_mask.sum()}")
    print(f"       Difference      : {delta_age:.4f}")
    if delta_age <= 0.02:
        print("       [PASS] Age fairness margin (<=0.02) achieved.")
    else:
        print("       [FAIL] Age fairness margin exceeded.")

    results.extend([
        {"Dataset": dataset_name, "Subgroup": "Age", "Category": "< 60", "N": young_mask.sum(), "AUROC": auc_y, "CI_Lower": ci_lower_y, "CI_Upper": ci_upper_y},
        {"Dataset": dataset_name, "Subgroup": "Age", "Category": ">= 60", "N": old_mask.sum(), "AUROC": auc_o, "CI_Lower": ci_lower_o, "CI_Upper": ci_upper_o}
    ])
    
    return results

def main():
    print("[*] Initiating Multi-Cohort Algorithmic Fairness Audit (1000 Bootstraps)...")
    start_time = time.time()

    all_results = []

    # 1. MIMIC-IV
    df_mimic_preds = pd.read_csv(MIMIC_PREDS)
    df_mimic_cohort = pl.read_parquet(MIMIC_COHORT).select(["stay_id", "age", "gender"]).to_pandas()
    df_mimic_eval = df_mimic_preds.merge(df_mimic_cohort, on="stay_id", how="inner")
    
    # Normalize gender (just in case)
    df_mimic_eval["gender"] = df_mimic_eval["gender"].astype(str).str.upper().str[0]
    
    all_results.extend(evaluate_fairness(df_mimic_eval, "MIMIC-IV (Internal)"))

    # 2. eICU
    if EICU_PREDS.exists() and EICU_COHORT.exists():
        df_eicu_preds = pd.read_csv(EICU_PREDS)
        df_eicu_cohort = pl.read_parquet(EICU_COHORT).select(["stay_id", "age", "gender"]).to_pandas()
        
        # eICU predictions might not have a stay_id column named 'stay_id', or labels might differ. 
        # Ensure alignment.
        df_eicu_eval = df_eicu_preds.merge(df_eicu_cohort, on="stay_id", how="inner")
        
        # Normalize eICU gender (Male -> M, Female -> F)
        df_eicu_eval["gender"] = df_eicu_eval["gender"].astype(str).str.upper().str[0]
        
        all_results.extend(evaluate_fairness(df_eicu_eval, "eICU (External)"))

        # 3. ATLAS (Combined)
        # Rename columns to ensure concat compatibility if they differ slightly
        prob_col_m = "pred_probability" if "pred_probability" in df_mimic_eval.columns else "probability"
        label_col_m = "true_label" if "true_label" in df_mimic_eval.columns else "label"
        df_m = df_mimic_eval[["stay_id", prob_col_m, label_col_m, "age", "gender"]].rename(columns={prob_col_m: "probability", label_col_m: "label"})

        prob_col_e = "pred_probability" if "pred_probability" in df_eicu_eval.columns else "probability"
        label_col_e = "true_label" if "true_label" in df_eicu_eval.columns else "label"
        df_e = df_eicu_eval[["stay_id", prob_col_e, label_col_e, "age", "gender"]].rename(columns={prob_col_e: "probability", label_col_e: "label"})

        df_atlas_eval = pd.concat([df_m, df_e], ignore_index=True)
        all_results.extend(evaluate_fairness(df_atlas_eval, "Global Atlas (MIMIC + eICU)"))
    else:
        print("\n[WARNING] eICU files not found. Skipping eICU and Atlas audits.")

    # Save Results
    pd.DataFrame(all_results).to_csv(REPORT_FILE, index=False)
    print(f"\n[+] Success! Fairness audit completed in {time.time() - start_time:.2f} seconds.")
    print(f"    -> Full report saved to {REPORT_FILE.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()