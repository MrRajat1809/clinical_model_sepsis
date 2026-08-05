"""
07_decision_curve_analysis.py

Phase 7: Decision Curve Analysis (DCA)
Evaluates the clinical utility of the Champion model compared to standard clinical baselines.
- Calculates Net Benefit across a continuum of probability thresholds.
- Compares Champion (Uncalibrated & Platt), SOFA, Age + SOFA, Treat All, and Treat None.
- Generates a publication-quality DCA plot and a structured summary table for standard thresholds.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Inputs
PREDS_CHAMPION_FILE = BASE_DIR / "outputs" / "calibration" / "predictions" / "06_calibrated_predictions.csv"
PREDS_CLINICAL_FILE = BASE_DIR / "outputs" / "predictions" / "01_clinical_baselines_predictions.csv"

# Outputs
OUT_DCA_DIR = BASE_DIR / "outputs" / "decision_curve"
OUT_DCA_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds to evaluate for the continuous curve
THRESHOLDS = np.linspace(0.01, 0.99, 99)
# Key clinical thresholds for the summary table
SUMMARY_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

# ==========================================
# NET BENEFIT CALCULATION
# ==========================================
def compute_net_benefit(y_true, y_prob, threshold):
    """
    Computes Net Benefit for a given threshold.
    Formula: (True Positives / N) - (False Positives / N) * (threshold / (1 - threshold))
    """
    if threshold >= 1.0 or threshold <= 0.0:
        return 0.0
        
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    return net_benefit

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Initiating Phase 7: Decision Curve Analysis (DCA)...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # 1. LOAD & MERGE PREDICTIONS
    # ---------------------------------------------------------
    print("    -> Loading and merging model and baseline predictions...")
    if not PREDS_CHAMPION_FILE.exists() or not PREDS_CLINICAL_FILE.exists():
        print("[ERROR] Prediction files not found. Ensure Phase 1 and Phase 6 have been run.")
        return
        
    df_champion = pd.read_csv(PREDS_CHAMPION_FILE)
    df_clinical = pd.read_csv(PREDS_CLINICAL_FILE)
    
    # Merge on stay_id to ensure absolute alignment
    df_merged = pd.merge(df_champion, df_clinical[["stay_id", "sofa_probability", "age_sofa_probability"]], 
                         on="stay_id", how="inner")
                         
    y_true = df_merged["true_label"].values
    n_patients = len(y_true)
    prevalence = np.mean(y_true)
    
    print(f"       - Cohort Size: {n_patients} | Sepsis Mortality Prevalence: {prevalence*100:.1f}%")

    # ---------------------------------------------------------
    # 2. CALCULATE NET BENEFIT ACROSS ALL THRESHOLDS
    # ---------------------------------------------------------
    print("    -> Calculating Net Benefit across thresholds...")
    
    models = {
        "Champion (Platt)": df_merged["platt"].values,
        "Champion (Uncalibrated)": df_merged["uncalibrated"].values,
        "SOFA": df_merged["sofa_probability"].values,
        "Age + SOFA": df_merged["age_sofa_probability"].values
    }
    
    dca_results = {"Threshold": THRESHOLDS}
    
    # Baseline strategies
    dca_results["Treat None"] = [0.0] * len(THRESHOLDS)
    dca_results["Treat All"] = [compute_net_benefit(y_true, np.ones_like(y_true), t) for t in THRESHOLDS]
    
    # Models
    for name, probs in models.items():
        dca_results[name] = [compute_net_benefit(y_true, probs, t) for t in THRESHOLDS]
        
    df_dca = pd.DataFrame(dca_results)
    df_dca.to_csv(OUT_DCA_DIR / "dca_plot_data.csv", index=False)

    # ---------------------------------------------------------
    # 3. GENERATE SUMMARY TABLE FOR SPECIFIC THRESHOLDS
    # ---------------------------------------------------------
    print("    -> Generating threshold summary table...")
    summary_data = []
    
    for t in SUMMARY_THRESHOLDS:
        row = {"Threshold (%)": f"{t*100:.0f}%"}
        row["Treat All"] = compute_net_benefit(y_true, np.ones_like(y_true), t)
        row["Treat None"] = 0.0
        for name, probs in models.items():
            row[name] = compute_net_benefit(y_true, probs, t)
        summary_data.append(row)
        
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(OUT_DCA_DIR / "threshold_summary.csv", index=False)
    
    # Create a cleaner version for JSON export (transposed conceptually)
    metrics_json = {"prevalence": float(prevalence), "thresholds": {}}
    for _, row in df_summary.iterrows():
        t_key = row["Threshold (%)"]
        metrics_json["thresholds"][t_key] = {k: float(v) for k, v in row.items() if k != "Threshold (%)"}
        
    with open(OUT_DCA_DIR / "dca_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=4)

    # ---------------------------------------------------------
    # 4. PLOT DECISION CURVE
    # ---------------------------------------------------------
    print("    -> Generating publication-quality DCA plot...")
    
    plt.figure(figsize=(10, 7))
    
    # Plot baselines
    plt.plot(THRESHOLDS, df_dca["Treat None"], color="black", linestyle="-", linewidth=2, label="Treat None")
    plt.plot(THRESHOLDS, df_dca["Treat All"], color="gray", linestyle="--", linewidth=2, label="Treat All")
    
    # Plot clinical baselines
    plt.plot(THRESHOLDS, df_dca["SOFA"], color="orange", linestyle="-", linewidth=2, label="SOFA")
    plt.plot(THRESHOLDS, df_dca["Age + SOFA"], color="darkorange", linestyle="-.", linewidth=2, label="Age + SOFA")
    
    # Plot Champion
    plt.plot(THRESHOLDS, df_dca["Champion (Uncalibrated)"], color="steelblue", linestyle=":", linewidth=2, label="Champion (Uncalibrated)")
    plt.plot(THRESHOLDS, df_dca["Champion (Platt)"], color="firebrick", linestyle="-", linewidth=2.5, label="Champion (Platt)")
    
    # Aesthetics
    plt.ylim([-0.05, max(prevalence + 0.05, 0.25)])  # Zoom in on the relevant clinical range
    plt.xlim([0.0, 0.6]) # Limit to 60% threshold for sepsis mortality as >60% is rarely clinically actionable
    
    plt.title("Decision Curve Analysis: Early Sepsis Mortality")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    plt.savefig(OUT_DCA_DIR / "dca_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 5. PRINT SUMMARY
    # ---------------------------------------------------------
    print("\n" + "="*95)
    print(" DECISION CURVE ANALYSIS (NET BENEFIT AT KEY THRESHOLDS)")
    print("="*95)
    # Print formatted table
    header = f"{'Threshold':<12} | {'Treat All':<10} | {'SOFA':<10} | {'Age+SOFA':<10} | {'Champion(Uncal)':<17} | {'Champion(Platt)':<17}"
    print(header)
    print("-" * 95)
    
    for _, row in df_summary.iterrows():
        print(f" {row['Threshold (%)']:<11} | {row['Treat All']:<10.4f} | {row['SOFA']:<10.4f} | {row['Age + SOFA']:<10.4f} | {row['Champion (Uncalibrated)']:<17.4f} | {row['Champion (Platt)']:<17.4f}")
    print("="*95)
    
    elapsed = time.time() - start_time
    print(f"[*] DCA completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()