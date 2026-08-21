"""
02_feature_importance_correlation.py

Calculates Spearman's Rank Correlation between the 100-iteration RFECV 
selection stability and the 50-iteration Consensus SHAP values.
This validates that the features most consistently selected by the algorithm 
are the exact same features driving its clinical predictions across resamples.
"""

import time
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Inputs (From Phase 2 / Phase 4)
RFECV_FILE = BASE_DIR / "outputs" / "metrics" / "mimic_rfecv_100_iteration_stability.csv"
SHAP_FILE = BASE_DIR / "outputs" / "features" / "mimic_consensus_feature_importance.csv"

# Output
OUT_ANALYSIS = BASE_DIR / "outputs" / "analysis"
OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
REPORT_FILE = OUT_ANALYSIS / "consensus_feature_correlation_report.csv"

def main():
    print("[*] Initiating Consensus Feature Correlation Audit...")
    start_time = time.time()
    
    if not RFECV_FILE.exists() or not SHAP_FILE.exists():
        print(f"[ERROR] Missing consensus files.")
        print(f"    RFECV Found: {RFECV_FILE.exists()}")
        print(f"    SHAP Found:  {SHAP_FILE.exists()}")
        return

    # Load the files
    df_rfecv = pd.read_csv(RFECV_FILE)
    df_shap = pd.read_csv(SHAP_FILE)
    
    # Normalize the feature column name just in case ("Feature", "feature", "feature_name")
    df_rfecv.rename(columns=lambda x: "Feature" if x.lower() in ["feature", "feature_name", "name"] else x, inplace=True)
    df_shap.rename(columns=lambda x: "Feature" if x.lower() in ["feature", "feature_name", "name"] else x, inplace=True)
    
    # Identify the value columns dynamically (usually the second column)
    rfecv_val_col = [col for col in df_rfecv.columns if col != "Feature"][0]
    shap_val_col = [col for col in df_shap.columns if col != "Feature"][0]
    
    # Merge on the Feature name
    df_compare = df_rfecv.merge(df_shap, on="Feature", how="inner")
    
    if df_compare.empty:
        print("[ERROR] Could not match feature names between RFECV and SHAP files.")
        return
        
    print(f"    -> Successfully matched {len(df_compare)} common features across consensus runs.")
    
    # Generate Ranks (Ascending=False means the highest value gets Rank 1)
    df_compare["Rank_RFECV_Stability"] = df_compare[rfecv_val_col].rank(ascending=False)
    df_compare["Rank_SHAP_Consensus"] = df_compare[shap_val_col].rank(ascending=False)
    
    # Calculate Spearman's Rank Correlation
    rho, p_val = spearmanr(df_compare["Rank_RFECV_Stability"], df_compare["Rank_SHAP_Consensus"])
    
    print("\n" + "="*60)
    print(" CONSENSUS FEATURE STABILITY VS. EXPLAINABILITY")
    print("="*60)
    print(f"    Spearman's Rho (ρ) : {rho:.4f}")
    print(f"    p-value            : {p_val:.3e}")
    
    if rho >= 0.8:
        print("    [PASS] Excellent correlation. Feature selection is highly robust and perfectly aligns with predictive logic.")
    elif rho >= 0.6:
        print("    [PASS] Strong correlation. The most stable features drive the majority of SHAP importance.")
    else:
        print("    [WARNING] Weak correlation. Model selection stability diverges from explanation stability.")
    print("="*60)

    # Sort by consensus SHAP rank and save
    df_compare = df_compare.sort_values("Rank_SHAP_Consensus").reset_index(drop=True)
    df_compare.to_csv(REPORT_FILE, index=False)
    
    print(f"\n[+] Success! Consensus correlation completed in {time.time() - start_time:.2f} seconds.")
    print(f"    -> Report saved to {REPORT_FILE.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()