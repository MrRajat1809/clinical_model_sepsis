"""
05_shap_visualization.py

Generates publication-quality SHAP plots (Beeswarm and Bar) using the 
exported consensus SHAP values from the MIMIC-IV hold-out test set.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Inputs from Phase 8
SHAP_DATA_DIR = BASE_DIR / "outputs" / "shap" / "data"
SHAP_VALUES_FILE = SHAP_DATA_DIR / "shap_values_exact_test.npy"
TEST_FEATS_FILE = SHAP_DATA_DIR / "test_features_scaled.csv"

# Outputs
OUT_PLOTS_DIR = BASE_DIR / "outputs" / "shap" / "plots"
OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BEESWARM_OUT = OUT_PLOTS_DIR / "Fig3A_SHAP_Beeswarm.pdf"
BAR_OUT = OUT_PLOTS_DIR / "Fig3B_SHAP_Bar.pdf"

MAX_DISPLAY = 20  # Number of top features to show in the plot

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Initiating SHAP Visualization Generation...")
    
    if not SHAP_VALUES_FILE.exists() or not TEST_FEATS_FILE.exists():
        print(f"[ERROR] SHAP data not found. Run 07_shap_interpretation.py first.")
        return

    # 1. Load the precomputed Consensus SHAP arrays
    print("    -> Loading 50-model consensus SHAP values and feature matrix...")
    shap_values = np.load(SHAP_VALUES_FILE)
    X_test_scaled = pd.read_csv(TEST_FEATS_FILE)

    # 2. Generate SHAP Beeswarm Plot (Local Interpretability)
    print("    -> Rendering Patient-Level Beeswarm Plot...")
    plt.figure(figsize=(10, 8))
    
    # shap.summary_plot automatically handles numpy arrays and pandas dataframes
    shap.summary_plot(
        shap_values, 
        X_test_scaled, 
        max_display=MAX_DISPLAY,
        show=False,
        plot_type="dot",  # 'dot' creates the beeswarm visualization
        color_bar_label="Feature Value (Standardized)"
    )
    
    # Format for publication
    fig = plt.gcf()
    fig.axes[0].set_title("A. Global Impact on Sepsis Mortality (Beeswarm)", fontsize=14, pad=20, weight='bold')
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=12)
    plt.tight_layout()
    plt.savefig(BEESWARM_OUT, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Generate SHAP Bar Plot (Global Importance)
    print("    -> Rendering Global Feature Importance Bar Plot...")
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values, 
        X_test_scaled, 
        max_display=MAX_DISPLAY,
        show=False,
        plot_type="bar",
        color="#8B0000"  # Dark red for mortality indication
    )
    
    fig = plt.gcf()
    fig.axes[0].set_title("B. Mean Absolute Feature Importance (Consensus)", fontsize=14, pad=20, weight='bold')
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output Magnitude)", fontsize=12)
    plt.tight_layout()
    plt.savefig(BAR_OUT, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n============================================================")
    print(" SHAP VISUALIZATIONS SUCCESSFULLY GENERATED")
    print("============================================================")
    print(f" -> Beeswarm Plot : {BEESWARM_OUT.relative_to(BASE_DIR)}")
    print(f" -> Bar Plot      : {BAR_OUT.relative_to(BASE_DIR)}")
    print("============================================================")

if __name__ == "__main__":
    main()