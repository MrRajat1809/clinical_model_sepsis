"""
03_plot_prognostic_landscape.py

Generates a multi-panel visualization of the PHATE manifold.

Features included:
- Overlays Cohort Source and Mortality to prove batch-correction and severity gradients.
- Maps specific physiological domains (Neurological, Renal/Metabolic, Coagulation) 
  identified by the RFECV and SHAP analyses onto the 2D space.
- Uses depth-sorting (`np.argsort`) so high-severity dots are rendered on top, 
  preventing them from being buried inside dense normal clusters.
- Automatically handles color scaling (95th percentile clipping) for heavy-tailed EHR variables.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]

# Inputs
PROCESSED_DIR_ATLAS = BASE_DIR / "data" / "processed" / "atlas"
PROCESSED_DIR_MIMIC = BASE_DIR / "data" / "processed" / "mimiciv"

ATLAS_FEATURES_FILE = PROCESSED_DIR_ATLAS / "atlas_sepsis_features_124.npy"
ATLAS_META_FILE = PROCESSED_DIR_ATLAS / "atlas_metadata.parquet"

OUT_FEATURES = BASE_DIR / "outputs" / "features"
PHATE_COORDS_FILE = OUT_FEATURES / "atlas_phate_coordinates.parquet"

# Outputs
OUT_FIGURES = BASE_DIR / "outputs" / "figures"
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

PLOT_FILE = OUT_FIGURES / "atlas_Prognostic_Landscape.png"

def get_feature_index(feature_names, search_term):
    """Helper to flexibly find feature indices across cohorts."""
    for i, name in enumerate(feature_names):
        if search_term.lower() in name.lower():
            return i
    return None

def main():
    print("[*] Initiating Phase 4: Prognostic Landscape Visualization...")
    start_time = time.time()

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    print("    -> Loading Coordinates, Features, and Metadata...")
    if not PHATE_COORDS_FILE.exists():
        print(f"[ERROR] Coordinates not found at {PHATE_COORDS_FILE}")
        return

    df_coords = pd.read_parquet(PHATE_COORDS_FILE)
    df_meta = pd.read_parquet(ATLAS_META_FILE)
    X_124 = np.load(ATLAS_FEATURES_FILE)
    
    # Load base feature names
    features_30 = list(np.load(PROCESSED_DIR_MIMIC / "mimic_sepsis_tensor_features.npy", allow_pickle=True))

    phate_x = df_coords["PHATE_1"].values
    phate_y = df_coords["PHATE_2"].values

    # ---------------------------------------------------------
    # 2. EXTRACT CLINICAL OVERLAYS
    # ---------------------------------------------------------
    print("    -> Extracting clinical domains for color mapping...")
    
    # Static & Target Variables
    cohort = df_meta["cohort_source"].values
    mortality = df_meta["hospital_expire_flag"].values
    baseline_sofa = df_meta["baseline_sofa"].values

    # Find indices for the 30 temporal base features
    idx_lactate = get_feature_index(features_30, "lactate")
    idx_bun = get_feature_index(features_30, "bun")
    idx_ptt = get_feature_index(features_30, "ptt")  # aPTT or PTT
    idx_motor = get_feature_index(features_30, "motor")
    idx_verbal = get_feature_index(features_30, "verbal")

    # In 124D: [0-29: Mean], [30-59: Min], [60-89: Max], [90-119: Std], [120-123: Static]
    # If a feature isn't found (fallback), we will map it to zeros.
    def extract_val(base_idx, offset):
        if base_idx is not None:
            return X_124[:, base_idx + offset]
        return np.zeros(len(X_124))

    val_lactate_mean = extract_val(idx_lactate, 0)       # Mean
    val_bun_max = extract_val(idx_bun, 60)               # Max
    val_ptt_mean = extract_val(idx_ptt, 0)               # Mean
    val_motor_min = extract_val(idx_motor, 30)           # Min (Worst GCS)
    val_verbal_std = extract_val(idx_verbal, 90)         # Std (Fluctuation)

    # ---------------------------------------------------------
    # 3. PLOT ARCHITECTURE
    # ---------------------------------------------------------
    print("    -> Generating 8-Panel Landscape...")
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.flatten()
    
    # Global Plot Settings
    point_size = 2
    alpha = 0.7
    bg_color = "#f8f9fa"
    fig.patch.set_facecolor('white')

    def plot_continuous(ax, title, values, cmap="magma", invert=False):
        """Helper to plot continuous variables with depth-sorting."""
        # Sort so highest values (usually most severe) are plotted on top
        if invert:
            order = np.argsort(-values) # Lowest on top
        else:
            order = np.argsort(values)  # Highest on top
            
        x_sorted, y_sorted, v_sorted = phate_x[order], phate_y[order], values[order]
        
        # Clip top 5% to prevent extreme EHR outliers from washing out the colormap
        vmax = np.percentile(values, 95)
        vmin = np.percentile(values, 1)
        
        sc = ax.scatter(x_sorted, y_sorted, c=v_sorted, cmap=cmap, 
                        s=point_size, alpha=alpha, vmin=vmin, vmax=vmax, edgecolors='none')
        
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(bg_color)
        sns.despine(ax=ax, left=True, bottom=True)
        return sc

    # Panel 0: Cohort (Categorical)
    ax = axes[0]
    mimic_mask = cohort == "MIMIC-IV"
    eicu_mask = cohort == "eICU-CRD"
    ax.scatter(phate_x[mimic_mask], phate_y[mimic_mask], c="#4C72B0", s=point_size, alpha=0.5, label="MIMIC-IV", edgecolors='none')
    ax.scatter(phate_x[eicu_mask], phate_y[eicu_mask], c="#C44E52", s=point_size, alpha=0.5, label="eICU-CRD", edgecolors='none')
    ax.set_title("A) Cohort Origin (Batch Correction)", fontweight="bold", fontsize=12)
    ax.legend(markerscale=5, loc="upper right")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(bg_color); sns.despine(ax=ax, left=True, bottom=True)

    # Panel 1: Mortality (Categorical)
    ax = axes[1]
    surv_mask = mortality == 0
    death_mask = mortality == 1
    # Plot survivors first, deaths on top
    ax.scatter(phate_x[surv_mask], phate_y[surv_mask], c="#74c476", s=point_size, alpha=0.4, label="Survived", edgecolors='none')
    ax.scatter(phate_x[death_mask], phate_y[death_mask], c="#2b8cbe", s=point_size, alpha=0.9, label="Died", edgecolors='none')
    ax.set_title("B) Sepsis Mortality (Prognostic Signal)", fontweight="bold", fontsize=12)
    ax.legend(markerscale=5, loc="upper right")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(bg_color); sns.despine(ax=ax, left=True, bottom=True)

    # Panels 2-7: Continuous Physiological Gradients
    sc2 = plot_continuous(axes[2], "C) Baseline SOFA Score", baseline_sofa, cmap="inferno")
    fig.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    sc3 = plot_continuous(axes[3], "D) Mean Lactate (Metabolic)", val_lactate_mean, cmap="plasma")
    fig.colorbar(sc3, ax=axes[3], fraction=0.046, pad=0.04)

    sc4 = plot_continuous(axes[4], "E) Max BUN (Renal)", val_bun_max, cmap="viridis")
    fig.colorbar(sc4, ax=axes[4], fraction=0.046, pad=0.04)

    sc5 = plot_continuous(axes[5], "F) Mean aPTT (Coagulation)", val_ptt_mean, cmap="cividis")
    fig.colorbar(sc5, ax=axes[5], fraction=0.046, pad=0.04)

    # Min Motor GCS is inverted (Lower is Worse, so we want low scores plotted on top and hot)
    sc6 = plot_continuous(axes[6], "G) Min GCS Motor (Neurological)", val_motor_min, cmap="rocket_r", invert=True)
    fig.colorbar(sc6, ax=axes[6], fraction=0.046, pad=0.04)

    sc7 = plot_continuous(axes[7], "H) Std GCS Verbal (Neuro Fluctuation)", val_verbal_std, cmap="mako")
    fig.colorbar(sc7, ax=axes[7], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n[+] Success! Landscape plotted in {elapsed:.2f} seconds.")
    print(f"    -> Plot saved to: {PLOT_FILE.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    main()