"""
Embed and annotate the eICU severity manifold.

Runs PHATE on the precomputed severity DTW distance matrix with the same eight
overlays as the shape manifold. Read against 08b, the difference between the two
embeddings is attributable to magnitude, which is the only thing the two
distance metrics treat differently.

Reads:
    outputs/features/eicu_dtw_severity_pairwise_distance_matrix.npy
    eicu_final_sepsis3_cohort.parquet, imputed and raw tensors
Writes:
    outputs/features/eicu_phate_severity_coordinates.parquet
    outputs/figures/eicu_Severity_Trajectory_Manifold.png
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import phate

import warnings
warnings.filterwarnings("ignore")

# --- Configuration -------------------------------------------------------
# Fixed rather than -1: thread count changes the order of floating-point
# accumulation, so "all cores" makes results depend on the machine.
N_JOBS = 8
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed" / "eicu"

OUT_FEATS = BASE_DIR / "outputs" / "features"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

def run_atlas_characterization():
    print("[*] Initializing eICU PHATE Manifold Projection (Clinical Severity Space)...")
    start_time = time.time()
    
    OUT_FEATS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    
    # Inputs (from 09a and earlier)
    dist_matrix_file = OUT_FEATS / "eicu_dtw_severity_pairwise_distance_matrix.npy"
    atlas_ids_file = OUT_FEATS / "eicu_severity_atlas_stay_ids.npy"
    cohort_file = PROCESSED_DIR / "eicu_final_sepsis3_cohort.parquet"
    tensor_file = PROCESSED_DIR / "eicu_sepsis_imputed_tensor.npy"
    raw_tensor_file = PROCESSED_DIR / "eicu_sepsis_tensor_raw.npy"  
    features_file = PROCESSED_DIR / "eicu_sepsis_tensor_features.npy"
    
    # Outputs (Explicitly named with "severity")
    atlas_coords_file = OUT_FEATS / "eicu_phate_severity_coordinates.parquet"
    plot_file = OUT_FIGURES / "eicu_Severity_Trajectory_Manifold.png"
    
    if not dist_matrix_file.exists():
        print(f"[ERROR] Distance matrix not found at {dist_matrix_file}")
        return

    # 1. Load the DTW Matrix and IDs
    print("    -> Loading eICU DTW Distance Matrix and Metadata...")
    dtw_matrix = np.load(dist_matrix_file)
    stay_ids = np.load(atlas_ids_file)
    
    # 2. Extract Clinical Metadata
    print("    -> Extracting static cohort metadata...")
    df_cohort = pl.read_parquet(cohort_file).select([
        "stay_id", "hospital_expire_flag", "baseline_sofa", "age"
    ]).to_pandas()
    
    print("    -> Extracting dynamic trajectory bounds (Lactate, NEQ, P/F, Urine, Vent)...")
    X_imputed = np.load(tensor_file)
    X_raw = np.load(raw_tensor_file)  
    features = list(np.load(features_file))
    
    # Extract dynamic extremes for each patient over the 24h window (Imputed)
    max_lactate = np.max(X_imputed[:, :, features.index("lactate")], axis=1)
    max_neq = np.max(X_imputed[:, :, features.index("neq")], axis=1)
    min_pf = np.min(X_imputed[:, :, features.index("pf_ratio")], axis=1)
    total_urine = np.sum(X_imputed[:, :, features.index("urine_output")], axis=1)
    
    # Extract ventilation from RAW data. nansum > 0 means they had at least one '1' recorded.
    vent_raw = X_raw[:, :, features.index("vent")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        any_vent = np.nansum(vent_raw, axis=1) > 0
    
    df_metadata = pd.DataFrame({
        "stay_id": stay_ids,
        "max_lactate": max_lactate,
        "max_neq": max_neq,
        "min_pf": min_pf,
        "total_urine": total_urine,
        "ventilation": np.where(any_vent, "Yes", "No") 
    })
    
    # Merge ensuring the exact order of stay_ids is preserved
    df_atlas = pd.DataFrame({"stay_id": stay_ids})
    df_atlas = df_atlas.merge(df_cohort, on="stay_id", how="left")
    df_atlas = df_atlas.merge(df_metadata, on="stay_id", how="left")
    assert (df_atlas["stay_id"].values == stay_ids).all(), "Order mismatch during merge!"

    # 3. Fit PHATE
    print("    -> Fitting PHATE embedding (knn_dist='precomputed')...")
    phate_operator = phate.PHATE(
        knn_dist='precomputed',
        n_components=2,
        knn=5,          
        decay=40,       
        t='auto',       
        n_jobs=N_JOBS,
        random_state=42,
        verbose=False
    )
    
    phate_coords = phate_operator.fit_transform(dtw_matrix)
    df_atlas["PHATE_1"] = phate_coords[:, 0]
    df_atlas["PHATE_2"] = phate_coords[:, 1]
    
    print("    -> Serializing eICU Atlas coordinates...")
    df_atlas.to_parquet(atlas_coords_file, index=False)

    # 4. Generate Publication-Quality 8-Panel Figure
    print("    -> Generating 8-panel manifold characterization figure...")
    
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    point_size = 10
    point_alpha = 0.7
    
    # Panel 1: Mortality
    df_sorted = df_atlas.sort_values("hospital_expire_flag")
    sns.scatterplot(
        data=df_sorted, x="PHATE_1", y="PHATE_2", hue="hospital_expire_flag", 
        palette={0: "#4C72B0", 1: "#C44E52"}, s=point_size, alpha=point_alpha, edgecolor=None, ax=axes[0]
    )
    axes[0].set_title("A. eICU Hospital Mortality", fontsize=14, weight='bold')
    axes[0].legend(title="Mortality", labels=["Survivor", "Non-Survivor"])

    # Panel 2: Baseline SOFA
    df_sorted = df_atlas.sort_values("baseline_sofa")
    scatter = axes[1].scatter(
        df_sorted["PHATE_1"], df_sorted["PHATE_2"],
        c=df_sorted["baseline_sofa"], cmap="viridis", s=point_size, alpha=point_alpha, edgecolor=None
    )
    axes[1].set_title("B. Baseline SOFA Score", fontsize=14, weight='bold')
    plt.colorbar(scatter, ax=axes[1])

    # Panel 3: Max Lactate
    df_atlas["plot_lactate"] = df_atlas["max_lactate"].clip(upper=10)
    df_sorted = df_atlas.sort_values("plot_lactate")
    scatter = axes[2].scatter(
        df_sorted["PHATE_1"], df_sorted["PHATE_2"],
        c=df_sorted["plot_lactate"], cmap="magma", s=point_size, alpha=point_alpha, edgecolor=None
    )
    axes[2].set_title("C. Max Lactate (mmol/L)", fontsize=14, weight='bold')
    plt.colorbar(scatter, ax=axes[2])

    # Panel 4: Max NEQ
    df_atlas["plot_neq"] = df_atlas["max_neq"].clip(upper=0.5)
    df_sorted = df_atlas.sort_values("plot_neq")
    scatter = axes[3].scatter(
        df_sorted["PHATE_1"], df_sorted["PHATE_2"],
        c=df_sorted["plot_neq"], cmap="inferno", s=point_size, alpha=point_alpha, edgecolor=None
    )
    axes[3].set_title("D. Max Vasopressor Dose (NEQ)", fontsize=14, weight='bold')
    plt.colorbar(scatter, ax=axes[3])

    # Panel 5: Min P/F Ratio
    df_atlas["plot_pf"] = df_atlas["min_pf"].clip(lower=50, upper=400)
    df_sorted = df_atlas.sort_values("plot_pf", ascending=False)
    scatter = axes[4].scatter(
        df_sorted["PHATE_1"], df_sorted["PHATE_2"],
        c=df_sorted["plot_pf"], cmap="Spectral", s=point_size, alpha=point_alpha, edgecolor=None
    )
    axes[4].set_title("E. Worst P/F Ratio (Resp Failure)", fontsize=14, weight='bold')
    plt.colorbar(scatter, ax=axes[4])

    # Panel 6: Total 24h Urine Output
    df_atlas["plot_urine"] = df_atlas["total_urine"].clip(upper=3000)
    df_sorted = df_atlas.sort_values("plot_urine", ascending=False)
    scatter = axes[5].scatter(
        df_sorted["PHATE_1"], df_sorted["PHATE_2"],
        c=df_sorted["plot_urine"], cmap="YlGnBu_r", s=point_size, alpha=point_alpha, edgecolor=None
    )
    axes[5].set_title("F. Total 24h Urine Output (mL)", fontsize=14, weight='bold')
    plt.colorbar(scatter, ax=axes[5])

    # Panel 7: Ventilation (Shuffled for fair overlap, using muted contrasting palette)
    df_shuffled = df_atlas.sample(frac=1, random_state=42)
    sns.scatterplot(
        data=df_shuffled, x="PHATE_1", y="PHATE_2", hue="ventilation", 
        palette={"No": "#55A868", "Yes": "#8172B3"}, s=point_size, alpha=0.6, edgecolor=None, ax=axes[6]
    )
    axes[6].set_title("G. Mechanical Ventilation", fontsize=14, weight='bold')
    axes[6].legend(title="Ventilated")

    # Panel 8: Age
    df_sorted = df_atlas.sort_values("age")
    scatter = axes[7].scatter(
        df_sorted["PHATE_1"], df_sorted["PHATE_2"],
        c=df_sorted["age"], cmap="coolwarm", s=point_size, alpha=point_alpha, edgecolor=None
    )
    axes[7].set_title("H. Patient Age", fontsize=14, weight='bold')
    plt.colorbar(scatter, ax=axes[7])

    # Clean up all axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PHATE 1")
        ax.set_ylabel("PHATE 2")
        sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    
    elapsed = time.time() - start_time
    print(f"\n[+] Success! eICU Manifold embedded and multi-panel clinical figure generated.")
    print(f"    -> Atlas Coordinates: {atlas_coords_file.relative_to(BASE_DIR)}")
    print(f"    -> Atlas Figure: {plot_file.relative_to(BASE_DIR)}")
    print(f"    -> Total Execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    run_atlas_characterization()
