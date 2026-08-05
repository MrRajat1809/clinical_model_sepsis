"""
01_plot_clinical_trajectories.py

Visualizes the dynamic physiological progression of the 4 DTW clusters over the first 24 hours.
Plots standard clinical variables (MAP, Lactate, Heart Rate, Creatinine) using Seaborn 
to calculate mean trends and 95% confidence intervals.
"""

import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"
CLUSTER_DIR = PROCESSED_DIR / "clusters"
VIS_DIR = BASE_DIR / "outputs" / "figures" / "mimiciv"

def plot_temporal_clusters():
    print("[*] Generating Clinical Trajectory Plots...")
    start_time = time.time()
    
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    cluster_file = CLUSTER_DIR / "dtw_temporal_clusters.parquet"
    temporal_file = PROCESSED_DIR / "sepsis_temporal_data_cleaned.parquet"
    cohort_file = PROCESSED_DIR / "final_sepsis3_cohort.parquet"
    
    if not cluster_file.exists():
        print(f"[ERROR] Cluster file not found at {cluster_file}")
        return

    print("    -> Loading data and merging clusters...")
    df_clusters = pl.read_parquet(cluster_file)
    df_cohort = pl.read_parquet(cohort_file).select(["stay_id", "sepsis_onset_time"])
    df_vitals = pl.scan_parquet(temporal_file)
    
    # Define the 4 key variables we want to visualize
    target_itemids = {
        220181: "MAP (mmHg)", 220052: "MAP (mmHg)",         # Mean Arterial Pressure
        50813: "Lactate (mmol/L)", 227442: "Lactate (mmol/L)", # Lactate
        220045: "Heart Rate (bpm)",                         # Heart Rate
        50912: "Creatinine (mg/dL)", 220615: "Creatinine (mg/dL)" # Creatinine
    }
    
    mapping_df = pl.DataFrame({
        "itemid": list(target_itemids.keys()),
        "feature": list(target_itemids.values())
    }, schema={"itemid": pl.Int64, "feature": pl.Utf8})

    print("    -> Extracting 24-hour trajectories...")
    # Join vitals with onset times, filter itemids, and calculate hours
    df_joined = df_vitals.join(mapping_df.lazy(), on="itemid", how="inner").join(
        df_cohort.lazy(), on="stay_id", how="inner"
    ).join(
        df_clusters.lazy(), on="stay_id", how="inner"
    )
    
    # Filter to the first 24 hours and bin by hour for smoother plotting
    df_plot_data = df_joined.with_columns(
        ((pl.col("event_time") - pl.col("sepsis_onset_time")).dt.total_hours()).alias("hours_from_onset")
    ).filter(
        (pl.col("hours_from_onset") >= 0) & (pl.col("hours_from_onset") <= 24)
    ).with_columns(
        pl.col("hours_from_onset").round(0).alias("hour_bin")
    ).collect().to_pandas()

    print("    -> Rendering Seaborn plots...")
    sns.set_theme(style="whitegrid", palette="tab10")
    
    # Map cluster names based on our previous mortality findings
    cluster_labels = {
        0: "Cluster 0 (Rapid Responders)",
        1: "Cluster 1",
        2: "Cluster 2",
        3: "Cluster 3 (Slow Crashers)"
    }
    df_plot_data["Cluster"] = df_plot_data["temporal_cluster"].map(cluster_labels).fillna("Cluster " + df_plot_data["temporal_cluster"].astype(str))

    # Create a 2x2 figure grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()
    
    features_to_plot = ["MAP (mmHg)", "Lactate (mmol/L)", "Heart Rate (bpm)", "Creatinine (mg/dL)"]
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        subset = df_plot_data[df_plot_data["feature"] == feature]
        
        sns.lineplot(
            data=subset, 
            x="hour_bin", 
            y="valuenum", 
            hue="Cluster", 
            errorbar=("ci", 95), 
            linewidth=2.5,
            ax=ax
        )
        
        ax.set_title(f"Trajectory of {feature}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Hours from Sepsis Onset", fontsize=12)
        ax.set_ylabel(feature, fontsize=12)
        ax.set_xlim(0, 24)
        
        # Add a subtle vertical line at Time Zero
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        
        if i == 0:
            ax.legend(title="DTW Subphenotypes", loc="upper right")
        else:
            ax.legend_.remove() # Only show legend on the first plot to save space

    plt.tight_layout()
    
    out_file = VIS_DIR / "dtw_clinical_trajectories.png"
    plt.savefig(out_file, bbox_inches="tight")
    
    elapsed = time.time() - start_time
    print(f"[+] Success! Plots generated in {elapsed:.2f} seconds.")
    print(f"    -> Saved to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    plot_temporal_clusters()