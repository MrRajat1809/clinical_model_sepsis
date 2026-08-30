"""
02_plot_phate_embedding.py

Applies PHATE to the fully imputed 3D time-series tensor to visualize the continuous
branching trajectories of sepsis subphenotypes in a 2D manifold.
"""

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import phate
import warnings

warnings.filterwarnings("ignore")

# Fixed rather than -1: thread count changes the order of floating-point
# accumulation, so "all cores" makes results depend on the machine.
N_JOBS = 8
BASE_DIR = Path(__file__).resolve().parents[2]
TENSOR_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "tensors"
CLUSTER_DIR = BASE_DIR / "data" / "processed" / "mimiciv" / "clusters"
VIS_DIR = BASE_DIR / "outputs" / "figures" / "mimiciv"

def plot_phate():
    print("[*] Generating PHATE Trajectory Embedding...")
    start_time = time.time()
    
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    tensor_file = TENSOR_DIR / "sepsis_imputed_tensor.npy"
    cluster_file = CLUSTER_DIR / "dtw_temporal_clusters.parquet"
    id_file = TENSOR_DIR / "sepsis_tensor_stay_ids.npy"
    
    if not tensor_file.exists() or not cluster_file.exists():
        print(f"[ERROR] Required files not found in {TENSOR_DIR} or {CLUSTER_DIR}")
        return

    print("    -> Loading imputed 3D tensor and cluster labels...")
    X_3d = np.load(tensor_file)
    stay_ids = np.load(id_file)
    df_clusters = pl.read_parquet(cluster_file)
    
    num_patients, n_steps, n_features = X_3d.shape
    
    # Flatten the temporal dimension to treat the entire 24-hour movie as a single state
    print("    -> Flattening tensor for manifold learning...")
    X_flat = X_3d.reshape(num_patients, n_steps * n_features)
    
    # Ensure the cluster labels align perfectly with the tensor patient IDs
    df_ids = pl.DataFrame({"stay_id": stay_ids})
    aligned_clusters = df_ids.join(df_clusters, on="stay_id", how="left")["temporal_cluster"].to_numpy()
    
    # Cluster naming map based on our clinical trajectory findings
    cluster_names = {
        0: "Cluster 0 (Rapid Responders)",
        1: "Cluster 1 (Renal/Tachycardic)",
        2: "Cluster 2 (Resolving Shock)",
        3: "Cluster 3 (Slow Crashers)"
    }
    
    print("    -> Initializing and fitting PHATE (this may take a moment)...")
    phate_operator = phate.PHATE(
        n_jobs=N_JOBS,
        random_state=42,
        verbose=False,
        t='auto',     # Automatically determine diffusion time
        knn=5,        # Focus on local neighborhoods for tight branching
        decay=40      # Default alpha decay
    )
    
    # Fit and transform the high-dimensional flattened trajectories
    Y_phate = phate_operator.fit_transform(X_flat)
    
    print("    -> Rendering embedding plot...")
    df_phate = pd.DataFrame(Y_phate, columns=["PHATE 1", "PHATE 2"])
    df_phate["Cluster"] = [cluster_names.get(c, f"Cluster {c}") for c in aligned_clusters]
    
    plt.figure(figsize=(12, 10), dpi=300)
    sns.set_theme(style="white", context="paper")
    
    # Match colors from the previous clinical trajectory plots
    palette = {
        "Cluster 0 (Rapid Responders)": "#1f77b4", # Blue
        "Cluster 1 (Renal/Tachycardic)": "#d62728", # Red
        "Cluster 2 (Resolving Shock)": "#2ca02c",   # Green
        "Cluster 3 (Slow Crashers)": "#ff7f0e"      # Orange
    }
    
    sns.scatterplot(
        data=df_phate,
        x="PHATE 1",
        y="PHATE 2",
        hue="Cluster",
        palette=palette,
        s=15,
        alpha=0.7,
        edgecolor="none"
    )
    
    plt.title("PHATE Embedding of 24-Hour Sepsis Trajectories", fontsize=16, fontweight="bold")
    plt.xlabel("PHATE 1", fontsize=12)
    plt.ylabel("PHATE 2", fontsize=12)
    plt.legend(title="DTW Subphenotypes", loc="best", frameon=True)
    
    sns.despine()
    
    out_file = VIS_DIR / "phate_trajectory_embedding.png"
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    
    elapsed = time.time() - start_time
    print(f"[+] Success! PHATE embedding generated in {elapsed:.2f} seconds.")
    print(f"    -> Saved to: {out_file.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    plot_phate()