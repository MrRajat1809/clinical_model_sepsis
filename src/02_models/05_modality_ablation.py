"""
Which modality carries the prognostic signal.

Fits the primary model's hyperparameters, held fixed, to seven combinations of
three feature sources: static variables, engineered temporal summaries, and the
deep BiGRU embeddings. Holding the hyperparameters constant means any difference
is attributable to the inputs rather than to tuning.

Combinations: each source alone, each pair, and all three together.

Significance is assessed by paired bootstrap, 1000 resamples drawn once and
applied identically to every model's predictions, so the comparisons share
resampling noise. Two-sided empirical p-values come from the sign of the paired
AUROC difference against the primary model.

Reads:
    outputs/metrics/mimic_champion_metrics.json for the hyperparameters
    outputs/features/mimic_temporal_bigru_embeddings.npy
Writes:
    outputs/metrics/mimic_modality_ablation_results.csv
    per-combination models, predictions and a summary figure
"""

import time
import json
import joblib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# --- Configuration & Reproducibility -------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed" / "mimiciv"

# Inputs from previous scripts mapped to flattened structure
CHAMPION_METRICS_FILE = BASE_DIR / "outputs" / "metrics" / "mimic_champion_metrics.json"
LATENT_EMBEDDINGS_FILE = BASE_DIR / "outputs" / "features" / "mimic_temporal_bigru_embeddings.npy"

OUT_MODELS = BASE_DIR / "outputs" / "models"
OUT_PREDS = BASE_DIR / "outputs" / "predictions"
OUT_METRICS = BASE_DIR / "outputs" / "metrics"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"

for d in [OUT_MODELS, OUT_PREDS, OUT_METRICS, OUT_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOTSTRAPS = 1000
CHAMPION_NAME = "Static + Aggregated (Champion)"

def set_seed(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_p_value(dist_modality, dist_champion):
    """Computes two-tailed empirical p-value from paired bootstrap distributions."""
    delta = np.array(dist_modality) - np.array(dist_champion)
    p_value = 2 * min(np.mean(delta > 0), np.mean(delta < 0))
    return min(p_value, 1.0)

# --- Main Execution ------------------------------------------------------
def main():
    set_seed(RANDOM_STATE)
    print("[*] Initiating Phase 5: Modality Ablation Study & Paired Bootstrapping...")
    start_time = time.time()
    
    # --- Load Champion Hyperparameters -----------------------------------
    if not CHAMPION_METRICS_FILE.exists():
        print(f"[ERROR] Champion config not found at {CHAMPION_METRICS_FILE}.")
        return
    with open(CHAMPION_METRICS_FILE, "r") as f:
        champion_config = json.load(f)["hyperparameters"]
        
    print("    -> Loaded Champion Hyperparameters to ensure fair evaluation.")

    # --- Load Data & Splits ----------------------------------------------
    X_imputed = np.load(PROCESSED_DIR / "mimic_sepsis_imputed_tensor.npy")
    stay_ids = np.load(PROCESSED_DIR / "mimic_sepsis_tensor_stay_ids.npy")
    
    df_cohort = pl.read_parquet(PROCESSED_DIR / "mimic_final_sepsis3_cohort.parquet").to_pandas()
    df_cohort = pd.DataFrame({"stay_id": stay_ids}).merge(df_cohort, on="stay_id", how="left")
    y = df_cohort["hospital_expire_flag"].values
    
    idx_train_val = np.load(OUT_MODELS / "mimic_train_indices.npy")
    idx_test = np.load(OUT_MODELS / "mimic_test_set_indices.npy")
    stay_ids_test = np.load(OUT_MODELS / "mimic_stay_ids_test.npy")

    # --- Extract Individual Modalities -----------------------------------
    print("    -> Extracting Modality 1: Static Features...")
    static_cols = [col for col in ["age", "baseline_sofa"] if col in df_cohort.columns]
    df_static = df_cohort[static_cols].copy()
    
    scaler_static = StandardScaler()
    scaler_static.fit(df_static.fillna(0).values[idx_train_val])
    X_static = scaler_static.transform(df_static.fillna(0).values)

    print("    -> Extracting Modality 2: Aggregated Temporal Features...")
    X_mean = np.mean(X_imputed, axis=1)
    X_min = np.min(X_imputed, axis=1)
    X_max = np.max(X_imputed, axis=1)
    X_std = np.std(X_imputed, axis=1)
    
    scaler_agg = StandardScaler()
    scaler_agg.fit(np.concatenate([X_mean, X_min, X_max, X_std], axis=1)[idx_train_val])
    X_temporal_agg = scaler_agg.transform(np.concatenate([X_mean, X_min, X_max, X_std], axis=1))

    print("    -> Loading Modality 3: Deep BiGRU Latent Embeddings...")
    if not LATENT_EMBEDDINGS_FILE.exists():
        print(f"[ERROR] Latent embeddings not found at {LATENT_EMBEDDINGS_FILE}.")
        return
    X_temporal_emb = np.load(LATENT_EMBEDDINGS_FILE)

    # --- Define Ablation Combinations ------------------------------------
    combinations = {
        "Single Modality": {
            "Static Only": X_static,
            "Aggregated Only": X_temporal_agg,
            "Latents Only": X_temporal_emb
        },
        "Dual Modality": {
            CHAMPION_NAME: np.concatenate([X_static, X_temporal_agg], axis=1),
            "Static + Latents": np.concatenate([X_static, X_temporal_emb], axis=1),
            "Aggregated + Latents": np.concatenate([X_temporal_agg, X_temporal_emb], axis=1)
        },
        "Three Modality": {
            "SuperStack": np.concatenate([X_static, X_temporal_agg, X_temporal_emb], axis=1)
        }
    }

    scale_weight = float((len(y[idx_train_val]) - sum(y[idx_train_val])) / sum(y[idx_train_val]))
    champion_config.update({"scale_pos_weight": scale_weight, "random_state": RANDOM_STATE, "n_jobs": -1})

    # --- Train & Generate Predictions ------------------------------------
    print("\n    -> Training Modalities using Champion Hyperparameters...")
    
    all_test_preds = {}
    feature_counts = {}
    
    for category, models in combinations.items():
        for name, X_combo in models.items():
            print(f"        - Fitting {name} ({X_combo.shape[1]} features)...")
            X_train, y_train = X_combo[idx_train_val], y[idx_train_val]
            X_test, y_test = X_combo[idx_test], y[idx_test]
            
            model = XGBClassifier(**champion_config)
            model.fit(X_train, y_train)
            
            preds = model.predict_proba(X_test)[:, 1]
            all_test_preds[name] = preds
            feature_counts[name] = X_combo.shape[1]
            
            # Export Models & Predictions
            safe_name = name.replace(" ", "_").replace("+", "and").replace("(", "").replace(")", "")
            joblib.dump(model, OUT_MODELS / f"mimic_ablation_{safe_name}.joblib")
            
            pd.DataFrame({
                "stay_id": stay_ids_test,
                "true_label": y_test,
                "pred_probability": preds
            }).to_csv(OUT_PREDS / f"mimic_ablation_{safe_name}_predictions.csv", index=False)

    # --- Paired Bootstrap Statistical Testing ----------------------------
    print(f"\n    -> Running {N_BOOTSTRAPS} Paired Bootstrap Iterations...")
    rng = np.random.default_rng(RANDOM_STATE)
    test_size = len(y_test)
    
    bootstrap_results = {name: {"auroc": [], "auprc": [], "brier": []} for name in all_test_preds.keys()}
    
    for _ in range(N_BOOTSTRAPS):
        idx = rng.choice(test_size, size=test_size, replace=True)
        y_true_b = y_test[idx]
        
        if len(np.unique(y_true_b)) < 2: continue
            
        for name, preds in all_test_preds.items():
            y_pred_b = preds[idx]
            bootstrap_results[name]["auroc"].append(roc_auc_score(y_true_b, y_pred_b))
            bootstrap_results[name]["auprc"].append(average_precision_score(y_true_b, y_pred_b))
            bootstrap_results[name]["brier"].append(brier_score_loss(y_true_b, y_pred_b))

    # Compile Final Leaderboard with Deltas & p-values
    leaderboard = []
    champ_auroc_dist = bootstrap_results[CHAMPION_NAME]["auroc"]
    champ_base_auroc = np.mean(champ_auroc_dist)
    
    for category, models in combinations.items():
        for name in models.keys():
            auroc_dist = bootstrap_results[name]["auroc"]
            auprc_dist = bootstrap_results[name]["auprc"]
            brier_dist = bootstrap_results[name]["brier"]
            
            mean_auroc = np.mean(auroc_dist)
            ci_auroc = (np.percentile(auroc_dist, 2.5), np.percentile(auroc_dist, 97.5))
            
            mean_auprc = np.mean(auprc_dist)
            ci_auprc = (np.percentile(auprc_dist, 2.5), np.percentile(auprc_dist, 97.5))
            
            mean_brier = np.mean(brier_dist)
            ci_brier = (np.percentile(brier_dist, 2.5), np.percentile(brier_dist, 97.5))
            
            delta_auroc = mean_auroc - champ_base_auroc
            pct_improvement = (mean_auroc - np.mean(bootstrap_results["Static Only"]["auroc"])) / np.mean(bootstrap_results["Static Only"]["auroc"]) * 100
            
            p_val = compute_p_value(auroc_dist, champ_auroc_dist) if name != CHAMPION_NAME else 1.0
            
            leaderboard.append({
                "Category": category,
                "Modality": name,
                "Features": feature_counts[name],
                "AUROC": mean_auroc,
                "AUROC_95CI_Lower": ci_auroc[0],
                "AUROC_95CI_Upper": ci_auroc[1],
                "Δ_AUROC_from_Champion": delta_auroc,
                "Gain_from_Static": pct_improvement,
                "P_Value_vs_Champion": p_val,
                "AUPRC": mean_auprc,
                "Brier": mean_brier
            })

    df_results = pd.DataFrame(leaderboard)
    df_results.to_csv(OUT_METRICS / "mimic_modality_ablation_results.csv", index=False)
    
    with open(OUT_METRICS / "mimic_ablation_metadata.json", "w") as f:
        json.dump({"n_bootstraps": N_BOOTSTRAPS, "random_seed": RANDOM_STATE, "champion_used": CHAMPION_NAME}, f, indent=4)

    # --- Generate Publication Plot ---------------------------------------
    print("    -> Generating horizontal bar plot...")
    df_plot = df_results.sort_values("AUROC", ascending=True).copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df_plot["Modality"], df_plot["AUROC"], xerr=[df_plot["AUROC"] - df_plot["AUROC_95CI_Lower"], df_plot["AUROC_95CI_Upper"] - df_plot["AUROC"]], 
                   capsize=5, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Highlight Champion
    for i, bar in enumerate(bars):
        if df_plot.iloc[i]["Modality"] == CHAMPION_NAME:
            bar.set_color('firebrick')
            bar.set_edgecolor('black')
            
    ax.set_xlabel("AUROC (95% CI)")
    ax.set_title("Modality Ablation Study: Impact on Mortality Discrimination")
    ax.set_xlim([0.70, 0.92])
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUT_FIGURES / "mimic_ablation_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Display Leaderboard ---------------------------------------------
    print("\n" + "="*95)
    print(" MODALITY ABLATION STUDY RESULTS (PAIRED BOOTSTRAP VS CHAMPION)")
    print("="*95)
    for category in df_results["Category"].unique():
        print(f"\n [{category}]")
        cat_df = df_results[df_results["Category"] == category]
        for _, row in cat_df.iterrows():
            marker = "*" if row["Modality"] == CHAMPION_NAME else " "
            p_text = f"p={row['P_Value_vs_Champion']:.3f}" if row['Modality'] != CHAMPION_NAME else "Reference "
            print(f"{marker} {row['Modality']:<30} | {row['Features']:<3} feats | "
                  f"AUROC: {row['AUROC']:.3f} [{row['AUROC_95CI_Lower']:.3f}-{row['AUROC_95CI_Upper']:.3f}] | "
                  f"Δ: {row['Δ_AUROC_from_Champion']:+.3f} | {p_text}")
    print("="*95)
    
    elapsed = time.time() - start_time
    print(f"[*] Ablation study completed in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()
