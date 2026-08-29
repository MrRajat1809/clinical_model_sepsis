"""
Compare pre-imputation recording density between the two cohorts.

Measures how much real data each database actually contains for each of the 30
variables, before any reconstruction. Density is the share of patient-hour cells
carrying an observation, computed on the raw tensors, so it reflects charting
practice rather than what the imputation model produced.

This is the evidence behind two claims in the manuscript: that the databases
record different things at different rates, and that the absent-scores-zero
convention in the SOFA calculators therefore affects them unequally. A feature
with high MIMIC-IV density and near-zero eICU density cannot transport, whatever
the model does with it.

Asserts that the two feature orders match before comparing, since a silent
reorder would make every row meaningless.

Reads:
    mimic_sepsis_tensor_raw.npy, eicu_sepsis_tensor_raw.npy, and feature names
Writes:
    outputs/metrics/feature_parity_density.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]

# MIMIC-IV Tensor Paths (Flattened)
MIMIC_DIR = BASE_DIR / "data" / "processed" / "mimiciv"
MIMIC_TENSOR = MIMIC_DIR / "mimic_sepsis_tensor_raw.npy"
MIMIC_FEATS = MIMIC_DIR / "mimic_sepsis_tensor_features.npy"

# eICU Tensor Paths (Flattened)
EICU_DIR = BASE_DIR / "data" / "processed" / "eicu"
EICU_TENSOR = EICU_DIR / "eicu_sepsis_tensor_raw.npy"
EICU_FEATS = EICU_DIR / "eicu_sepsis_tensor_features.npy"

def run_audit():
    print("[*] Building Final Tensor-Level Feature Parity Matrix...")
    
    if not MIMIC_TENSOR.exists() or not EICU_TENSOR.exists():
        print(f"[ERROR] Tensors not found.\nChecked MIMIC: {MIMIC_TENSOR}\nChecked eICU: {EICU_TENSOR}\nEnsure tensor builders have been run.")
        return

    # Load 3D Tensors: Shape = [Patients, 24 Hours, Features]
    m_X = np.load(MIMIC_TENSOR)
    e_X = np.load(EICU_TENSOR)
    
    m_feats = list(np.load(MIMIC_FEATS))
    e_feats = list(np.load(EICU_FEATS))
    
    # Assert features match exactly
    if m_feats != e_feats:
        print("[ERROR] Feature order mismatch between MIMIC and eICU tensors!")
        return
        
    m_pts, m_steps, m_f_count = m_X.shape
    e_pts, e_steps, e_f_count = e_X.shape
    
    m_total_cells = m_pts * m_steps
    e_total_cells = e_pts * e_steps

    print(f"    -> MIMIC-IV : {m_pts:,} patients | {m_total_cells:,} total hourly cells")
    print(f"    -> eICU     : {e_pts:,} patients | {e_total_cells:,} total hourly cells\n")

    print("=========================================================================================")
    print(" TRUE TENSOR DENSITY AUDIT (Pre-Imputation 24-Hour Bins)")
    print("=========================================================================================")
    print(f"{'Feature':<15} | {'MIMIC N (Hrs)':<15} | {'eICU N (Hrs)':<15} | {'M_Dens%':<7} | {'e_Dens%':<7} | {'Δ Diff%':<8}")
    print("-" * 89)

    parity_rows = []
    for i, feature in enumerate(m_feats):
        # Extract the specific feature slice across all patients and all 24 hours
        m_slice = m_X[:, :, i]
        e_slice = e_X[:, :, i]
        
        # Count non-NaN values
        m_n = int(np.sum(~np.isnan(m_slice)))
        e_n = int(np.sum(~np.isnan(e_slice)))
        
        # Calculate Density (Coverage)
        m_dens = (m_n / m_total_cells) * 100
        e_dens = (e_n / e_total_cells) * 100
        
        # Absolute difference in density
        d_diff = abs(m_dens - e_dens)
        
        # Flag major discrepancies (e.g., completely missing or >30% drift)
        flag = "🚨" if e_dens == 0.0 or d_diff > 30.0 else ""
        
        parity_rows.append({"feature": feature, "mimic_n_hours": m_n, "eicu_n_hours": e_n,
                            "mimic_density_pct": round(m_dens, 2), "eicu_density_pct": round(e_dens, 2),
                            "abs_density_diff_pct": round(d_diff, 2)})
        print(f"{feature.upper():<15} | {m_n:<15,} | {e_n:<15,} | {m_dens:>6.2f}% | {e_dens:>6.2f}% | {d_diff:>6.2f}% {flag}")
        
    # Print overall tensor sparsity
    m_overall = np.mean(~np.isnan(m_X)) * 100
    e_overall = np.mean(~np.isnan(e_X)) * 100
    
    print("-" * 89)
    import pandas as _pd
    _metrics = BASE_DIR / "outputs" / "metrics"
    _metrics.mkdir(parents=True, exist_ok=True)
    parity_rows.append({"feature": "OVERALL", "mimic_n_hours": None, "eicu_n_hours": None,
                        "mimic_density_pct": round(m_overall, 2), "eicu_density_pct": round(e_overall, 2),
                        "abs_density_diff_pct": round(abs(m_overall - e_overall), 2)})
    _pd.DataFrame(parity_rows).to_csv(_metrics / "feature_parity_density.csv", index=False)

    print(f"{'OVERALL DENSITY':<15} | {'-':<15} | {'-':<15} | {m_overall:>6.2f}% | {e_overall:>6.2f}% | {abs(m_overall-e_overall):>6.2f}%")
    print("=========================================================================================")

if __name__ == "__main__":
    run_audit()
