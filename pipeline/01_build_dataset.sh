#!/usr/bin/env bash
#
# build_dataset.sh - regenerate the ClinicalTensorSepsis release from source.
#
# Runs only the scripts needed to produce the released dataset. No model is
# trained for the paper here and no result is computed; this exists so that a
# reader with MIMIC-IV and eICU-CRD can rebuild the published files without
# reproducing the analysis.
#
# 01_ml_baselines.py is included for one reason: it draws the development and
# held-out partition that features_model.csv.gz records. It also fits four
# benchmark models, which takes about fifteen seconds and writes to outputs/;
# none of that enters the release.
#
# Scripts run, in dependency order:
#   MIMIC-IV cohort, phenotype, tensor, imputation model      9
#   eICU-CRD cohort, phenotype, tensor, imputation inference  16
#   partition assignment                                       1
#   cross-database alignment and its quality control           2
#   export                                                     1
#
# For reproducing the paper instead, use pipeline/run_pipeline.sh.
#
# Usage:
#   bash pipeline/build_dataset.sh
#   bash pipeline/build_dataset.sh --dry-run
#   bash pipeline/build_dataset.sh --outdir release_v1
#
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
LOG_DIR="outputs/logs"
LOG="$LOG_DIR/build_dataset.log"
OUTDIR="release"
DRY_RUN=0
SEQ=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)  OUTDIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"
exec 3>&1 4>&2
exec > >(tee "$LOG") 2>&1
trap 'exec 1>&3 2>&4' EXIT

run() {
  local script="$1"
  if [[ ! -f "$script" ]]; then
    echo "[ERROR] missing script: $script" >&2; exit 1
  fi
  SEQ=$((SEQ + 1))
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '   [%2d] %s\n' "$SEQ" "$script"; return 0
  fi
  echo ""
  echo "--------------------------------------------------------------------------------"
  echo ">>> [$SEQ] $script"
  echo "--------------------------------------------------------------------------------"
  local t0 t1
  t0=$(date +%s)
  "$PY" "$script"
  t1=$(date +%s)
  printf '    done in %dm %02ds\n' $(( (t1-t0)/60 )) $(( (t1-t0)%60 ))
}

echo "ClinicalTensorSepsis dataset build"
echo "  python  : $($PY --version 2>&1)"
echo "  started : $(date)"
echo "  outdir  : $OUTDIR"
if [[ $DRY_RUN -eq 1 ]]; then echo "  MODE    : dry run, nothing will execute"; fi

echo ""
echo "########## MIMIC-IV cohort and tensor ##########"
run src/01_mimic_datasets/01_mimic_base_cohort.py
run src/01_mimic_datasets/02_mimic_confirmed_infection.py
run src/01_mimic_datasets/03_mimic_phenotype_lock.py
run src/01_mimic_datasets/04a_mimic_temporal_slice.py
run src/01_mimic_datasets/04b_mimic_physiological_bounds.py
run src/01_mimic_datasets/05_mimic_sofa_calculator.py
run src/01_mimic_datasets/06_mimic_seymour_verification.py
run src/01_mimic_datasets/07a_mimic_tensor_builder.py
run src/01_mimic_datasets/07b_mimic_saits_imputation.py

echo ""
echo "########## eICU-CRD cohort and tensor ##########"
run src/03_eicu_datasets/01_eicu_cohort.py
run src/03_eicu_datasets/02_eicu_confirmed_infection.py
run src/03_eicu_datasets/03_eicu_phenotype_lock.py
run src/03_eicu_datasets/04a_eicu_temporal_slice.py
run src/03_eicu_datasets/04b_eicu_physiological_bounds.py
run src/03_eicu_datasets/04c_eicu_profile_infusiondrug.py
run src/03_eicu_datasets/04d_eicu_extract_pressors.py
run src/03_eicu_datasets/04e_eicu_standardize_units.py
run src/03_eicu_datasets/04f_eicu_compute_neq.py
run src/03_eicu_datasets/04g_eicu_extract_gcs.py
run src/03_eicu_datasets/04h_eicu_extract_fio2.py
run src/03_eicu_datasets/05_eicu_sofa_calculator.py
run src/03_eicu_datasets/06_eicu_seymour_verification.py
run src/03_eicu_datasets/07a_eicu_tensor_builder.py
run src/03_eicu_datasets/07b_eicu_saits_imputation.py
run src/03_eicu_datasets/07c_feature_parity_audit.py

echo ""
echo "########## Partition assignment ##########"
run src/02_models/01_ml_baselines.py

echo ""
echo "########## Cross-database alignment ##########"
run src/04_atlas_datasets/01a_harmonize_ot_tensor.py
run src/04_atlas_datasets/01b_verify_atlas_harmonization.py

echo ""
echo "########## Export ##########"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "   would run: pipeline/export_release.py --outdir $OUTDIR"
else
  "$PY" pipeline/export_release.py --outdir "$OUTDIR"
fi

echo ""
echo "########## BUILD COMPLETE - $(date) ##########"
cat <<NEXT

  Package written to $OUTDIR/
  Log at $LOG

  Before submitting:
    physionet validate $OUTDIR
    croissant-baker --input $OUTDIR

  Submission text: pipeline/metadata_draft.md
NEXT
