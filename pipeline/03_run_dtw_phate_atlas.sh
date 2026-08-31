#!/usr/bin/env bash
#
# run_dtw_phate_atlas.sh - within-cohort trajectory manifolds (DTW + PHATE).
#
# Split out of run_pipeline.sh because these eight scripts are the slowest part
# of the project by a wide margin. Each *_DTW_atlas_* script computes an all-pairs
# dynamic time warping distance matrix over the 24 x 30 imputed tensors, which is
# O(N^2) in patients: roughly 85 million DTW alignments for a 13k MIMIC cohort and
# 29 million for a 7.6k eICU cohort. Expect hours, and a distance matrix of
# N^2 x 4 bytes held in memory (about 680 MB at N = 13,000).
#
# Nothing in run_pipeline.sh reads any output of this script, so the two are
# independent. This one needs only the imputed tensors, i.e. stage 2 of
# run_pipeline.sh must have completed.
#
# Two manifolds are built per cohort:
#   shape     - each trajectory is mean-variance normalised along time, so the
#               distance reflects the morphology of physiological change
#   severity  - one global scaler across all patient-hours, so absolute
#               magnitude is preserved
#
# Usage:
#   bash pipeline/run_dtw_phate_atlas.sh                 # both cohorts, both manifolds
#   bash pipeline/run_dtw_phate_atlas.sh --cohort mimic  # MIMIC only
#   bash pipeline/run_dtw_phate_atlas.sh --cohort eicu   # eICU only
#   bash pipeline/run_dtw_phate_atlas.sh --dry-run
#
# Tip: this is a good candidate for nohup / tmux.
#   nohup bash pipeline/run_dtw_phate_atlas.sh > dtw.out 2>&1 &
#
set -Eeuo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
LOG_DIR="outputs/logs"
RUN_LOG="${LOG_DIR}/dtw_phate_atlas.log"
RUN_ID="$(date '+%Y%m%d-%H%M%S')"
RUN_T0=$(date +%s)
COHORT="both"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohort)  COHORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$COHORT" in
  both|mimic|eicu) ;;
  *) echo "--cohort must be one of: both, mimic, eicu" >&2; exit 2 ;;
esac

mkdir -p "$LOG_DIR"

require() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] required input missing: $1" >&2
    echo "        Run stage 2 of run_pipeline.sh first." >&2
    exit 1
  fi
}

run() {
  local script="$1"
  if [[ ! -f "$script" ]]; then
    echo "[ERROR] missing script: $script" >&2
    exit 1
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "   would run: $script"
    return 0
  fi
  echo ""
  echo "--------------------------------------------------------------------------------"
  echo ">>> $script"
  echo "    started $(date '+%H:%M:%S')"
  echo "--------------------------------------------------------------------------------"
  local t0 t1
  t0=$(date +%s)
  if ! "$PY" "$script"; then
    echo "" >&2
    echo "[FAILED] $script" >&2
    exit 1
  fi
  t1=$(date +%s)
  printf "    done in %dh %02dm %02ds\n" \
    $(( (t1-t0)/3600 )) $(( ((t1-t0)%3600)/60 )) $(( (t1-t0)%60 ))
}

banner() {
  echo ""
  echo "================================================================================"
  echo " $1"
  echo "================================================================================"
}

main() {
  echo "DTW + PHATE trajectory atlas"
  echo "  python  : $($PY --version 2>&1)"
  echo "  started : $(date)"
  echo "  cohort  : $COHORT"
  echo "  log     : $RUN_LOG"
  if [[ $DRY_RUN -eq 1 ]]; then echo "  MODE    : dry run, nothing will execute"; fi
  echo ""
  echo "  This is the long one. All-pairs DTW is O(N^2); budget hours, not minutes."

  if [[ "$COHORT" == "both" || "$COHORT" == "mimic" ]]; then
    if [[ $DRY_RUN -eq 0 ]]; then
      require data/processed/mimiciv/mimic_sepsis_imputed_tensor.npy
      require data/processed/mimiciv/mimic_sepsis_tensor_stay_ids.npy
    fi

    banner "MIMIC-IV - shape manifold (morphology of change)"
    run src/01_mimic_datasets/08a_mimic_DTW_atlas_shape.py
    run src/01_mimic_datasets/08b_mimic_atlas_shape_characterization.py

    banner "MIMIC-IV - severity manifold (absolute magnitude preserved)"
    run src/01_mimic_datasets/09a_mimic_DTW_atlas_severity.py
    run src/01_mimic_datasets/09b_mimic_atlas_severity_characterization.py
  fi

  if [[ "$COHORT" == "both" || "$COHORT" == "eicu" ]]; then
    if [[ $DRY_RUN -eq 0 ]]; then
      require data/processed/eicu/eicu_sepsis_imputed_tensor.npy
      require data/processed/eicu/eicu_sepsis_tensor_stay_ids.npy
    fi

    banner "eICU-CRD - shape manifold (morphology of change)"
    run src/03_eicu_datasets/08a_eicu_DTW_atlas_shape.py
    run src/03_eicu_datasets/08b_eicu_atlas_shape_characterization.py

    banner "eICU-CRD - severity manifold (absolute magnitude preserved)"
    run src/03_eicu_datasets/09a_eicu_DTW_atlas_severity.py
    run src/03_eicu_datasets/09b_eicu_atlas_severity_characterization.py
  fi

  echo ""
  echo "================================================================================"
  echo " TRAJECTORY ATLAS COMPLETE - $(date)"
  echo "================================================================================"
  cat <<'NEXT'

Written to outputs/features/:
  {mimic,eicu}_dtw_{shape,severity}_pairwise_distance_matrix.npy
  {mimic,eicu}_phate_{shape,severity}_coordinates.parquet

Written to outputs/figures/:
  {mimic,eicu}_{Shape,Severity}_Trajectory_Manifold.png   8-panel characterisation

Note: the joint cross-database PHATE manifold is a different analysis and is
already covered by run_pipeline.sh stage 5
(src/04_atlas_datasets/02_compute_joint_manifold.py). That one runs on the 122-D
OT-harmonised representation with pre-diffusion PCA, so it is fast. Only the
within-cohort DTW manifolds live here.
NEXT
}

# Append, never truncate. Each cohort takes hours and the two are often run
# days apart; a second invocation must extend the record, not replace it.
{
  echo ""
  echo "################################################################################"
  echo "# SESSION $RUN_ID  started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "################################################################################"
} >> "$RUN_LOG"

rc=0
main 2>&1 | tee -a "$RUN_LOG" || rc=$?

{
  el=$(( $(date +%s) - RUN_T0 ))
  printf '\n# SESSION %s ended %s after %dh %02dm, exit %d\n' \
    "$RUN_ID" "$(date '+%Y-%m-%d %H:%M:%S')" $((el / 3600)) $(((el % 3600) / 60)) "$rc"
  echo "################################################################################"
} >> "$RUN_LOG"
exit $rc
exit "${PIPESTATUS[0]}"
