#!/usr/bin/env bash
#
# run_pipeline.sh - Sepsis-3 portability pipeline.
#
# Interactive by default: it shows what has already completed, asks what to run,
# and pauses between scripts. Every prompt auto-answers after 15 seconds, so an
# unattended run still finishes on its own.
#
# Order follows the artifact dependencies, not filename order. Two places
# interleave the cohorts and cannot be reordered: eICU 04f and 07b read MIMIC
# artifacts, so stage 1 runs first; 02_models/09b reads the eICU imputed tensor
# while eICU 10-13 read the champion model, so eICU data prep sits before the
# models and eICU evaluation after them.
#
# Logs, all under outputs/logs/:
#   pipeline.log        this run, everything, in order
#   pipeline.prev.log   the run before it
#   manifest.tsv        script, stage, status, wall time
#   completed.txt       scripts that finished, used to resume mid-stage
#
# Flags, all optional:
#   --resume            skip scripts already in completed.txt
#   --from N --to N     stage range
#   --auto              never prompt
#   --dry-run           print the plan, execute nothing
#   --no-digest         skip the results digest at the end
#
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
LOG_DIR="outputs/logs"
LOG="$LOG_DIR/pipeline.log"
PREV="$LOG_DIR/pipeline.prev.log"
MANIFEST="$LOG_DIR/manifest.tsv"
DONE_FILE="$LOG_DIR/completed.txt"
TIMEOUT=15

START_STAGE=1; END_STAGE=6; AUTO=0; DRY_RUN=0; DIGEST=1; RESUME=0; CHOSE=0
SEQ=0; STAGE=0; STOP=0; STALE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)      START_STAGE="$2"; CHOSE=1; shift 2 ;;
    --to)        END_STAGE="$2"; CHOSE=1; shift 2 ;;
    --resume)    RESUME=1; CHOSE=1; shift ;;
    --auto)      AUTO=1; shift ;;
    --dry-run)   DRY_RUN=1; CHOSE=1; shift ;;
    --no-digest) DIGEST=0; shift ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

INTERACTIVE=0
if [[ -e /dev/tty && -r /dev/tty ]]; then INTERACTIVE=1; fi
if [[ $AUTO -eq 1 ]]; then INTERACTIVE=0; fi

mkdir -p "$LOG_DIR"
if [[ -f "$LOG" ]]; then mv -f "$LOG" "$PREV"; fi
exec 3>&1 4>&2
exec > >(tee "$LOG") 2>&1
trap 'exec 1>&3 2>&4' EXIT

# Prompts are framed and coloured so they cannot be mistaken for the output of
# whichever script just finished. Colour is skipped if NO_COLOR is set.
if [[ $INTERACTIVE -eq 1 && -z "${NO_COLOR:-}" ]]; then
  BAR=$'\033[1;33m'; INV=$'\033[7;33m'; OFF=$'\033[0m'
else
  BAR=""; INV=""; OFF=""
fi
RULE="=============================================================================="

say() { if [[ $INTERACTIVE -eq 1 ]]; then printf '%s\n' "$*" > /dev/tty; fi; }

frame_open() {  # $1 = title, $2 = right-hand note
  say ""
  say ""
  say "${BAR}${RULE}${OFF}"
  say "${INV} $1 ${OFF}${BAR}  $2${OFF}"
  say "${BAR}${RULE}${OFF}"
}
frame_close() { say "${BAR}${RULE}${OFF}"; say ""; }
askt() { # prompt, default, timeout -> reply on stdout
  local reply
  if [[ $INTERACTIVE -eq 0 ]]; then printf '%s' "$2"; return; fi
  printf '%s' "${BAR}$1${OFF}" > /dev/tty
  if read -r -t "$3" reply < /dev/tty; then
    printf '%s' "${reply:-$2}"
  else
    printf '\n' > /dev/tty
    printf '%s' "$2"
  fi
}

# --- Stage Table ----------------------------------------------------------
# name | sentinel artifact written by the stage's last script | script count
STAGE_NAME=(
  "MIMIC-IV cohort, phenotype and tensor"
  "eICU-CRD cohort, phenotype and tensor"
  "Model development on MIMIC-IV"
  "External validation on eICU-CRD"
  "Optimal transport and cross-database atlas"
  "Statistical analysis")
STAGE_SENTINEL=(
  "data/processed/mimiciv/mimic_sepsis_imputed_tensor.npy"
  "outputs/metrics/feature_parity_density.csv"
  "outputs/metrics/mimic_temporal_early_warning_metrics.json"
  "outputs/metrics/eicu_dca_summary.csv"
  "outputs/metrics/atlas_ot_constrained_variants.json"
  "outputs/analysis/eicu_multicenter_variance_report.csv")
STAGE_COUNT=(9 16 15 5 8 5)

show_status() {
  local first=0 i
  say ""
  say "  SEPSIS-3 PIPELINE"
  say "  ---------------------------------------------------------------------"
  for i in 0 1 2 3 4 5; do
    if is_stale "${STAGE_SENTINEL[$i]}"; then
      say "   STALE     stage $((i+1))  ${STAGE_NAME[$i]}   (output predates the code)"
      STALE=1
      if [[ $first -eq 0 ]]; then first=$((i+1)); fi
    elif [[ -f "${STAGE_SENTINEL[$i]}" ]]; then
      say "   done      stage $((i+1))  ${STAGE_NAME[$i]}"
    else
      say "   pending   stage $((i+1))  ${STAGE_NAME[$i]}   (${STAGE_COUNT[$i]} scripts)"
      if [[ $first -eq 0 ]]; then first=$((i+1)); fi
    fi
  done
  local ndone=0
  for f in outputs/features/mimic_phate_shape_coordinates.parquet \
           outputs/features/mimic_phate_severity_coordinates.parquet \
           outputs/features/eicu_phate_shape_coordinates.parquet \
           outputs/features/eicu_phate_severity_coordinates.parquet; do
    if [[ -f "$f" ]]; then ndone=$((ndone+1)); fi
  done
  say "   ${ndone}/4       DTW trajectory atlas (separate, slow)"
  say ""
  NEXT_STAGE=${first:-1}
  if [[ $first -eq 0 ]]; then NEXT_STAGE=1; fi
  return 0
}

# An artifact older than any script in src/ was produced by earlier code.
is_stale() {
  [[ -f "$1" ]] || return 1
  [[ -n "$(find src -name '*.py' -newer "$1" -print -quit 2>/dev/null)" ]]
}

n_completed() { [[ -f "$DONE_FILE" ]] && wc -l < "$DONE_FILE" | tr -d ' ' || echo 0; }

choose_what_to_run() {
  if [[ $INTERACTIVE -eq 0 || $CHOSE -eq 1 ]]; then return 0; fi
  show_status
  local ndone; ndone=$(n_completed)
  frame_open "INPUT NEEDED" "auto-selects in ${TIMEOUT}s"
  say "  What would you like to run?"
  say ""
  if [[ $ndone -gt 0 ]]; then
    say "    r) resume where the last run stopped   ($ndone scripts already done)"
  fi
  say "    1) the full pipeline, stages 1-6"
  say "    2) one stage only"
  say "    3) the DTW / PHATE trajectory atlas   (slow, hours)"
  say "    4) rebuild the results digest only"
  say "    5) dry run, print the plan and execute nothing"
  say ""
  say "    c) start clean: delete data/processed/ and outputs/, keep data/raw/"
  say ""
  local def="1"
  if [[ $ndone -gt 0 ]]; then def="r"; fi
  if [[ $STALE -eq 1 ]]; then
    def="c"
    say "  Some output is older than the code that produces it. Those artifacts came"
    say "  from an earlier version of the scripts and cannot be mixed with new ones."
    say "  Starting clean is the safe choice."
    say ""
  fi
  frame_close
  local c; c=$(askt "  your choice [$def] > " "$def" "$TIMEOUT")
  say ""
  case "${c,,}" in
    r) RESUME=1 ;;
    2) local s; s=$(askt "  which stage [1-6, default $NEXT_STAGE]: " "$NEXT_STAGE" "$TIMEOUT")
       START_STAGE="$s"; END_STAGE="$s"; say "" ;;
    3) say "  handing off to the trajectory atlas..."
       exec bash pipeline/run_dtw_phate_atlas.sh ;;
    4) "$PY" pipeline/collect_results.py; exit $? ;;
    5) DRY_RUN=1 ;;
    c) say "  This will delete data/processed/ and outputs/ and keep data/raw/."
       local yn; yn=$(askt "  type 'delete' to confirm: " "" 30)
       if [[ "$yn" == "delete" ]]; then
         rm -rf data/processed outputs
         mkdir -p "$LOG_DIR"
         : > "$DONE_FILE"
         printf 'seq	stage	script	status	seconds
' > "$MANIFEST"
         say "  cleared. running the full pipeline from stage 1."
       else
         say "  not confirmed, nothing deleted; exiting so you can decide."
         exit 0
       fi ;;
    *) : ;;
  esac
}

between_scripts() {
  local next="$1"
  if [[ $AUTO -eq 1 || $INTERACTIVE -eq 0 || $DRY_RUN -eq 1 ]]; then return 0; fi
  frame_open "INPUT NEEDED" "runs automatically in ${TIMEOUT}s"
  say "  next script:  $next"
  say ""
  say "    [Enter]  run it"
  say "    [a]      run everything remaining without asking again"
  say "    [s]      stop here, resume later with --resume"
  say "    [l]      show the last 30 lines of output"
  frame_close
  local r
  r=$(askt "  your choice > " "" "$TIMEOUT")
  case "${r,,}" in
    a) AUTO=1; say "   continuing without further prompts" ;;
    s) STOP=1 ;;
    l) tail -n 30 "$LOG" > /dev/tty; between_scripts "$next" ;;
    *) : ;;
  esac
}

run() {
  local script="$1"
  if [[ $STOP -eq 1 ]]; then return 0; fi
  [[ -f "$script" ]] || { echo "[ERROR] missing script: $script" >&2; exit 1; }
  SEQ=$((SEQ + 1))

  if [[ $RESUME -eq 1 ]] && grep -qxF "$script" "$DONE_FILE" 2>/dev/null; then
    printf '   [%2d] already done, skipping   %s\n' "$SEQ" "$script"; return 0
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '   [%2d] %s\n' "$SEQ" "$script"; return 0
  fi

  between_scripts "$script"
  if [[ $STOP -eq 1 ]]; then return 0; fi

  echo ""
  echo "================================================================================"
  echo ">>> [$SEQ] $script"
  echo "    stage $STAGE, started $(date '+%H:%M:%S')"
  echo "================================================================================"

  local t0 t1 dt rc
  t0=$(date +%s); set +e; "$PY" "$script"; rc=$?; set -e
  t1=$(date +%s); dt=$((t1 - t0))

  if [[ $rc -ne 0 ]]; then
    printf '%d\t%d\t%s\tFAILED\t%d\n' "$SEQ" "$STAGE" "$script" "$dt" >> "$MANIFEST"
    echo ""
    echo "[FAILED] $script  (exit $rc after ${dt}s)" >&2
    echo "         Nothing downstream would have valid inputs. Fix, then:" >&2
    echo "           bash pipeline/run_pipeline.sh --resume" >&2
    exit "$rc"
  fi

  printf '%d\t%d\t%s\tok\t%d\n' "$SEQ" "$STAGE" "$script" "$dt" >> "$MANIFEST"
  echo "$script" >> "$DONE_FILE"
  printf '    done in %dm %02ds\n' $((dt / 60)) $((dt % 60))
}

stage() {
  STAGE=$1
  if [[ $STOP -eq 1 ]]; then return 1; fi
  if [[ $START_STAGE -gt $STAGE || $END_STAGE -lt $STAGE ]]; then return 1; fi
  echo ""
  echo "########## STAGE $STAGE - ${STAGE_NAME[$((STAGE-1))]} ##########"
  return 0
}

main() {
  choose_what_to_run

  if [[ $RESUME -eq 0 && $DRY_RUN -eq 0 && $START_STAGE -eq 1 && $END_STAGE -eq 6 ]]; then
    : > "$DONE_FILE"
    printf 'seq	stage	script	status	seconds
' > "$MANIFEST"
  fi
  [[ -f "$MANIFEST" ]] || printf 'seq\tstage\tscript\tstatus\tseconds\n' > "$MANIFEST"

  echo "Sepsis-3 portability pipeline"
  echo "  python  : $($PY --version 2>&1)"
  echo "  started : $(date)"
  echo "  stages  : $START_STAGE to $END_STAGE"
  echo "  log     : $LOG"
  if [[ $RESUME  -eq 1 ]]; then echo "  mode    : resume, skipping $(n_completed) completed scripts"; fi
  if [[ $DRY_RUN -eq 1 ]]; then echo "  mode    : dry run, nothing will execute"; fi

  if stage 1; then
    run src/01_mimic_datasets/01_mimic_base_cohort.py
    run src/01_mimic_datasets/02_mimic_confirmed_infection.py
    run src/01_mimic_datasets/03_mimic_phenotype_lock.py
    run src/01_mimic_datasets/04a_mimic_temporal_slice.py
    run src/01_mimic_datasets/04b_mimic_physiological_bounds.py
    run src/01_mimic_datasets/05_mimic_sofa_calculator.py
    run src/01_mimic_datasets/06_mimic_seymour_verification.py
    run src/01_mimic_datasets/07a_mimic_tensor_builder.py
    run src/01_mimic_datasets/07b_mimic_saits_imputation.py
  fi

  if stage 2; then
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
  fi

  if stage 3; then
    run src/02_models/01_ml_baselines.py
    run src/02_models/02_clinical_scores.py
    run src/02_models/03a_temporal_bigru.py
    run src/02_models/03b_static_mlp.py
    run src/02_models/03c_multimodal_bigru.py
    run src/02_models/03d_attention_bigru.py
    run src/02_models/04a_champion_xgboost.py
    run src/02_models/04b_champion_lr.py
    run src/02_models/05_modality_ablation.py
    run src/02_models/06_probability_calibration.py
    run src/02_models/07_shap_interpretation.py
    run src/02_models/08_clinical_rfecv.py
    run src/02_models/09a_mimic_pruned_internal_test.py
    run src/02_models/09b_eicu_pruned_external_validation.py
    run src/02_models/10_temporal_early_warning.py
  fi

  if stage 4; then
    run src/03_eicu_datasets/10_eicu_xgboost_external_validation.py
    run src/03_eicu_datasets/11a_eicu_recalibration.py
    run src/03_eicu_datasets/11b_eicu_pruned_recalibration.py
    run src/03_eicu_datasets/12_eicu_temporal_early_warning.py
    run src/03_eicu_datasets/13_eicu_decision_curve_analysis.py
  fi

  if stage 5; then
    run src/04_atlas_datasets/01a_harmonize_ot_tensor.py
    run src/04_atlas_datasets/01b_verify_atlas_harmonization.py
    run src/04_atlas_datasets/02_compute_joint_manifold.py
    run src/04_atlas_datasets/03_plot_prognostic_landscape.py
    run src/04_atlas_datasets/04_evaluate_domain_adapted_eicu.py
    run src/04_atlas_datasets/05_evaluate_pruned_domain_adapted_eicu.py
    run src/04_atlas_datasets/06_ot_hyperparameter_sweep.py
    run src/04_atlas_datasets/07_evaluate_feature_gated_ot.py
  fi

  if stage 6; then
    run src/05_statistical_analysis/01_delong_statistical_significance.py
    run src/05_statistical_analysis/02_feature_importance_correlation.py
    run src/05_statistical_analysis/03_algorithmic_fairness_audit.py
    run src/05_statistical_analysis/04_wong_clinical_utility_audits.py
    run src/05_statistical_analysis/05_multicenter_variance_test.py
  fi

  echo ""
  if [[ $STOP -eq 1 ]]; then
    echo "########## STOPPED AT YOUR REQUEST - $(date) ##########"
    echo "Resume with:  bash pipeline/run_pipeline.sh --resume"
    return 0
  fi
  echo "########## PIPELINE COMPLETE - $(date) ##########"

  if [[ $DRY_RUN -eq 0 ]]; then
    echo ""
    echo "Slowest scripts:"
    awk -F'\t' 'NR>1 && $4=="ok" {printf "  %5ds  %s\n", $5, $3}' "$MANIFEST" | sort -rn | head -8
  fi

  if [[ $DIGEST -eq 1 && $DRY_RUN -eq 0 ]]; then
    echo ""
    echo "########## RESULTS DIGEST ##########"
    "$PY" pipeline/collect_results.py \
      || echo "[WARN] digest failed; run 'python pipeline/collect_results.py' by hand"
  fi

  cat <<'NEXT'

--------------------------------------------------------------------------------
  outputs/RESULTS_DIGEST.md    every headline number, ordered to match the
                               Results subsections of main.tex
  outputs/logs/pipeline.log    this run
  outputs/logs/manifest.tsv    script, status, wall time

  Not run here:
    bash pipeline/run_dtw_phate_atlas.sh    the 8 slow DTW scripts
    src/05_statistical_analysis/06_demographics...ipynb   run by hand
--------------------------------------------------------------------------------
NEXT
}

main
