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
#   pipeline.log        every session, appended, never overwritten. Stopping at
#                       stage 2 and resuming at stage 3 next week leaves a single
#                       file that reads end to end, each invocation delimited by
#                       a SESSION banner. Only --fresh clears it, and that
#                       archives the old one first.
#   archive/            logs retired by --fresh, timestamped
#   manifest.tsv        script, stage, status, wall time, session; also appended
#   completed.txt       scripts that finished, used to resume mid-stage
#
# Flags, all optional:
#   --resume            skip scripts already in completed.txt
#   --from N --to N     stage range
#   --auto              never prompt
#   --dry-run           print the plan, execute nothing
#   --no-digest         skip the results digest at the end
#   --fresh             archive the current log, start a new one
#   --trust N|all       record a stage's output as matching the code, no rerun
#
set -Eeuo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
LOG_DIR="outputs/logs"
LOG="$LOG_DIR/pipeline.log"
ARCHIVE_DIR="$LOG_DIR/archive"
MANIFEST="$LOG_DIR/manifest.tsv"
DONE_FILE="$LOG_DIR/completed.txt"
HASHES="$LOG_DIR/stage_hashes.tsv"
TIMEOUT=15
RUN_ID="$(date '+%Y%m%d-%H%M%S')"
RUN_T0=$(date +%s)
PRIOR_LINES=0

START_STAGE=1; END_STAGE=6; AUTO=0; DRY_RUN=0; DIGEST=1; RESUME=0; CHOSE=0
SEQ=0; STAGE=0; STOP=0; STALE=0; FRESH=0; STAGE_SKIPPED=0; SKIPPED=0
TRUST=""
SKIP_THIS=0; STAGE_HAD_SKIP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)      START_STAGE="$2"; CHOSE=1; shift 2 ;;
    --to)        END_STAGE="$2"; CHOSE=1; shift 2 ;;
    --resume)    RESUME=1; CHOSE=1; shift ;;
    --auto)      AUTO=1; shift ;;
    --dry-run)   DRY_RUN=1; CHOSE=1; shift ;;
    --no-digest) DIGEST=0; shift ;;
    --fresh)     FRESH=1; shift ;;
    --trust)     TRUST="$2"; shift 2 ;;
    -h|--help)   awk 'NR>1 && /^#/; NR>1 && !/^#/{exit}' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

INTERACTIVE=0
if [[ -e /dev/tty ]] && (exec 3>/dev/tty) 2>/dev/null; then INTERACTIVE=1; fi
if [[ $AUTO -eq 1 ]]; then INTERACTIVE=0; fi

mkdir -p "$LOG_DIR"
if [[ $FRESH -eq 1 && -f "$LOG" ]]; then
  mkdir -p "$ARCHIVE_DIR"
  mv -f "$LOG" "$ARCHIVE_DIR/pipeline_$RUN_ID.log"
  echo "archived the previous log to $ARCHIVE_DIR/pipeline_$RUN_ID.log"
fi
if [[ -f "$LOG" ]]; then PRIOR_LINES=$(wc -l < "$LOG" | tr -d ' '); fi

# Append, never truncate. A resumed run has to read as a continuation of the
# session that stopped, not as a replacement for it.
exec 3>&1 4>&2
exec > >(tee -a "$LOG") 2>&1

finish() {
  local rc=$?
  local el=$(( $(date +%s) - RUN_T0 ))
  exec 1>&3 2>&4          # restore first, so the footer cannot race tee
  {
    printf '\n# SESSION %s ended %s after %dm %02ds, exit %d\n' \
      "$RUN_ID" "$(date '+%Y-%m-%d %H:%M:%S')" $((el / 60)) $((el % 60)) "$rc"
    printf '%s\n' "################################################################################"
  } >> "$LOG"
}
trap finish EXIT

# Prompts are framed and coloured so they cannot be mistaken for the output of
# whichever script just finished. Colour is skipped if NO_COLOR is set.
if [[ $INTERACTIVE -eq 1 && -z "${NO_COLOR:-}" ]]; then
  BAR=$'\033[1;33m'; INV=$'\033[7;33m'; OFF=$'\033[0m'
else
  BAR=""; INV=""; OFF=""
fi
RULE="=============================================================================="

say() { if [[ $INTERACTIVE -eq 1 ]]; then printf '%s\n' "$*" > /dev/tty 2>/dev/null || true; fi; }

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
  printf '%s' "${BAR}$1${OFF}" > /dev/tty 2>/dev/null || true
  if read -r -t "$3" reply < /dev/tty 2>/dev/null; then
    printf '%s' "${reply:-$2}"
  else
    printf '\n' > /dev/tty 2>/dev/null || true
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

# The scripts each stage runs. Data, not control flow: the hash that decides
# whether a stage is stale is taken over exactly these files.
STAGE_SCRIPTS=(
  # stage 1
  "src/01_mimic_datasets/01_mimic_base_cohort.py
src/01_mimic_datasets/02_mimic_confirmed_infection.py
src/01_mimic_datasets/03_mimic_phenotype_lock.py
src/01_mimic_datasets/04a_mimic_temporal_slice.py
src/01_mimic_datasets/04b_mimic_physiological_bounds.py
src/01_mimic_datasets/05_mimic_sofa_calculator.py
src/01_mimic_datasets/06_mimic_seymour_verification.py
src/01_mimic_datasets/07a_mimic_tensor_builder.py
src/01_mimic_datasets/07b_mimic_saits_imputation.py"
  # stage 2
  "src/03_eicu_datasets/01_eicu_cohort.py
src/03_eicu_datasets/02_eicu_confirmed_infection.py
src/03_eicu_datasets/03_eicu_phenotype_lock.py
src/03_eicu_datasets/04a_eicu_temporal_slice.py
src/03_eicu_datasets/04b_eicu_physiological_bounds.py
src/03_eicu_datasets/04c_eicu_profile_infusiondrug.py
src/03_eicu_datasets/04d_eicu_extract_pressors.py
src/03_eicu_datasets/04e_eicu_standardize_units.py
src/03_eicu_datasets/04f_eicu_compute_neq.py
src/03_eicu_datasets/04g_eicu_extract_gcs.py
src/03_eicu_datasets/04h_eicu_extract_fio2.py
src/03_eicu_datasets/05_eicu_sofa_calculator.py
src/03_eicu_datasets/06_eicu_seymour_verification.py
src/03_eicu_datasets/07a_eicu_tensor_builder.py
src/03_eicu_datasets/07b_eicu_saits_imputation.py
src/03_eicu_datasets/07c_feature_parity_audit.py"
  # stage 3
  "src/02_models/01_ml_baselines.py
src/02_models/02_clinical_scores.py
src/02_models/03a_temporal_bigru.py
src/02_models/03b_static_mlp.py
src/02_models/03c_multimodal_bigru.py
src/02_models/03d_attention_bigru.py
src/02_models/04a_champion_xgboost.py
src/02_models/04b_champion_lr.py
src/02_models/05_modality_ablation.py
src/02_models/06_probability_calibration.py
src/02_models/07_shap_interpretation.py
src/02_models/08_clinical_rfecv.py
src/02_models/09a_mimic_pruned_internal_test.py
src/02_models/09b_eicu_pruned_external_validation.py
src/02_models/10_temporal_early_warning.py"
  # stage 4
  "src/03_eicu_datasets/10_eicu_xgboost_external_validation.py
src/03_eicu_datasets/11a_eicu_recalibration.py
src/03_eicu_datasets/11b_eicu_pruned_recalibration.py
src/03_eicu_datasets/12_eicu_temporal_early_warning.py
src/03_eicu_datasets/13_eicu_decision_curve_analysis.py"
  # stage 5
  "src/04_atlas_datasets/01a_harmonize_ot_tensor.py
src/04_atlas_datasets/01b_verify_atlas_harmonization.py
src/04_atlas_datasets/02_compute_joint_manifold.py
src/04_atlas_datasets/03_plot_prognostic_landscape.py
src/04_atlas_datasets/04_evaluate_domain_adapted_eicu.py
src/04_atlas_datasets/05_evaluate_pruned_domain_adapted_eicu.py
src/04_atlas_datasets/06_ot_hyperparameter_sweep.py
src/04_atlas_datasets/07_evaluate_feature_gated_ot.py"
  # stage 6
  "src/05_statistical_analysis/01_delong_statistical_significance.py
src/05_statistical_analysis/02_feature_importance_correlation.py
src/05_statistical_analysis/03_algorithmic_fairness_audit.py
src/05_statistical_analysis/04_wong_clinical_utility_audits.py
src/05_statistical_analysis/05_multicenter_variance_test.py"
)

show_status() {
  local first=0 i
  say ""
  say "  SEPSIS-3 PIPELINE"
  say "  ---------------------------------------------------------------------"
  for i in 0 1 2 3 4 5; do
    if is_stale "$i"; then
      say "   STALE     stage $((i+1))  ${STAGE_NAME[$i]}   (scripts changed since this ran)"
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

# Whether a stage's output was produced by the code now on disk.
#
# This used to compare mtimes, which is the wrong signal: copying src/ between
# machines rewrites every mtime without changing a line of code, so a fresh
# checkout marked every artifact stale. A content hash only moves when the code
# moves. With no recorded hash the stage is treated as current, so existing
# output is never condemned on a guess; 'trust' in the advanced menu records
# one deliberately.
HASHER="sha256sum"
if ! command -v sha256sum >/dev/null 2>&1; then
  if command -v shasum >/dev/null 2>&1; then HASHER="shasum -a 256"; else HASHER="cksum"; fi
fi

stage_hash() {
  local i=$1 s
  for s in ${STAGE_SCRIPTS[$i]}; do
    if [[ -f "$s" ]]; then cat "$s"; fi
  done | $HASHER | cut -d' ' -f1
}

stored_stage_hash() {
  [[ -f "$HASHES" ]] || return 1
  awk -F'\t' -v s="$(($1 + 1))" '$1==s {print $2; f=1} END {exit !f}' "$HASHES"
}

record_stage_hash() {
  local i=$1 h; h=$(stage_hash "$i")
  mkdir -p "$LOG_DIR"
  if [[ -f "$HASHES" ]]; then
    grep -v "^$((i + 1))	" "$HASHES" > "$HASHES.tmp" 2>/dev/null || : > "$HASHES.tmp"
    mv -f "$HASHES.tmp" "$HASHES"
  fi
  printf '%d\t%s\n' "$((i + 1))" "$h" >> "$HASHES"
}

# Only a stage that ran end to end under the current code earns a hash. If a
# script was skipped, the output on disk did not all come from this code, and
# recording a hash would make the staleness check repeat that claim later.
stage_done() {
  if [[ $DRY_RUN -eq 1 || $STOP -eq 1 ]]; then return 0; fi
  if [[ $STAGE_HAD_SKIP -eq 1 ]]; then
    echo "    stage $(($1 + 1)) had skipped scripts; leaving it unverified"
    STAGE_HAD_SKIP=0
    return 0
  fi
  record_stage_hash "$1"
}

is_stale() {
  local i=$1 stored
  [[ -f "${STAGE_SENTINEL[$i]}" ]] || return 1
  stored=$(stored_stage_hash "$i") || return 1
  [[ "$stored" != "$(stage_hash "$i")" ]]
}

n_completed() { [[ -f "$DONE_FILE" ]] && wc -l < "$DONE_FILE" | tr -d ' ' || echo 0; }

advanced_settings() {
  local c n p m
  while true; do
    frame_open "ADVANCED" "returns in ${TIMEOUT}s"
    say "  Reruns and existing output"
    say ""
    say "    t) trust a stage's existing output   record its hash, no rerun"
    say "    b) back"
    say ""
    say "  To skip a single script, answer [k] at the prompt before it runs."
    frame_close
    c=$(askt "  your choice [b] > " "b" "$TIMEOUT")
    say ""
    case "${c,,}" in
      t) n=$(askt "  stage to trust [1-6, or 'all'] > " "" "$TIMEOUT")
         if [[ "$n" == "all" ]]; then
           for i in 0 1 2 3 4 5; do record_stage_hash "$i"; done
           STALE=0; say "  all six stages recorded as matching the code on disk."
         elif [[ "$n" =~ ^[1-6]$ ]]; then
           record_stage_hash "$((n - 1))"
           STALE=0; say "  stage $n recorded as matching the code on disk."
         else
           say "  not a stage number; nothing changed."
         fi ;;
      *) return 0 ;;
    esac
    say ""
  done
}

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
  say "    a) advanced: trust a stage's existing output"
  say "    c) start clean: delete data/processed/ and outputs/, keep data/raw/"
  say ""
  local def="1"
  if [[ $ndone -gt 0 ]]; then def="r"; fi
  if [[ $STALE -eq 1 ]]; then
    def="c"
    say "  Some output was produced by scripts that have changed since. Starting"
    say "  clean is the safe choice; if you know the change does not affect that"
    say "  stage, 'a' then 't' records it as current without rerunning."
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
    a) advanced_settings; choose_what_to_run; return 0 ;;
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
  say "    [k]      skip it, keep the output already on disk"
  say "    [a]      run everything remaining without asking again"
  say "    [s]      stop here, resume later with --resume"
  say "    [l]      show the last 30 lines of output"
  frame_close
  local r
  r=$(askt "  your choice > " "" "$TIMEOUT")
  case "${r,,}" in
    k) SKIP_THIS=1 ;;
    a) AUTO=1; say "   continuing without further prompts" ;;
    s) STOP=1 ;;
    l) tail -n 30 "$LOG" > /dev/tty 2>/dev/null || true; between_scripts "$next" ;;
    *) : ;;
  esac
}

run() {
  local script="$1"
  if [[ $STOP -eq 1 ]]; then return 0; fi
  [[ -f "$script" ]] || { echo "[ERROR] missing script: $script" >&2; exit 1; }
  SEQ=$((SEQ + 1))

  if [[ $RESUME -eq 1 ]] && grep -qxF "$script" "$DONE_FILE" 2>/dev/null; then
    STAGE_SKIPPED=$((STAGE_SKIPPED + 1)); return 0
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '   [%2d] %s\n' "$SEQ" "$script"; return 0
  fi

  SKIP_THIS=0
  between_scripts "$script"
  if [[ $STOP -eq 1 ]]; then return 0; fi
  if [[ $SKIP_THIS -eq 1 ]]; then
    printf '%d\t%d\t%s\tskipped\t0\t%s\n' "$SEQ" "$STAGE" "$script" "$RUN_ID" >> "$MANIFEST"
    printf '   [%2d] skipped at your request, keeping existing output   %s\n' "$SEQ" "$script"
    STAGE_HAD_SKIP=1
    return 0
  fi

  echo ""
  echo "================================================================================"
  echo ">>> [$SEQ] $script"
  echo "    stage $STAGE, started $(date '+%H:%M:%S')"
  echo "================================================================================"

  local t0 t1 dt rc
  t0=$(date +%s); set +e; "$PY" "$script"; rc=$?; set -e
  t1=$(date +%s); dt=$((t1 - t0))

  if [[ $rc -ne 0 ]]; then
    printf '%d\t%d\t%s\tFAILED\t%d\t%s\n' "$SEQ" "$STAGE" "$script" "$dt" "$RUN_ID" >> "$MANIFEST"
    echo ""
    echo "[FAILED] $script  (exit $rc after ${dt}s)" >&2
    echo "         Nothing downstream would have valid inputs. Fix, then:" >&2
    echo "           bash pipeline/run_pipeline.sh --resume" >&2
    exit "$rc"
  fi

  printf '%d\t%d\t%s\tok\t%d\t%s\n' "$SEQ" "$STAGE" "$script" "$dt" "$RUN_ID" >> "$MANIFEST"
  echo "$script" >> "$DONE_FILE"
  printf '    done in %dm %02ds\n' $((dt / 60)) $((dt % 60))
}

# One line per stage rather than one per script: whatever was skipped already
# has its real output in this file, written by the session that ran it.
flush_skips() {
  if [[ $STAGE_SKIPPED -gt 0 ]]; then
    echo "    ($STAGE_SKIPPED script(s) ran in an earlier session; their output is above)"
    SKIPPED=$((SKIPPED + STAGE_SKIPPED)); STAGE_SKIPPED=0
  fi
}

stage() {
  flush_skips
  STAGE=$1
  if [[ $STOP -eq 1 ]]; then return 1; fi
  if [[ $START_STAGE -gt $STAGE || $END_STAGE -lt $STAGE ]]; then return 1; fi
  echo ""
  echo "########## STAGE $STAGE - ${STAGE_NAME[$((STAGE-1))]} ##########"
  return 0
}

apply_flags() {
  local one
  if [[ "$TRUST" == "all" ]]; then
    for one in 0 1 2 3 4 5; do record_stage_hash "$one"; done
  elif [[ "$TRUST" =~ ^[1-6]$ ]]; then
    record_stage_hash "$((TRUST - 1))"
  fi
}

main() {
  apply_flags
  choose_what_to_run

  if [[ $RESUME -eq 0 && $DRY_RUN -eq 0 && $START_STAGE -eq 1 && $END_STAGE -eq 6 ]]; then
    : > "$DONE_FILE"
  fi
  # The manifest accumulates too; the session column separates the invocations.
  [[ -f "$MANIFEST" ]] || \
    printf 'seq\tstage\tscript\tstatus\tseconds\tsession\n' > "$MANIFEST"

  echo ""
  echo "################################################################################"
  echo "# SESSION $RUN_ID  started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "################################################################################"
  echo "Sepsis-3 portability pipeline"
  echo "  python  : $($PY --version 2>&1)"
  echo "  stages  : $START_STAGE to $END_STAGE"
  if [[ $PRIOR_LINES -gt 0 ]]; then
    echo "  log     : $LOG (appending below $PRIOR_LINES lines from earlier sessions)"
  else
    echo "  log     : $LOG (new)"
  fi
  if [[ $RESUME  -eq 1 ]]; then echo "  mode    : resume, skipping $(n_completed) completed scripts"; fi
  if [[ $DRY_RUN -eq 1 ]]; then echo "  mode    : dry run, nothing will execute"; fi

  if stage 1; then
    for s in ${STAGE_SCRIPTS[0]}; do run "$s"; done
    stage_done 0
  fi

  if stage 2; then
    for s in ${STAGE_SCRIPTS[1]}; do run "$s"; done
    stage_done 1
  fi

  if stage 3; then
    for s in ${STAGE_SCRIPTS[2]}; do run "$s"; done
    stage_done 2
  fi

  if stage 4; then
    for s in ${STAGE_SCRIPTS[3]}; do run "$s"; done
    stage_done 3
  fi

  if stage 5; then
    for s in ${STAGE_SCRIPTS[4]}; do run "$s"; done
    stage_done 4
  fi

  if stage 6; then
    for s in ${STAGE_SCRIPTS[5]}; do run "$s"; done
    stage_done 5
  fi

  echo ""
  flush_skips
  if [[ $STOP -eq 1 ]]; then
    echo "########## STOPPED AT YOUR REQUEST - $(date) ##########"
    echo "Resume with:  bash pipeline/run_pipeline.sh --resume"
    return 0
  fi
  echo "########## PIPELINE COMPLETE - $(date) ##########"
  if [[ $SKIPPED -gt 0 ]]; then
    echo "  $SKIPPED of this session's scripts had already run; the rest is above."
  fi

  if [[ $DRY_RUN -eq 0 ]]; then
    echo ""
    echo "Slowest scripts:"
    awk -F'\t' -v s="$RUN_ID" 'NR>1 && $4=="ok" && $6==s {printf "  %5ds  %s\n", $5, $3}' \
      "$MANIFEST" | sort -rn | head -8
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
  outputs/logs/pipeline.log    every session, end to end, appended
  outputs/logs/manifest.tsv    script, status, wall time, session

  Not run here:
    bash pipeline/run_dtw_phate_atlas.sh    the 8 slow DTW scripts
    src/05_statistical_analysis/06_demographics...ipynb   run by hand
--------------------------------------------------------------------------------
NEXT
}

main
