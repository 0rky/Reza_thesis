#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/users/kashkoul/model-compression-experiments"
LOG_DIR="$PROJECT_ROOT/logs/final_10epoch"
mkdir -p "$LOG_DIR"

EPOCHS_MAIN=10
FINE_TUNE_EPOCHS_STRUCTURED=2

run_case () {
  local case_name="$1"
  local code_dir="$2"
  local log_file="$3"

  echo "================================================================================"
  echo "RUNNING CASE: $case_name"
  echo "Start time: $(date)"
  echo "Code dir   : $code_dir"
  echo "Log file   : $log_file"
  echo "================================================================================"

  cd "$code_dir"
  shift 3
  env "$@" python3 train.py 2>&1 | tee "$log_file"

  echo
  echo "Finished case: $case_name"
  echo "End time: $(date)"
  echo
}

cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Logs folder : $LOG_DIR"
echo "Main epochs : $EPOCHS_MAIN"
echo "Structured fine-tune epochs: $FINE_TUNE_EPOCHS_STRUCTURED"
echo

run_case \
  "baseline" \
  "$PROJECT_ROOT/baseline/code" \
  "$LOG_DIR/baseline_10ep.log" \
  EPOCHS="$EPOCHS_MAIN"

run_case \
  "mixed_precision" \
  "$PROJECT_ROOT/mixed_precision/code" \
  "$LOG_DIR/mixed_precision_10ep.log" \
  EPOCHS="$EPOCHS_MAIN"

run_case \
  "quantization" \
  "$PROJECT_ROOT/quantization/code" \
  "$LOG_DIR/quantization_10ep.log" \
  EPOCHS="$EPOCHS_MAIN"

run_case \
  "qat" \
  "$PROJECT_ROOT/qat/code" \
  "$LOG_DIR/qat_10ep.log" \
  EPOCHS="$EPOCHS_MAIN"

run_case \
  "pruning" \
  "$PROJECT_ROOT/pruning/code" \
  "$LOG_DIR/pruning_10ep.log" \
  EPOCHS="$EPOCHS_MAIN"

run_case \
  "structured_pruning" \
  "$PROJECT_ROOT/structured_pruning/code" \
  "$LOG_DIR/structured_pruning_10ep.log" \
  EPOCHS="$EPOCHS_MAIN" \
  FINE_TUNE_EPOCHS="$FINE_TUNE_EPOCHS_STRUCTURED"

cd "$PROJECT_ROOT"

echo "================================================================================"
echo "REBUILDING REPORTS"
echo "================================================================================"

python3 shared/make_all_experiments_report.py
python3 shared/make_all_experiments_report_enriched.py
python3 shared/make_compression_plots.py
python3 shared/make_compression_plots_enriched.py

echo
echo "================================================================================"
echo "FINAL OUTPUT CHECK"
echo "================================================================================"

echo
echo "---- Main report ----"
sed -n '1,40p' reports/comparison_all_experiments.md || true

echo
echo "---- Enriched report ----"
sed -n '1,40p' reports/comparison_all_experiments_enriched.md || true

echo
echo "---- Plot folders ----"
find reports/plots -maxdepth 1 -type f | sort || true
find reports/plots_enriched -maxdepth 1 -type f | sort || true

echo
echo "All runs and report generation finished successfully."
echo "Completed at: $(date)"
