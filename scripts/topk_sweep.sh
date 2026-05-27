#!/usr/bin/env bash
# topk_sweep.sh -- Run eval_golden_set.py with top_k_stage1 in [20, 50, 100, 150].
# Single-Instance-Policy: runs sequentially, not in parallel.
# Results tracked in MLflow under experiment_group=topk_sweep.
#
# Usage: bash scripts/topk_sweep.sh
# Requires: set -a && source .env && set +a first, or run from a shell that has
#           DATABASE_URL, OLLAMA_BASE_URL, MLFLOW_TRACKING_URI set.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_golden_set.py"

export PYTHONPATH="${REPO_ROOT}/src"

if [ -z "$DATABASE_URL" ]; then
  echo "ERROR: DATABASE_URL not set. Run: set -a && source .env && set +a"
  exit 1
fi

if [ -z "$OLLAMA_BASE_URL" ]; then
  echo "ERROR: OLLAMA_BASE_URL not set."
  exit 1
fi

TOP_K_VALUES=(20 50 100 150)
# Stage 1+2 only -- no LLM, ~5 min per run, no Ollama dependency
# Run names suffixed _s12 to distinguish from LLM runs

# Accumulators for summary table
declare -a SUMMARY_ROWS

echo "============================================================"
echo "topk_sweep -- Stage 1+2 only (no LLM)"
echo "top_k values: ${TOP_K_VALUES[*]}"
echo "============================================================"

for K in "${TOP_K_VALUES[@]}"; do
  RUN_NAME="topk_sweep_${K}_s12"
  echo ""
  echo "------------------------------------------------------------"
  echo "Running: top_k_stage1=${K}  run_name=${RUN_NAME}"
  echo "------------------------------------------------------------"

  T0=$(date +%s)
  python3 "${EVAL_SCRIPT}" \
    --run-name     "${RUN_NAME}" \
    --top-k-stage1 "${K}" \
    2>&1 | tee "/tmp/topk_sweep_${K}.log"
  EXIT_CODE=$?
  T1=$(date +%s)
  ELAPSED=$((T1 - T0))

  if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "ERROR: run topk_sweep_${K} exited with code ${EXIT_CODE}"
    SUMMARY_ROWS+=("${K}|${ELAPSED}|ERROR|ERROR|ERROR|ERROR")
    continue
  fi

  # Parse key metrics from log
  R5=$(grep    "recall_at_5 " "/tmp/topk_sweep_${K}.log" | grep -v "_" | awk '{print $2}' | tail -1)
  SR5=$(grep   "soft_recall_at_5 " "/tmp/topk_sweep_${K}.log" | awk '{print $2}' | tail -1)
  CATA=$(grep  "cat_A_stage1_miss " "/tmp/topk_sweep_${K}.log" | awk '{print $2}' | tail -1)
  P1=$(grep    "p_at_1_llm " "/tmp/topk_sweep_${K}.log" | awk '{print $2}' | tail -1)

  SUMMARY_ROWS+=("${K}|${ELAPSED}|${R5}|${SR5}|${CATA}|${P1}")

  echo "Done: top_k=${K} elapsed=${ELAPSED}s"
done

echo ""
echo "============================================================"
echo "SWEEP SUMMARY"
echo "top_k | elapsed_s | recall@5 | soft_recall@5 | cat_A | p_at_1_llm"
echo "------+----------+---------+--------------+-------+-----------"
for ROW in "${SUMMARY_ROWS[@]}"; do
  IFS='|' read -r K ELAPSED R5 SR5 CATA P1 <<< "${ROW}"
  printf "  %3s | %9s | %8s | %13s | %5s | %10s\n" \
    "${K}" "${ELAPSED}" "${R5}" "${SR5}" "${CATA}" "${P1}"
done
echo "============================================================"
