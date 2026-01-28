#!/usr/bin/env bash
set -euo pipefail

# Patch (idempotent: run multiple times is ok)
sed -i 's/params\.diagonalDamping = true;/params.setDiagonalDamping(true);/' \
  /workspace/roman/dependencies/Kimera-RPGO/src/GenericSolver.cpp || true

sed -i 's/lmParams\.diagonalDamping = params_\.lm_diagonal_damping;/lmParams.setDiagonalDamping(params_.lm_diagonal_damping);/' \
  /workspace/roman/dependencies/Kimera-RPGO/src/RobustSolver.cpp || true

# Run initialize once per container filesystem
MARKER="/opt/.roman_initialized"
if [ ! -f "$MARKER" ]; then
  cd /workspace/roman
  ./initialize.sh
  touch "$MARKER"
fi

# Optional: run All_runs.sh (NO PROMPT)
ALL_RUNS="/workspace/roman/roman_mount/runs/All_runs.sh"
echo ">> Checking All_runs at: $ALL_RUNS"
if [ -f "$ALL_RUNS" ]; then
  chmod +x "$ALL_RUNS" || true
fi

if [ -x "$ALL_RUNS" ]; then
  if [ "${RUN_ALL_RUNS:-0}" = "1" ]; then
    echo ">> RUN_ALL_RUNS=1, running All_runs.sh"
    bash "$ALL_RUNS"
  else
    echo ">> RUN_ALL_RUNS not set, skipping All_runs.sh"
  fi
else
  echo ">> All_runs.sh not found or not executable"
fi

# Optional: run run_count.sh (NO PROMPT)
RUN_COUNT="/workspace/roman/roman_mount/runs/run_count.sh"
echo ">> Checking run_count at: $RUN_COUNT"
if [ -f "$RUN_COUNT" ]; then
  chmod +x "$RUN_COUNT" || true
fi

if [ -x "$RUN_COUNT" ]; then
  if [ "${RUN_RUN_COUNT:-0}" = "1" ]; then
    echo ">> RUN_RUN_COUNT=1, running run_count.sh"
    bash "$RUN_COUNT"
  else
    echo ">> RUN_RUN_COUNT not set, skipping run_count.sh"
  fi
else
  echo ">> run_count.sh not found or not executable"
fi

exec "$@"

