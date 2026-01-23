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

# Optional: run All_runs.sh
ALL_RUNS="/workspace/roman/roman_mount/runs/All_runs.sh"

echo ">> Checking All_runs at: $ALL_RUNS"
if [ -f "$ALL_RUNS" ]; then
  chmod +x "$ALL_RUNS" || true
fi

if [ -x "$ALL_RUNS" ]; then
  # If env var is set, don't prompt
  if [ "${RUN_ALL_RUNS:-0}" = "1" ]; then
    echo ">> RUN_ALL_RUNS=1, running All_runs.sh"
    bash "$ALL_RUNS"
  # Otherwise prompt only if interactive
  elif [ -t 0 ]; then
    read -rp "Run All_runs.sh now? [y/N]: " answer
    case "$answer" in
      y|Y|yes|YES)
        echo ">> Running All_runs.sh"
        bash "$ALL_RUNS"
        ;;
      *)
        echo ">> Skipping All_runs.sh"
        ;;
    esac
  else
    echo ">> Non-interactive shell, skipping All_runs.sh (set RUN_ALL_RUNS=1 to force)"
  fi
else
  echo ">> All_runs.sh not found or not executable"
fi

exec "$@"

