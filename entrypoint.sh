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

# Optional: run all_runs.sh
ALL_RUNS="/workspace/roman/all_runs.sh"

if [ -x "$ALL_RUNS" ]; then
  # Only prompt if stdin is a TTY (interactive)
  if [ -t 0 ]; then
    read -rp "Run all_runs.sh now? [y/N]: " answer
    case "$answer" in
      y|Y|yes|YES)
        echo ">> Running all_runs.sh"
        bash "$ALL_RUNS"
        ;;
      *)
        echo ">> Skipping all_runs.sh"
        ;;
    esac
  else
    echo ">> Non-interactive shell, skipping all_runs.sh"
  fi
fi

exec "$@"

