#!/usr/bin/env bash
set -euo pipefail

#############################
# Patch Kimera (idempotent)
#############################
sed -i 's/params\.diagonalDamping = true;/params.setDiagonalDamping(true);/' \
  /workspace/roman/dependencies/Kimera-RPGO/src/GenericSolver.cpp || true

sed -i 's/lmParams\.diagonalDamping = params_\.lm_diagonal_damping;/lmParams.setDiagonalDamping(params_.lm_diagonal_damping);/' \
  /workspace/roman/dependencies/Kimera-RPGO/src/RobustSolver.cpp || true

#############################
# Initialize ROMAN (once)
#############################
MARKER="/opt/.roman_initialized"

if [ ! -f "$MARKER" ]; then
  echo ">> Running initialize.sh (first time only)"
  cd /workspace/roman
  ./initialize.sh
  touch "$MARKER"
else
  echo ">> initialize.sh already completed"
fi

#############################
# Run All_runs.sh (always)
#############################
ALL_RUNS="/workspace/roman/roman_mount/runs/All_runs.sh"

echo ">> Running All_runs.sh"

if [ ! -f "$ALL_RUNS" ]; then
  echo "ERROR: $ALL_RUNS not found"
  exit 1
fi

chmod +x "$ALL_RUNS" || true
bash "$ALL_RUNS"

#############################
# Hand over to CMD
#############################
exec "$@"

