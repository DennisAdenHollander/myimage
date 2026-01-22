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

exec "$@"

