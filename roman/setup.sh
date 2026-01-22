#!/bin/bash
set -euo pipefail

echo "[ROMAN setup] Start"

# Ga naar de gemounte ROMAN-repo
cd /workspace/roman

# Voor de zekerheid: oude/generieke clipperpy weggooien
echo "[ROMAN setup] Verwijder eventueel oude clipperpy"
pip uninstall -y clipperpy >/dev/null 2>&1 || true

# ROMAN install.sh runt:
#  - CLIPPER (juiste versie + bindings)
#  - Kimera-RPGO als dependency
echo "[ROMAN setup] Run install.sh"
chmod +x install.sh
./install.sh

# ROMAN als editable package installeren
echo "[ROMAN setup] pip install -e ."
pip install --no-cache-dir -e .

# Kimera-RPGO volledig builden (binairen zoals RpgoReadG2o)
echo "[ROMAN setup] Build Kimera-RPGO"
cd /workspace/roman/dependencies/Kimera-RPGO
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"

# Zorg dat /usr/local/lib in LD_LIBRARY_PATH zit (voor gtsam/clipper/kimera)
# Dit is vaak al zo, maar voor de zekerheid:
echo "[ROMAN setup] LD_LIBRARY_PATH aanvullen"
if ! grep -q "LD_LIBRARY_PATH" /root/.bashrc 2>/dev/null; then
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib' >> /root/.bashrc
fi

echo "[ROMAN setup] Klaar!"

