#!/usr/bin/env bash
set -euo pipefail

# Always run from this script directory so _DATA is created locally.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ARCHIVE="hamer_demo_data.tar.gz"
URL_GDRIVE="https://drive.google.com/uc?id=1mv7CUAnm73oKsEEG1xE3xH2C_oqcFSzT"
URL_MIRROR="https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"

download_archive() {
  echo "[INFO] Downloading demo data archive..."

  rm -f "${ARCHIVE}"

  if command -v gdown >/dev/null 2>&1; then
    echo "[INFO] Try 1/3: gdown (Google Drive)"
    if gdown "${URL_GDRIVE}" -O "${ARCHIVE}"; then
      return 0
    fi
    echo "[WARN] gdown download failed. Trying mirror URL..."
  else
    echo "[WARN] gdown not found. Trying mirror URL..."
  fi

  if command -v wget >/dev/null 2>&1; then
    echo "[INFO] Try 2/3: wget (UT mirror)"
    if wget -O "${ARCHIVE}" "${URL_MIRROR}"; then
      return 0
    fi
    echo "[WARN] wget download failed. Trying curl..."
  fi

  if command -v curl >/dev/null 2>&1; then
    echo "[INFO] Try 3/3: curl (UT mirror)"
    if curl -L "${URL_MIRROR}" -o "${ARCHIVE}"; then
      return 0
    fi
  fi

  echo "[ERROR] Failed to download ${ARCHIVE} via all methods." >&2
  return 1
}

download_archive

echo "[INFO] Extracting archive..."
tar --warning=no-unknown-keyword --exclude=".*" -xvf "${ARCHIVE}"

# MediaPipe-only repo: remove assets not used by demo_mediapipe.py.
echo "[INFO] Removing unused assets (ViTPose checkpoint + training metadata)..."
rm -rf _DATA/vitpose_ckpts
rm -f _DATA/hamer_ckpts/dataset_config.yaml

echo "[INFO] Cleaning up archive..."
rm -f "${ARCHIVE}"

echo "[INFO] Remaining required files:"
echo "  - _DATA/hamer_ckpts/checkpoints/hamer.ckpt"
echo "  - _DATA/hamer_ckpts/model_config.yaml"
echo "  - _DATA/data/mano_mean_params.npz"
echo
echo "[INFO] Place MANO_RIGHT.pkl manually at:"
echo "  - _DATA/data/mano/MANO_RIGHT.pkl"
