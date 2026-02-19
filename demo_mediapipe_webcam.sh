#!/usr/bin/env bash
set -euo pipefail

# Usage: ./demo_mediapipe_webcam.sh [output_folder]

OUTPUT_FOLDER="${1:-results_webcam}"

echo "[INFO] Starting webcam demo"
echo "[INFO] Output folder: $OUTPUT_FOLDER"
echo "[INFO] Press ESC to stop"

python demo_mediapipe.py \
  --input webcam \
  --out_folder "$OUTPUT_FOLDER" \
  --device cuda:0 \
  --show
