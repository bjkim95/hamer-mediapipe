#!/usr/bin/env bash
set -euo pipefail

# Usage: ./demo_mediapipe.sh <image_path> [output_folder]

if [ $# -lt 1 ]; then
  echo "[ERROR] Please provide an image path" >&2
  echo "Usage: $0 <image_path> [output_folder]" >&2
  exit 1
fi

IMAGE_PATH="$1"
OUTPUT_FOLDER="${2:-demo_out}"

if [ ! -f "$IMAGE_PATH" ]; then
  echo "[ERROR] Image file not found: $IMAGE_PATH" >&2
  exit 1
fi

echo "[INFO] Input image: $IMAGE_PATH"
echo "[INFO] Output folder: $OUTPUT_FOLDER"

python demo_mediapipe.py \
  --input "$IMAGE_PATH" \
  --out_folder "$OUTPUT_FOLDER" \
  --device cuda:0 \
  --show
