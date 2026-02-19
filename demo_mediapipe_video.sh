#!/usr/bin/env bash
set -euo pipefail

# Usage: ./demo_mediapipe_video.sh <video_path> [output_folder]

if [ $# -lt 1 ]; then
  echo "[ERROR] Please provide a video path" >&2
  echo "Usage: $0 <video_path> [output_folder]" >&2
  exit 1
fi

VIDEO_PATH="$1"
OUTPUT_FOLDER="${2:-demo_out}"

if [ ! -f "$VIDEO_PATH" ]; then
  echo "[ERROR] Video file not found: $VIDEO_PATH" >&2
  exit 1
fi

echo "[INFO] Input video: $VIDEO_PATH"
echo "[INFO] Output folder: $OUTPUT_FOLDER"

python demo_mediapipe.py \
  --input "$VIDEO_PATH" \
  --out_folder "$OUTPUT_FOLDER" \
  --device cuda:0 \
  --save_video \
  --hand_colors

echo "[INFO] Completed. Results saved to $OUTPUT_FOLDER"
