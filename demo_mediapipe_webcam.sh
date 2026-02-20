#!/usr/bin/env bash
set -euo pipefail

# Usage: ./demo_mediapipe_webcam.sh [output_folder] [--show-fps|--show_fps]

OUTPUT_FOLDER="results_webcam"
SHOW_FPS_FLAG=""
FPS_VALUE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --show-fps|--show_fps)
      SHOW_FPS_FLAG="--show_fps"
      shift
      ;;
    --fps)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --fps requires a value"
        echo "Usage: $0 [output_folder] [--show-fps|--show_fps] [--fps <value>]"
        exit 1
      fi
      FPS_VALUE="$2"
      shift 2
      ;;
    -*)
      echo "[ERROR] Unknown option: $1"
      echo "Usage: $0 [output_folder] [--show-fps|--show_fps] [--fps <value>]"
      exit 1
      ;;
    *)
      if [[ "$OUTPUT_FOLDER" == "results_webcam" ]]; then
        OUTPUT_FOLDER="$1"
      else
        echo "[ERROR] Too many positional arguments: $1"
        echo "Usage: $0 [output_folder] [--show-fps|--show_fps] [--fps <value>]"
        exit 1
      fi
      shift
      ;;
  esac
done

echo "[INFO] Starting webcam demo"
echo "[INFO] Output folder: $OUTPUT_FOLDER"
echo "[INFO] Output format: MP4 (ffmpeg)"
if [[ -n "$FPS_VALUE" ]]; then
  echo "[INFO] Output FPS: manual ($FPS_VALUE)"
else
  echo "[INFO] Output FPS: auto"
fi
if [[ -n "$SHOW_FPS_FLAG" ]]; then
  echo "[INFO] FPS overlay: enabled (top-right)"
fi
echo "[INFO] Press ESC or q to stop"

python demo_mediapipe.py \
  --input webcam \
  --out_folder "$OUTPUT_FOLDER" \
  --device cuda:0 \
  --save_video \
  ${FPS_VALUE:+--fps "$FPS_VALUE"} \
  ${SHOW_FPS_FLAG:+$SHOW_FPS_FLAG} \
  --show
