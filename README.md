# HaMeR + MediaPipe

[![Demo Preview](assets/demo_preview.gif)](assets/demo.mp4)

This repository is a fast, lightweight HaMeR variant using MediaPipe for hand detection and HaMeR for hand mesh inference.
It is adapted from the original HaMeR project: https://github.com/geopavlakos/hamer

## 1) Installation

### Required (Conda)

```bash
conda create -n hamer-mp python=3.10 -y
conda activate hamer-mp

# CUDA 12.4 PyTorch
pip install "torch==2.6.*" "torchvision==0.21.*" --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
```

`chumpy` is installed separately because it may fail under default build isolation.

### Optional (video export)

If you use `--save_video`, install `ffmpeg` on your system.

### Optional (PyTorch3D renderer backend)

```bash
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.9+pt2.6.0cu124
```

## 2) Model Files

```bash
bash fetch_demo_data.sh
```

This script keeps only MediaPipe-inference-required files and removes unused ViTPose assets.

You also need to download the MANO model file manually:

1. Go to the MANO website: https://mano.is.tue.mpg.de
2. Register/login and open the downloads section.
3. Download the MANO model package.
4. Use the **right-hand** model file `MANO_RIGHT.pkl`.

Then place `MANO_RIGHT.pkl` at:

```text
_DATA/data/mano/MANO_RIGHT.pkl
```

For the upgraded HaMeR checkpoint (`new_hamer_weights.ckpt`), download it from:

https://gkarv.github.io/hand-texture-module/

Then place it at:

```text
_DATA/hamer_ckpts/checkpoints/new_hamer_weights.ckpt
```

The demo uses this upgraded checkpoint first when present.

## 3) Quick Start

### Image

```bash
bash demo_mediapipe.sh /path/to/image.jpg demo_out
```

By default, the output image file is saved as `<input_stem>.jpg`.

### Video

```bash
bash demo_mediapipe_video.sh /path/to/video.mp4 demo_out
```

Note: in video mode, frame-wise JPG files are not saved.  
If `--save_video` is enabled, only `<input_stem>.mp4` is written.

### Webcam

```bash
bash demo_mediapipe_webcam.sh results_webcam
```

## 4) Python CLI

```bash
python demo_mediapipe.py \
  --input <image/video/webcam> \
  --out_folder <output_folder> \
  --device cuda:0 \
  [--renderer pyrender|pytorch3d] \
  [--show] \
  [--save_video] \
  [--fps 30] \
  [--hand_colors] \
  [--mesh_only]
```
