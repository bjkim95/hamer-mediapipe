import argparse
import os
import time
import shutil
import subprocess

import cv2
import mediapipe as mp
import numpy as np
import torch
from skimage.filters import gaussian

from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.utils import (
    convert_cvimg_to_tensor,
    expand_to_aspect_ratio,
    generate_image_patch_cv2,
)
from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
from hamer.utils.renderer import Renderer, cam_crop_to_full

try:
    from pytorch3d.renderer import (
        MeshRasterizer,
        MeshRendererWithFragments,
        PerspectiveCameras,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes

    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help='Image path, video path, or "webcam"')
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path override")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_folder", type=str, default="out")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save_video", action="store_true", help="Save output as video")
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="Output video FPS (webcam: 0 means auto-estimate from processing speed)",
    )
    parser.add_argument("--show_fps", action="store_true", help="Overlay real-time FPS at top-right")
    parser.add_argument(
        "--hand_colors",
        action="store_true",
        help="Use distinct colors for left/right hands (green/red)",
    )
    parser.add_argument(
        "--mesh_only",
        action="store_true",
        help="Render hand mesh only on black background (no image overlay)",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="pyrender",
        choices=["pyrender", "pytorch3d"],
        help="Renderer backend for mesh rendering",
    )
    return parser.parse_args()


def draw_fps_overlay(image_bgr: np.ndarray, fps_value: float) -> np.ndarray:
    if not image_bgr.flags["C_CONTIGUOUS"]:
        image_bgr = np.ascontiguousarray(image_bgr)

    text = f"FPS: {fps_value:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    margin = 12

    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, image_bgr.shape[1] - text_w - margin)
    y = margin + text_h

    cv2.rectangle(
        image_bgr,
        (x - 8, y - text_h - 8),
        (x + text_w + 8, y + baseline + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image_bgr, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return image_bgr


def should_quit_from_key(key_code: int) -> bool:
    key = key_code & 0xFF
    return key in (27, ord("q"), ord("Q"))


def resolve_checkpoint(checkpoint_override: str | None) -> str:
    if checkpoint_override:
        if not os.path.exists(checkpoint_override):
            raise FileNotFoundError(f"Checkpoint override not found: {checkpoint_override}")
        print(f"[INFO] Using checkpoint override: {checkpoint_override}")
        return checkpoint_override

    new_checkpoint = os.path.join(CACHE_DIR_HAMER, "hamer_ckpts", "checkpoints", "new_hamer_weights.ckpt")
    if os.path.exists(new_checkpoint):
        print(f"[INFO] Using upgraded HaMeR checkpoint: {new_checkpoint}")
        return new_checkpoint

    if os.path.exists(DEFAULT_CHECKPOINT):
        print(f"[WARN] Upgraded checkpoint not found. Falling back to legacy checkpoint: {DEFAULT_CHECKPOINT}")
        return DEFAULT_CHECKPOINT

    raise FileNotFoundError(
        "No usable checkpoint found. Expected one of:\n"
        f"  - {new_checkpoint}\n"
        f"  - {DEFAULT_CHECKPOINT}\n"
        "Run `bash fetch_demo_data.sh` and place new_hamer_weights.ckpt manually if needed."
    )


def extract_mediapipe_hands(results, image_shape):
    bboxes, handedness = [], []
    if not results.multi_hand_landmarks:
        return bboxes, handedness

    for idx, hand in enumerate(results.multi_hand_landmarks):
        x_list = [lm.x for lm in hand.landmark]
        y_list = [lm.y for lm in hand.landmark]
        x1 = int(min(x_list) * image_shape[1])
        x2 = int(max(x_list) * image_shape[1])
        y1 = int(min(y_list) * image_shape[0])
        y2 = int(max(y_list) * image_shape[0])
        bboxes.append([x1, y1, x2, y2])

        label = results.multi_handedness[idx].classification[0].label
        handedness.append(1 if label == "Left" else 0)

    return bboxes, handedness


def prepare_input_from_box(cfg, img_cv2, box, right, rescale_factor=2.0):
    img_size = cfg.MODEL.IMAGE_SIZE
    mean = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
    std = 255.0 * np.array(cfg.MODEL.IMAGE_STD)

    center = (box[2:4] + box[0:2]) / 2.0
    scale = rescale_factor * (box[2:4] - box[0:2]) / 200.0
    bbox_shape = cfg.MODEL.get("BBOX_SHAPE", None)
    bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=bbox_shape).max()

    patch_width = patch_height = img_size
    flip = right == 0

    cvimg = img_cv2.copy()
    downsampling_factor = ((bbox_size / patch_width) / 2.0)
    if downsampling_factor > 1.1:
        cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1.0) / 2.0, channel_axis=2, preserve_range=True)

    img_patch_cv, _ = generate_image_patch_cv2(
        cvimg,
        center[0],
        center[1],
        bbox_size,
        bbox_size,
        patch_width,
        patch_height,
        flip,
        1.0,
        0,
        border_mode=cv2.BORDER_CONSTANT,
    )
    img_patch_cv = img_patch_cv[:, :, ::-1]
    img_patch = convert_cvimg_to_tensor(img_patch_cv)
    if not isinstance(img_patch, torch.Tensor):
        img_patch = torch.from_numpy(img_patch).float()

    for n_c in range(min(img_patch.shape[0], 3)):
        img_patch[n_c] = (img_patch[n_c] - mean[n_c]) / std[n_c]

    return {
        "img": img_patch,
        "box_center": torch.tensor(center, dtype=torch.float32),
        "box_size": torch.tensor(bbox_size, dtype=torch.float32),
        "img_size": torch.tensor([cvimg.shape[1], cvimg.shape[0]], dtype=torch.float32),
        "right": torch.tensor(right, dtype=torch.float32),
    }


def setup_input_source(input_path: str):
    is_webcam = input_path == "webcam"
    is_video = False
    cap = None

    if is_webcam:
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Failed to read from webcam")
        return cap, frame, is_webcam, is_video, None

    if input_path.lower().endswith(VIDEO_EXTENSIONS):
        is_video = True
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read first frame from video: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video loaded: {total_frames} frames")
        return cap, frame, is_webcam, is_video, total_frames

    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    return cap, frame, is_webcam, is_video, None


def get_output_stem(input_path: str, is_webcam: bool) -> str:
    if is_webcam:
        return "webcam"
    return os.path.splitext(os.path.basename(input_path))[0]


def make_batch(model_cfg, frame, boxes_np, right_np, device):
    samples = [prepare_input_from_box(model_cfg, frame, box, right, rescale_factor=2.0) for box, right in zip(boxes_np, right_np)]
    return {
        "img": torch.stack([s["img"] for s in samples]).to(device),
        "box_center": torch.stack([s["box_center"] for s in samples]).to(device),
        "box_size": torch.stack([s["box_size"] for s in samples]).to(device),
        "img_size": torch.stack([s["img_size"] for s in samples]).to(device),
        "right": torch.stack([s["right"] for s in samples]).to(device),
    }


def run_hamer_inference(model, batch_dict, scaled_focal):
    with torch.no_grad():
        out = model(batch_dict)

    pred_cam = out["pred_cam"]
    pred_cam[:, 1] *= (2 * batch_dict["right"] - 1)
    cam_t_full = cam_crop_to_full(
        pred_cam,
        batch_dict["box_center"],
        batch_dict["box_size"],
        batch_dict["img_size"],
        scaled_focal,
    ).cpu().numpy()

    all_verts, all_cam_t, all_right = [], [], []
    for i in range(len(cam_t_full)):
        verts = out["pred_vertices"][i].cpu().numpy()
        verts[:, 0] *= (2 * batch_dict["right"][i].item() - 1)
        all_verts.append(verts)
        all_cam_t.append(cam_t_full[i])
        all_right.append(batch_dict["right"][i].item())

    return all_verts, all_cam_t, all_right


def render_hands_pytorch3d_multi(
    verts_list,
    cam_t_list,
    faces_right,
    faces_left,
    render_res,
    focal_length,
    is_right_list,
    use_hand_colors=False,
    mesh_base_color=LIGHT_BLUE,
):
    if not PYTORCH3D_AVAILABLE:
        raise ImportError(
            "PyTorch3D renderer selected, but pytorch3d is not installed. "
            "Install it and rerun with --renderer pytorch3d."
        )
    if len(verts_list) == 0:
        raise ValueError("No vertices provided for rendering.")

    width = int(render_res[0])
    height = int(render_res[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    focal_tensor = torch.tensor([[float(focal_length), float(focal_length)]], device=device, dtype=torch.float32)
    princpt_tensor = torch.tensor([[width / 2.0, height / 2.0]], device=device, dtype=torch.float32)
    image_size_tensor = torch.tensor([[height, width]], device=device, dtype=torch.int64)

    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
    )
    lights = PointLights(
        device=device,
        location=[[0.0, 0.0, 0.0]],
        ambient_color=((0.7, 0.7, 0.7),),
        diffuse_color=((0.6, 0.6, 0.6),),
        specular_color=((0.0, 0.0, 0.0),),
    )
    cameras = PerspectiveCameras(
        R=torch.eye(3, device=device).unsqueeze(0),
        T=torch.zeros(1, 3, device=device),
        focal_length=focal_tensor,
        principal_point=princpt_tensor,
        in_ndc=False,
        image_size=image_size_tensor,
        device=device,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(device)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)

    left_color = torch.tensor((0.5, 0.8, 0.5), dtype=torch.float32, device=device)
    right_color = torch.tensor((0.8, 0.4, 0.4), dtype=torch.float32, device=device)
    default_color = torch.tensor(mesh_base_color, dtype=torch.float32, device=device)

    all_verts = []
    all_faces = []
    all_colors = []
    vert_offset = 0
    coord_convert = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float32, device=device))

    for verts_np, cam_t_np, is_right in zip(verts_list, cam_t_list, is_right_list):
        verts_tensor = torch.from_numpy(verts_np.astype(np.float32)).to(device)
        cam_t_tensor = torch.from_numpy(cam_t_np.astype(np.float32)).to(device)
        verts_cam = (verts_tensor + cam_t_tensor) @ coord_convert.T

        if int(is_right):
            faces_np = faces_right
            hand_color = right_color if use_hand_colors else default_color
        else:
            faces_np = faces_left
            hand_color = left_color if use_hand_colors else default_color

        faces_tensor = torch.from_numpy(faces_np.astype(np.int64)).to(device) + vert_offset
        vert_offset += verts_cam.shape[0]

        all_verts.append(verts_cam)
        all_faces.append(faces_tensor)
        all_colors.append(hand_color.view(1, 3).expand(verts_cam.shape[0], 3))

    mesh = Meshes(
        verts=torch.cat(all_verts, dim=0).unsqueeze(0),
        faces=torch.cat(all_faces, dim=0).unsqueeze(0),
        textures=TexturesVertex(torch.cat(all_colors, dim=0).unsqueeze(0)),
    )

    with torch.no_grad():
        images, fragments = renderer(mesh)

    final_rgb = images[0, :, :, :3]
    alpha = (fragments.pix_to_face[0, :, :, 0] >= 0).float().unsqueeze(-1)
    final_rgba = torch.cat([final_rgb.clamp(0.0, 1.0), alpha], dim=-1)
    return final_rgba.cpu().numpy()


def render_frame(renderer, frame, all_verts, all_cam_t, all_right, img_size, scaled_focal, args):
    bg_color = (0, 0, 0) if args.mesh_only else (1, 1, 1)
    if args.renderer == "pytorch3d":
        cam_view = render_hands_pytorch3d_multi(
            all_verts,
            all_cam_t,
            renderer.faces,
            renderer.faces_left,
            img_size,
            scaled_focal,
            all_right,
            use_hand_colors=args.hand_colors,
            mesh_base_color=LIGHT_BLUE,
        )
        if args.mesh_only:
            # For mesh-only mode with PyTorch3D, enforce black background.
            cam_view[:, :, :3] = cam_view[:, :, :3] * cam_view[:, :, 3:]
    else:
        cam_view = renderer.render_rgba_multiple(
            all_verts,
            cam_t=all_cam_t,
            render_res=img_size,
            is_right=all_right,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=bg_color,
            focal_length=scaled_focal,
            use_hand_colors=args.hand_colors,
        )

    if args.mesh_only:
        return (255 * cam_view[:, :, :3]).astype(np.uint8)

    input_img = frame.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
    overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
    return (255 * overlay).astype(np.uint8)


def start_ffmpeg_writer(out_folder: str, output_stem: str, output_fps: float, width: int, height: int):
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH. Install ffmpeg to use --save_video.")

    output_video_path = os.path.join(out_folder, f"{output_stem}.mp4")
    ffmpeg_cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(output_fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output_video_path,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin pipe.")
    return proc, output_video_path


def main():
    args = parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    if args.renderer == "pytorch3d" and not PYTORCH3D_AVAILABLE:
        raise ImportError(
            "PyTorch3D renderer selected but pytorch3d is unavailable. "
            "Install pytorch3d or run with --renderer pyrender."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    download_models(CACHE_DIR_HAMER)
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    model, model_cfg = load_hamer(checkpoint_path)
    model = model.to(device).eval()

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    cap, frame, is_webcam, is_video, total_frames = setup_input_source(args.input)
    output_stem = get_output_stem(args.input, is_webcam)
    if args.show:
        print("[INFO] Press q (or ESC) to stop.")

    img_h, img_w = frame.shape[:2]
    render_res = (img_w, img_h)
    renderer = Renderer(model_cfg, faces=model.mano.faces, render_res=render_res)

    default_fps = 30.0
    output_fps = float(args.fps if args.fps > 0 else default_fps)
    ffmpeg_proc = None
    output_video_path = None
    webcam_auto_fps = args.save_video and is_webcam and args.fps <= 0
    webcam_probe_frames = []
    webcam_probe_start = None
    webcam_probe_target = 45

    if is_video:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        output_fps = input_fps if input_fps > 0 else float(args.fps if args.fps > 0 else default_fps)
    if args.save_video and is_video:
        if is_video:
            print(f"Input video FPS: {output_fps:.2f}")
        ffmpeg_proc, output_video_path = start_ffmpeg_writer(args.out_folder, output_stem, output_fps, img_w, img_h)
    elif args.save_video and is_webcam and args.fps > 0:
        print(f"Webcam output FPS (manual): {output_fps:.2f}")
        ffmpeg_proc, output_video_path = start_ffmpeg_writer(args.out_folder, output_stem, output_fps, img_w, img_h)
    elif webcam_auto_fps:
        print("[INFO] Webcam output FPS: auto (estimating from live processing speed)")

    def write_frame_output(frame_bgr: np.ndarray):
        nonlocal ffmpeg_proc, output_video_path, output_fps, webcam_probe_start, webcam_probe_frames
        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.write(frame_bgr.tobytes())
            return

        if webcam_auto_fps:
            if webcam_probe_start is None:
                webcam_probe_start = time.time()
            webcam_probe_frames.append(np.ascontiguousarray(frame_bgr))
            if len(webcam_probe_frames) >= webcam_probe_target:
                elapsed = max(time.time() - webcam_probe_start, 1e-6)
                estimated_fps = (len(webcam_probe_frames) - 1) / elapsed
                output_fps = float(min(max(estimated_fps, 1.0), 60.0))
                print(f"[INFO] Webcam output FPS (auto): {output_fps:.2f}")
                ffmpeg_proc, output_video_path = start_ffmpeg_writer(args.out_folder, output_stem, output_fps, img_w, img_h)
                for buffered_frame in webcam_probe_frames:
                    ffmpeg_proc.stdin.write(buffered_frame.tobytes())
                webcam_probe_frames.clear()
            return

        out_name = f"{output_stem}.jpg" if not is_webcam else f"frame_{frame_id:06d}.jpg"
        out_path = os.path.join(args.out_folder, out_name)
        cv2.imwrite(out_path, frame_bgr)

    frame_id = 0
    prev_frame_time = None
    while True:
        if is_webcam:
            ok, frame = cap.read()
            if not ok:
                break
        elif is_video:
            if frame_id > 0:
                ok, frame = cap.read()
                if not ok:
                    break
            print(f"Processing frame {frame_id + 1}/{total_frames}")
        elif frame_id > 0:
            break

        if is_webcam:
            frame = cv2.flip(frame, 1)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        realtime_fps = 0.0
        if prev_frame_time is not None:
            frame_dt = t0 - prev_frame_time
            if frame_dt > 0:
                realtime_fps = 1.0 / frame_dt
        prev_frame_time = t0
        results = mp_hands.process(img_rgb)
        bboxes, handed_list = extract_mediapipe_hands(results, frame.shape)
        t1 = time.time()

        boxes_np = np.array(bboxes, dtype=np.float32)
        right_np = np.array(handed_list, dtype=np.float32)

        if boxes_np.ndim != 2 or boxes_np.shape[0] == 0:
            out_bgr = np.zeros_like(frame) if args.mesh_only else frame
            if args.show_fps:
                out_bgr = draw_fps_overlay(out_bgr, realtime_fps)
            write_frame_output(out_bgr)
            if args.show:
                cv2.imshow("HaMeR + MediaPipe", out_bgr)
                if should_quit_from_key(cv2.waitKey(1)):
                    break
            frame_id += 1
            continue

        t2 = time.time()
        img_size = np.array(render_res)
        scaled_focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()

        t_crop_start = time.time()
        batch_dict = make_batch(model_cfg, frame, boxes_np, right_np, device)
        t_crop_end = time.time()

        t_model_start = time.time()
        all_verts, all_cam_t, all_right = run_hamer_inference(model, batch_dict, scaled_focal)
        t_model_end = time.time()

        print(f"[Crop]     {t_crop_end - t_crop_start:.3f}s")
        print(f"[Model]    {t_model_end - t_model_start:.3f}s")

        t4 = time.time()
        out_img = render_frame(renderer, frame, all_verts, all_cam_t, all_right, img_size, scaled_focal, args)
        out_bgr = out_img[:, :, ::-1]
        if args.show_fps:
            out_bgr = draw_fps_overlay(out_bgr, realtime_fps)
        write_frame_output(out_bgr)

        if args.show:
            cv2.imshow("HaMeR + MediaPipe", out_bgr)
            if should_quit_from_key(cv2.waitKey(0 if not is_webcam else 1)):
                break
        t5 = time.time()

        frame_id += 1
        if is_webcam or is_video:
            print(f"[MediaPipe] {t1 - t0:.3f}s | [HaMeR] {t_model_end - t2:.3f}s | [Render] {t5 - t4:.3f}s | [Total] {time.time() - t0:.3f}s")

    if is_webcam or is_video:
        cap.release()
    if webcam_auto_fps and ffmpeg_proc is None and webcam_probe_frames:
        elapsed = max((time.time() - webcam_probe_start) if webcam_probe_start is not None else 0.0, 1e-6)
        estimated_fps = (len(webcam_probe_frames) - 1) / elapsed if len(webcam_probe_frames) > 1 else default_fps
        output_fps = float(min(max(estimated_fps, 1.0), 60.0))
        print(f"[INFO] Webcam output FPS (auto-final): {output_fps:.2f}")
        ffmpeg_proc, output_video_path = start_ffmpeg_writer(args.out_folder, output_stem, output_fps, img_w, img_h)
        for buffered_frame in webcam_probe_frames:
            ffmpeg_proc.stdin.write(buffered_frame.tobytes())
        webcam_probe_frames.clear()
    if ffmpeg_proc is not None:
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.wait()
        if ffmpeg_proc.returncode != 0:
            stderr = ffmpeg_proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"ffmpeg failed while creating video: {stderr}")
        ffmpeg_proc.stderr.close()
    if args.show:
        cv2.destroyAllWindows()

    if is_webcam or is_video:
        if args.save_video and output_video_path is not None:
            print(f"\nProcessing complete! Video saved to {output_video_path}")
        elif is_video:
            print(f"\nProcessing complete! {frame_id} video frames processed (no frame files saved).")
        else:
            print(f"\nProcessing complete! {frame_id} webcam frames processed.")
    else:
        print(f"\nProcessing complete! {frame_id} frames saved to {args.out_folder}")


if __name__ == "__main__":
    main()
