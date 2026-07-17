import os
# Workaround for duplicate OpenMP runtime loading on some Windows setups.
# Keep this before importing torch/numpy-backed libraries.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import yaml
import torch
import numpy as np
import struct
import imageio
import pickle

from argparse import ArgumentParser, Namespace
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from camera import IntrinsicsCamera
from dataset import FLAMEDataset, FuHeadDataset
from model import FLAMEBindingModel, FuHeadBindingModel
from submodules.flame import FLAME, FlameConfig
from submodules.fuhead import FuHead
from utils import Struct

from comfort import mask as comfort_mask
from comfort import landmarks as comfort_landmarks
from comfort import schema as comfort_schema


def _build_is_face(gaussian_model) -> np.ndarray:
    """Static per-Gaussian fixation mask from the FLAME binding + neck/boundary lists.

    ``face = ALL \\ (neck u boundary)``, lifted through ``binding_face_id`` (the triangle each
    Gaussian is bound to). Topology is fixed, so this is computed once per subject.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    excluded = comfort_mask.load_flame_mask_indices(
        os.path.join(here, "flame_mask", "neck.txt"),
        os.path.join(here, "flame_mask", "boundary.txt"),
    )
    return comfort_mask.build_is_face(
        gaussian_model.binding_face_id, gaussian_model.template_faces, excluded
    )


def parse_args(argv) -> Namespace:
    parser = ArgumentParser(description="Render sequence frames and optional deformed PLY/orbit views.")
    parser.add_argument("--subject", type=str, default="bala", help="Subject name under data_dir.")
    parser.add_argument("--output_dir", type=str, default="output", help="Root output directory.")
    parser.add_argument("--work_name", type=str, required=True, help="Experiment folder name.")
    parser.add_argument("--white_bg", action="store_true", help="Use white background instead of black.")
    parser.add_argument("--alpha", action="store_true", help="Save RGBA render output.")

    parser.add_argument("--batch_size", type=int, default=1, help="Render batch size.")
    parser.add_argument("--io_workers", type=int, default=8, help="Thread workers for image write.")

    parser.add_argument(
        "--save_ply_start",
        type=int,
        default=None,
        help="Start frame index (inclusive) for saving deformed .ply.",
    )
    parser.add_argument(
        "--save_ply_end",
        type=int,
        default=None,
        help="End frame index (inclusive) for saving deformed .ply.",
    )
    parser.add_argument(
        "--save_ply_every",
        type=int,
        default=1,
        help="Save every N frames within [start, end].",
    )
    parser.add_argument(
        "--save_ply_interval",
        type=int,
        default=86,
        help="Fallback interval when start/end are not provided.",
    )

    parser.add_argument("--disable_orbit", action="store_true", help="Skip orbit rendering.")
    parser.add_argument("--orbit_anchor_idx", type=int, default=86, help="Dataset frame index used as orbit anchor.")
    parser.add_argument("--orbit_fps", type=int, default=30, help="Orbit output FPS.")
    parser.add_argument("--orbit_duration_sec", type=float, default=4.0, help="Orbit duration in seconds.")
    parser.add_argument("--orbit_yaw_deg", type=float, default=45.0, help="Max yaw amplitude in degrees.")
    parser.add_argument("--disable_orbit_gif", action="store_true", help="Do not export orbit GIF.")
    return parser.parse_args(argv)


def save_image(image_data: np.ndarray, output_img_path: str, index: int) -> None:
    Image.fromarray(image_data).save(os.path.join(output_img_path, f"{index:05d}.png"))


def _should_save_deformed_ply(
    frame_idx: int,
    save_interval: int,
    save_ply_start: Optional[int],
    save_ply_end: Optional[int],
    save_ply_every: int,
) -> bool:
    if save_ply_start is not None or save_ply_end is not None:
        start = 0 if save_ply_start is None else int(save_ply_start)
        end = frame_idx if save_ply_end is None else int(save_ply_end)
        every = max(int(save_ply_every), 1)
        return (start <= frame_idx <= end) and ((frame_idx - start) % every == 0)
    return frame_idx == 0 or (save_interval > 0 and frame_idx % save_interval == 0)


def save_deformed_ply(gaussian, path: str) -> None:
    xyz = gaussian.xyz[0].detach().cpu().numpy()
    num_pts = xyz.shape[0]

    sh_orig = gaussian.sh[0].detach().cpu().numpy()
    opacity = gaussian.opacity[0].detach().cpu().numpy()
    
    # gaussian.scaling is activated/linear; PLY stores log(cm).
    scaling = np.log(gaussian.scaling[0].detach().cpu().numpy() * 100.0 + 1e-8)
    rotation = gaussian.rotation[0].detach().cpu().numpy()
    sh_final = sh_orig.reshape(num_pts, -1)

    if sh_final.shape[1] < 48:
        sh_final = np.concatenate([sh_final, np.zeros((num_pts, 48 - sh_final.shape[1]))], axis=1)

    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {num_pts}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float nx\nproperty float ny\nproperty float nz\n"
    header += "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    for i in range(45):
        header += f"property float f_rest_{i}\n"
    header += "property float opacity\nproperty float scale_0\nproperty float scale_1\n"
    header += "property float scale_2\nproperty float rot_0\nproperty float rot_1\n"
    header += "property float rot_2\nproperty float rot_3\nend_header\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        for i in range(num_pts):
            data = [
                xyz[i, 0], xyz[i, 1], xyz[i, 2],
                0.0, 0.0, 0.0,
            ] + sh_final[i].tolist() + [
                opacity[i, 0], scaling[i, 0], scaling[i, 1], scaling[i, 2],
                rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3],
            ]
            f.write(struct.pack("<62f", *data))
    print(f"PLY saved: {path} ({num_pts} pts)")


def render_frames(
    gaussian_model,
    dataset,
    camera,                  
    bg_color: torch.Tensor, 
    alpha: bool,            
    output_path: str,
    batch_size: int,
    io_workers: int,
    save_interval: int,     
    save_ply_start: Optional[int], 
    save_ply_end: Optional[int],   
    save_ply_every: int,    
) -> None:

    # 1. export path 
    pt_out_path = os.path.join(output_path, "srd_exports_pt")
    os.makedirs(pt_out_path, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    global_frame_idx = 0
    progress_bar = tqdm(range(len(dataset)), desc="Exporting Raw Gaussian Tensors")

    flame_model = dataset.flame_model

    # Static per-Gaussian fixation mask (topology fixed => computed once, shipped every frame).
    is_face_np = _build_is_face(gaussian_model)
    is_face_t = torch.from_numpy(is_face_np)

    # =================================================================
    # Deform -> export schema-v2 tensors (means/scales/rot/opacity/sh + nose landmark + mask)
    # =================================================================
    with torch.no_grad():
        for data in dataloader:
            mesh = data["mesh"].cpu()
            blend_weight = data["blend_weight"].cpu()
            bs = mesh.shape[0]

            gaussian = gaussian_model.gaussian_deform_batch(mesh, blend_weight.cpu())

            # FLAME landmarks (nose anchor). Fail loudly if the embedding is missing rather
            # than silently exporting a zero anchor (the previous behaviour).
            landmarks3D_batch = comfort_landmarks.compute_landmarks(flame_model, mesh.cpu())

            for batch_idx in range(bs):
                frame_idx = global_frame_idx + batch_idx
                
                xyz_raw = gaussian.xyz[batch_idx] if gaussian.xyz.dim() == 3 else gaussian.xyz.view(-1, 3)
                rot_raw = gaussian.rotation[batch_idx] if gaussian.rotation.dim() == 3 else gaussian.rotation.view(-1, 4)
                opac_raw = gaussian.opacity[batch_idx] if gaussian.opacity.dim() == 3 else gaussian.opacity.view(-1, 1)
                sh_raw = gaussian.sh[batch_idx] if gaussian.sh.dim() == 4 else gaussian.sh.view(-1, 16, 3)
                lmk = landmarks3D_batch[batch_idx]  
                

                # gaussian.scaling is ALREADY activated (linear) via scaling_act=exp in
                # model/gaussian.py; do NOT exp again (this previously double-exponentiated).
                scale_lin = gaussian.scaling[batch_idx] if gaussian.scaling.dim() == 3 else gaussian.scaling.view(-1, 3)

                # meters -> centimetres (the A->B bridge unit).
                xyz_cm = xyz_raw * 100.0
                scale_cm = scale_lin * 100.0
                lmk_cm = lmk * 100.0

                # Schema v2: the raw deformed avatar + nose landmark + static face mask, in cm.
                # No placement offset is baked in here — the frozen z0 is calibrated from the
                # exported sequence by comfort.placement, so the .pt stays a faithful record of
                # the deformed head (the retired "volumetric margin" anchor is gone).
                export_dict = {
                    "means3D": xyz_cm.cpu(),
                    "scales": scale_cm.cpu(),
                    "rotations": rot_raw.cpu(),
                    "opacities": opac_raw.cpu(),
                    "shs": sh_raw.cpu(),
                    "landmarks3D": lmk_cm.cpu(),
                    "is_face": is_face_t,
                    "units": "cm",
                    "nose_index": comfort_landmarks.NOSE_INDEX,
                }

                if frame_idx == 0:
                    # Validate the anchor and the schema once, up front, so a bad landmark
                    # embedding or unit slip fails immediately rather than mid-sequence.
                    comfort_landmarks.resolve_nose_landmark(
                        lmk_cm, comfort_landmarks.NOSE_INDEX, validate=True
                    )
                    comfort_schema.validate_pt(export_dict, strict=True)

                pt_name = f"frame_{frame_idx:05d}.pt"
                torch.save(export_dict, os.path.join(pt_out_path, pt_name))
            
            global_frame_idx += bs
            progress_bar.update(bs)
            
    progress_bar.close()
    print(f"\n[Exporter] SUCCESS: schema-v2 avatar tensors (means/scales/rot/opacity/sh + "
          f"nose landmark + face mask) exported to: {pt_out_path}")


def render_orbit(
    gaussian_model,
    dataset,
    output_path: str,
    bg_color: torch.Tensor,
    anchor_idx: int,
    fps: int,
    duration_sec: float,
    yaw_deg: float,
    make_gif: bool = True,
) -> None:
    print("\n[Start] Rendering Orbit (Novel Views)...")
    orbit_dir = os.path.join(output_path, "render_orbit")
    os.makedirs(orbit_dir, exist_ok=True)

    anchor_idx = int(np.clip(anchor_idx, 0, len(dataset) - 1))
    base_R = torch.tensor(dataset.camera_extri[:3, :3], device="cpu", dtype=torch.float32)
    base_T = torch.tensor(dataset.camera_extri[:3, 3], device="cpu", dtype=torch.float32)
    K = dataset.camera_intri

    data = dataset[anchor_idx]
    mesh = data["mesh"].cpu().unsqueeze(0)
    blend_weight = data["blend_weight"].cpu().unsqueeze(0)
    with torch.no_grad():
        gaussian = gaussian_model.gaussian_deform_batch(mesh, blend_weight.cpu())

    total_frames = max(int(fps * duration_sec), 1)
    yaw_rad = np.deg2rad(yaw_deg)
    frames_for_gif = []

    for i in tqdm(range(total_frames), desc="Orbit Rendering"):
        angle = np.sin(2 * np.pi * i / total_frames) * yaw_rad
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R_y = torch.tensor([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]], device="cpu", dtype=torch.float32)
        curr_R = base_R @ R_y.T

        cam = IntrinsicsCamera(
            K=K,
            R=curr_R.cpu().numpy(),
            T=base_T.cpu().numpy(),
            width=dataset.image_width,
            height=dataset.image_height,
        ).cpu()

    if make_gif and len(frames_for_gif) > 0:
        gif_path = os.path.join(output_path, "orbit_animation.gif")
        print(f"Saving GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames_for_gif, fps=fps)
    print(f"Done! Check {orbit_dir}")


def build_dataset_and_model(config, subject: str, glctx):
    data_path = os.path.join(config["data_dir"], subject)
    if config["template_type"] == "Flame":
        flame_model = FLAME(FlameConfig()).cpu()
        dataset = FLAMEDataset(flame_model, data_path, split="all", **config["dataset"])
        dataset.flame_model = flame_model
        gaussian_model = FLAMEBindingModel(Struct(**config["model"]), flame_model, glctx)
        gaussian_model.load_ply("output/justin/model.ply")

    elif config["template_type"] == "FuHead":
        fuhead_model = FuHead().cpu()
        dataset = FuHeadDataset(fuhead_model, data_path, split="all", **config["dataset"])
        gaussian_model = FuHeadBindingModel(Struct(**config["model"]), fuhead_model, glctx)
    else:
        raise NotImplementedError(f"Unsupported template_type: {config['template_type']}")
    return dataset, gaussian_model


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    glctx = None
    output_path = os.path.join(args.output_dir, args.subject, args.work_name)
    with open(os.path.join(output_path, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset, gaussian_model = build_dataset_and_model(config, args.subject, glctx)
    gaussian_model.load_ply(os.path.join(output_path, "model.ply"))

    camera = IntrinsicsCamera(
        K=dataset.camera_intri,
        R=dataset.camera_extri[:3, :3],
        T=dataset.camera_extri[:3, 3],
        width=dataset.image_width,
        height=dataset.image_height,
    ).cpu()
    bg_color = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_bg else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cpu",
    )

    render_frames(
        gaussian_model=gaussian_model,
        dataset=dataset,
        camera=camera,
        bg_color=bg_color,
        alpha=args.alpha,
        output_path=output_path,
        batch_size=args.batch_size,
        io_workers=args.io_workers,
        save_interval=args.save_ply_interval,
        save_ply_start=args.save_ply_start,
        save_ply_end=args.save_ply_end,
        save_ply_every=args.save_ply_every,
    )

    if not args.disable_orbit:
        render_orbit(
            gaussian_model=gaussian_model,
            dataset=dataset,
            output_path=output_path,
            bg_color=bg_color,
            anchor_idx=args.orbit_anchor_idx,
            fps=args.orbit_fps,
            duration_sec=args.orbit_duration_sec,
            yaw_deg=args.orbit_yaw_deg,
            make_gif=not args.disable_orbit_gif,
        )

if __name__ == "__main__":
    main()