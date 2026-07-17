#!/usr/bin/env python3
"""Device-free placement evaluation CLI (thin wrapper over the comfort library).

Loads a schema-v2 ``.pt`` sequence (or the synthetic fixture), runs all four placements, and
reports the objective comfort table exactly as the paper's Table tab:place is defined
(worst-case fixated-face disparity + depth-relief preservation over the sequence). Optionally
exports the "money figure" PLYs (original vs ours, with a display-plane grid).

This replaces the earlier single-frame, 2-placement, self-comparing script:
  * all FOUR placements (centroid / static-nose / uniform / ours), not just original vs ours;
  * disparity restricted to the fixated FACE mask (not all Gaussians);
  * depth-relief preservation computed against the ORIGINAL (no dummy self-compare);
  * the rigid offset has the correct sign (``means.z += z0``, z0<0 => head recedes behind);
  * the retired ``volume_margin`` / ``final_anchor_z`` keys are gone (calibration-window z0).
"""

from __future__ import annotations

import argparse
import glob
import os
import struct

import numpy as np

from comfort import evalsweep, fixtures, geometry, placement


def _load_sequence(seq_dir: str | None, n_frames: int, sway: float) -> list[dict]:
    if not seq_dir:
        return fixtures.make_synthetic_sequence(n_frames=n_frames, sway_cm=sway)
    import torch
    files = sorted(glob.glob(os.path.join(seq_dir, "frame_*.pt")))
    if not files:
        raise FileNotFoundError(f"no frame_*.pt under {seq_dir}")
    out = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        out.append({k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in d.items()})
    return out


def _save_ply(means_cm, scales_cm, rotations, opacities, shs, path: str):
    """Minimal binary PLY writer (coords already in cm; scales stored as log(cm))."""
    xyz = np.asarray(means_cm, np.float32)
    n = xyz.shape[0]
    sh = np.asarray(shs, np.float32).reshape(n, -1)
    if sh.shape[1] < 48:
        sh = np.concatenate([sh, np.zeros((n, 48 - sh.shape[1]), np.float32)], axis=1)
    scaling = np.log(np.asarray(scales_cm, np.float32) + 1e-8)
    opac = np.asarray(opacities, np.float32).reshape(n, -1)
    rot = np.asarray(rotations, np.float32)

    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property float nx\nproperty float ny\nproperty float nz\n"
    header += "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    header += "".join(f"property float f_rest_{i}\n" for i in range(45))
    header += "property float opacity\nproperty float scale_0\nproperty float scale_1\n"
    header += "property float scale_2\nproperty float rot_0\nproperty float rot_1\n"
    header += "property float rot_2\nproperty float rot_3\nend_header\n"
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for i in range(n):
            row = [xyz[i, 0], xyz[i, 1], xyz[i, 2], 0.0, 0.0, 0.0] + sh[i].tolist() + [
                opac[i, 0], scaling[i, 0], scaling[i, 1], scaling[i, 2],
                rot[i, 0], rot[i, 1], rot[i, 2], rot[i, 3],
            ]
            fh.write(struct.pack("<62f", *row))
    print(f"  [PLY] {path} ({n} pts)")


def _plane_grid(shs_ref, size_cm=15.0, n=60):
    grid = np.linspace(-size_cm, size_cm, n)
    gx, gy = np.meshgrid(grid, grid)
    means = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1).astype(np.float32)
    k = np.asarray(shs_ref).reshape(np.asarray(shs_ref).shape[0], -1).shape[1]
    scales = np.full((means.shape[0], 3), 0.05, np.float32)
    rot = np.tile(np.array([1, 0, 0, 0], np.float32), (means.shape[0], 1))
    opac = np.full((means.shape[0], 1), 1.0, np.float32)
    shs = np.zeros((means.shape[0], max(k // 3, 1), 3), np.float32)
    shs[:, 0, 1] = 1.772  # greenish plane
    return means, scales, rot, opac, shs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", default=None, help="Directory of frame_*.pt (real data). Omit for the fixture.")
    ap.add_argument("--n-frames", type=int, default=24, help="Synthetic sequence length.")
    ap.add_argument("--sway", type=float, default=1.2, help="Synthetic head-sway amplitude (cm).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ply", default=None, help="Output dir for original/ours money-figure PLYs.")
    args = ap.parse_args(argv)

    g = geometry.load_geometry(args.config)
    frames = _load_sequence(args.seq, args.n_frames, args.sway)
    observers = [(a, g["D"]) for a in (-25.0, -15.0, 0.0, 15.0, 25.0)]

    print("=" * 68)
    print(f"  Comfort-zone placement evaluation ({len(frames)} frames)")
    print(f"  D={g['D']:.3g} cm, IPD={g['ipd']:.2g} cm, delta_max={g['delta_max']:.3g} arcmin, "
          f"Z_behind={g['Z_behind']:.2f} cm")
    print("=" * 68)
    table = evalsweep.placement_table(frames, g, observers, N=len(frames))
    print(f"  {'placement':22s} {'disp(arcmin)':>12s} {'relief':>7s} {'front_z':>8s}  status")
    for key in evalsweep.PLACEMENT_ORDER:
        r = table[key]
        status = ("comfort" if r["comfortable"] else "OVER") + "/" + ("uncrossed" if r["uncrossed"] else "CROSSED")
        print(f"  {key:22s} {r['worst_disp']:12.2f} {r['min_relief']:7.2f} {r['worst_front_z']:+8.2f}  {status}")
    print("=" * 68)

    if args.ply:
        os.makedirs(args.ply, exist_ok=True)
        f0 = frames[0]
        z0 = evalsweep.calibrate_ours(frames, g, N=len(frames), extra_margin_cm=0.5).z0
        ours = placement.apply_placement(f0["means3D"], z0)
        shs0 = np.asarray(f0["shs"]).reshape(np.asarray(f0["shs"]).shape[0], -1, 3)
        _save_ply(f0["means3D"], f0["scales"], f0["rotations"], f0["opacities"], f0["shs"],
                  os.path.join(args.ply, "01_original.ply"))
        pm, ps, pr, po, psh = _plane_grid(f0["shs"])
        _save_ply(np.concatenate([ours, pm]),
                  np.concatenate([np.asarray(f0["scales"]), ps]),
                  np.concatenate([np.asarray(f0["rotations"]), pr]),
                  np.concatenate([np.asarray(f0["opacities"]).reshape(-1, 1), po]),
                  np.concatenate([shs0, psh]),
                  os.path.join(args.ply, "02_ours_with_plane.ply"))
        print(f"  frozen z0 = {z0:.3f} cm")


if __name__ == "__main__":
    main()
