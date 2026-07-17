"""Synthetic scene fixtures for device-free testing.

These let every placement operator, metric, and calibration step be unit-tested without the
~600 MB data blobs or the FLAME model. The default scene is a hand-built miniature head:
a small *face* cluster in front of the plane (with a splat slightly ahead of the nose
landmark, so the shell margin is non-trivial) and an unfixated *tail* receding behind it.
All coordinates are centimetres, in the display-plane frame (z>0 = toward viewer).
"""

from __future__ import annotations

import math

import numpy as np

NOSE_INDEX = 30  # iBUG-68 nose tip
_N_LANDMARKS = 68


def make_synthetic_scene() -> dict:
    """Return a deterministic schema-v2 scene as a dict of NumPy arrays.

    Layout (cm, z>0 in front of the plane):
      * face (is_face=True): a splat at z=8.5 *ahead* of the nose landmark (z=8.0), then the
        face receding to z=3.0 -> exercises the frontmost-splat shell margin.
      * tail (is_face=False): z from 1.0 down to -18.0.
    """
    # Fixated near-face: shallow (~3 cm), so it fits the comfort zone with room for sway.
    # The front splat sits 0.5 cm AHEAD of the nose landmark -> non-trivial shell margin.
    face = np.array(
        [
            # x,    y,    z
            [0.0, 0.5, 8.5],   # splat just IN FRONT of the nose landmark (drives shell margin)
            [0.0, 0.0, 8.0],   # coincides in depth with the nose landmark
            [1.0, 1.0, 7.5],
            [-1.0, 1.0, 7.0],
            [1.5, -1.0, 6.5],
            [-1.5, -1.0, 6.0],
            [0.5, 2.0, 5.5],
        ],
        dtype=np.float32,
    )
    # Unfixated tail: deep (recedes to -18 cm), left at identity.
    tail = np.array(
        [
            [0.0, 3.0, 5.0],
            [2.0, 2.0, 2.0],
            [-2.0, 4.0, -2.0],
            [3.0, -2.0, -6.0],
            [-3.0, 5.0, -12.0],
            [2.0, 6.0, -18.0],
        ],
        dtype=np.float32,
    )
    means = np.concatenate([face, tail], axis=0).astype(np.float32)
    n = means.shape[0]
    n_face = face.shape[0]

    is_face = np.zeros(n, dtype=bool)
    is_face[:n_face] = True

    # 68 landmarks; only the nose group is populated with meaningful depths so that the
    # nose tip (idx 30) is the anterior-most point of the nose group {27..35}.
    landmarks = np.zeros((_N_LANDMARKS, 3), dtype=np.float32)
    nose_group_z = {27: 5.0, 28: 5.5, 29: 6.0, 30: 8.0, 31: 7.0, 32: 7.2, 33: 7.4, 34: 7.6, 35: 7.8}
    for idx, z in nose_group_z.items():
        landmarks[idx] = (0.0, 0.0, z)

    scales = np.full((n, 3), 0.05, dtype=np.float32)          # linear cm (already activated)
    rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))  # identity wxyz
    opacities = np.full((n, 1), 0.8, dtype=np.float32)
    shs = np.full((n, 1, 3), 0.5, dtype=np.float32)

    return {
        "means3D": means,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "shs": shs,
        "landmarks3D": landmarks,
        "is_face": is_face,
        "units": "cm",
        "nose_index": NOSE_INDEX,
    }


def make_yaw_scene(base: dict | None = None, yaw_deg: float = 0.0) -> dict:
    """Rotate a scene's points about the vertical (y) axis by ``yaw_deg`` (for tab:s1 tests).

    Rotating the face's width into the gaze axis increases its depth extent, which is what
    ``geometry.face_extent_at_yaw`` / ``yaw_safe`` characterise.
    """
    scene = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in (base or make_synthetic_scene()).items()}
    a = np.radians(yaw_deg)
    ca, sa = np.cos(a), np.sin(a)
    R = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float32)  # about y
    scene["means3D"] = scene["means3D"] @ R.T
    scene["landmarks3D"] = scene["landmarks3D"] @ R.T
    return scene


def make_synthetic_sequence(n_frames: int = 24, sway_cm: float = 1.2) -> list[dict]:
    """A deterministic sequence with a rigid head-bob (nose sway) about the neutral pose.

    Drives the calibration accumulator (a natural-sway band) and the worst-case-over-a-sequence
    tables. Frame 0 is the neutral pose (sway 0). No RNG — sway is a sine over the frame index.
    """
    base = make_synthetic_scene()
    lm_populated = np.linalg.norm(base["landmarks3D"], axis=1) > 1e-6
    frames = []
    for t in range(n_frames):
        dz = sway_cm * math.sin(2.0 * math.pi * t / n_frames)
        s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        s["means3D"] = base["means3D"].copy()
        s["means3D"][:, 2] += dz
        lm = base["landmarks3D"].copy()
        lm[lm_populated, 2] += dz  # shift only populated landmarks (keep zeros zero)
        s["landmarks3D"] = lm
        frames.append(s)
    return frames


def save_scene_pt(scene: dict, path: str) -> None:
    """Serialise a scene as a real torch ``.pt`` (requires torch; used off the test path)."""
    import torch  # deferred: only the real pipeline needs torch

    out = {}
    for k, v in scene.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)
        else:
            out[k] = v
    torch.save(out, path)
