"""FLAME landmark handling for the comfort-budget placement anchor.

The placement anchor (the FLAME nose landmark on which the frozen, calibrated rigid ``z0`` is
seated within the two-sided comfort budget) is the FLAME **nose tip**, iBUG-68 index **30**
(not 35, which is a
nostril corner). This module resolves and *validates* that anchor, and wraps the FLAME
keypoint computation so a missing landmark embedding fails loudly instead of silently
exporting zero landmarks (the previous behaviour, which quietly set the anchor depth to 0).
"""

from __future__ import annotations

import numpy as np

from . import _xp

NOSE_INDEX = 30
NOSE_GROUP = tuple(range(27, 36))  # iBUG-68 nose group {27..35}


def resolve_nose_landmark(landmarks3D, nose_index: int = NOSE_INDEX, validate: bool = True):
    """Return the nose-tip landmark (a length-3 array), validating it is the anchor we expect.

    Validation: among the *populated* nose-group landmarks, index ``nose_index`` must be the
    anterior-most (largest gaze-axis depth). This catches a wrong index or a reordered
    embedding rather than trusting "index N = nose" blindly.
    """
    lm = _xp.to_numpy(landmarks3D)
    if lm.ndim != 2 or lm.shape[1] != 3:
        raise ValueError(f"landmarks3D must be [L,3], got {lm.shape}")
    if nose_index >= lm.shape[0]:
        raise IndexError(f"nose_index {nose_index} out of range for {lm.shape[0]} landmarks")

    if validate:
        populated = [i for i in NOSE_GROUP if i < lm.shape[0] and np.linalg.norm(lm[i]) > 1e-6]
        if populated:
            z_max = max(float(lm[i, 2]) for i in populated)
            if float(lm[nose_index, 2]) < z_max - 1e-3:
                raise ValueError(
                    f"nose_index {nose_index} is not the anterior-most nose-group landmark "
                    f"(z={float(lm[nose_index,2]):.3f} < {z_max:.3f}); wrong index or a "
                    f"reordered landmark embedding."
                )
    return lm[nose_index]


def nose_depth(landmarks3D, nose_index: int = NOSE_INDEX, validate: bool = False) -> float:
    """Gaze-axis depth (z) of the nose-tip landmark."""
    return float(resolve_nose_landmark(landmarks3D, nose_index, validate=validate)[2])


def compute_landmarks(flame_model, mesh):
    """Compute FLAME keypoints, failing loudly if the embedding is unavailable.

    The active ``submodules/flame/flame.py`` exposes ``compute_keypoints(vertices)`` only when
    the landmark-embedding buffers are loaded. If they are not, this raises rather than
    returning zeros (which would silently place the anchor at depth 0).
    """
    if not hasattr(flame_model, "compute_keypoints"):
        raise AttributeError(
            "FLAME model has no compute_keypoints(); enable the landmark-embedding buffers "
            "(landmark_embedding.npz) — see docs/method-pivot.md."
        )
    lms = flame_model.compute_keypoints(mesh)
    arr = _xp.to_numpy(lms)
    if arr.size == 0 or not np.isfinite(arr).all() or np.allclose(arr, 0.0):
        raise RuntimeError(
            "compute_keypoints() returned empty/zero landmarks — the landmark embedding is "
            "likely missing. Refusing to export a zero anchor."
        )
    return lms
