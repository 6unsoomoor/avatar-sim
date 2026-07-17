"""Objective comfort/preservation metrics for comfort-budget placement.

These measure the *disparity budget* attained along the gaze axis (we do not measure
subjective comfort — that is deferred to a sustained-viewing study). Two objective quantities
per the paper:

* **Max face disparity** — the maximum angular disparity (arc-min) over the fixated *face*
  region; the budget/comfort criterion is ``|delta| <= delta_max`` on either side of the plane
  (front pop-out or behind). Restricting to the face mask is what the
  paper reports (``max_{i: m_i=1} delta(z_i)``), so the disparity functions take an
  ``is_face`` mask.
* **Depth-relief preservation ratio** — ``range(placed_face_z) / range(orig_face_z)`` over the
  face region (1.0 = unchanged; ``k`` for a uniform ``k``-compression). This replaces the
  earlier "Pearson correlation" wording, which is scale-invariant and cannot separate a
  uniform flattening (see paper/supplemental).

Array-agnostic: works on NumPy (device-free tests) or torch tensors (the pipeline; the batched
observer sweep stays on the input's device, so it runs on the GPU when given CUDA tensors).
"""

from __future__ import annotations

import math

import numpy as np

from comfort import _xp, geometry


# --------------------------------------------------------------------------------------- #
# Canonical API
# --------------------------------------------------------------------------------------- #
def max_face_disparity(means, is_face, geom: dict) -> float:
    """Max angular disparity (arc-min) over the fixated face for the on-axis primary viewer."""
    return float(max_face_disparity_batched(means, is_face, [(0.0, geom["D"])], geom)[0])


def max_face_disparity_batched(means, is_face, observers, geom: dict):
    """Max face disparity for each observer in ``observers`` = list of (angle_deg, D_cm).

    Uses the exact vergence-angle definition ``delta = |verg(P) - verg(fixation)|`` from the two
    real eye positions of an observer at azimuth ``a`` (about vertical) and distance ``D`` who
    fixates the on-plane anchor. This is fully general: it handles off-axis observers correctly
    (horizontal parallax slightly *raises* disparity, as in the paper's Table S3), and for a
    point on the gaze axis it equals the closed form :func:`comfort.geometry.angular_disparity`
    exactly (so the derived Table ``tab:disp`` is reproduced). Returns an array of shape
    ``[len(observers)]`` on the same backend/device as ``means`` (runs on the GPU for CUDA
    tensors).
    """
    P = means[is_face]                                     # [Nf, 3]
    E_L, E_R = _observer_eyes(observers, geom, ref=means)  # [M, 3] each
    verg = _vergence(P[None, :, :], E_L[:, None, :], E_R[:, None, :])          # [M, Nf]
    fix = _xp.asarray_like([0.0, 0.0, geom["plane_z"]], means)                 # [3]
    verg0 = _vergence(fix[None, :], E_L, E_R)                                  # [M]
    delta = _xp.abs_(verg - verg0[:, None]) * geometry.RAD_TO_ARCMIN           # [M, Nf]
    return _xp.max_axis(delta, axis=1)                                         # [M]


def _observer_eyes(observers, geom: dict, ref):
    """Eye positions for observers fixating the plane anchor, on the same backend as ``ref``.

    Observer ``k`` sits at azimuth ``angle_k`` (deg, about vertical) and distance ``D_k``, so
    the binocular midpoint is ``M = (D sin a, 0, D cos a)`` and the eyes are offset by IPD/2
    along the horizontal direction perpendicular to the gaze.
    """
    e = geom["ipd"]
    left, right = [], []
    for angle_deg, D in observers:
        a = math.radians(angle_deg)
        M = np.array([D * math.sin(a), 0.0, D * math.cos(a)])
        r = np.array([math.cos(a), 0.0, -math.sin(a)])     # horizontal, perp to gaze
        left.append(M - 0.5 * e * r)
        right.append(M + 0.5 * e * r)
    return _xp.asarray_like(np.asarray(left), ref), _xp.asarray_like(np.asarray(right), ref)


def _vergence(P, E_L, E_R):
    """Angle (rad) subtended at ``P`` by the two eyes; broadcasts over leading dims."""
    a = P - E_L
    b = P - E_R
    dot = (a * b).sum(-1)
    na = _xp.sqrt((a * a).sum(-1))
    nb = _xp.sqrt((b * b).sum(-1))
    cos = _xp.clip(dot / (na * nb), -1.0, 1.0)
    return _xp.arccos(cos)


def depth_relief_ratio(orig_means, placed_means, is_face) -> float:
    """Face depth-relief preservation ratio in [0, ~1] (1.0 = relief unchanged)."""
    of = (orig_means[:, 2])[is_face]
    pf = (placed_means[:, 2])[is_face]
    orig_range = _xp.to_float(of.max() - of.min())
    placed_range = _xp.to_float(pf.max() - pf.min())
    if orig_range < 1e-9:
        return 1.0
    return placed_range / orig_range


# --------------------------------------------------------------------------------------- #
# Backward-compatible shims (used by the pre-refactor simulation script until Phase 3).
# Reimplemented on the exact closed form; prefer the canonical API above.
# --------------------------------------------------------------------------------------- #
def calculate_max_angular_disparity(means3D, eye_L, eye_R, screen_z: float = 0.0) -> float:
    """Deprecated: max disparity over *all* points. Prefer :func:`max_face_disparity`."""
    D = _xp.to_float(eye_L[2])
    ipd = _xp.to_float(_xp.abs_(eye_R[0] - eye_L[0]))
    depth = means3D[:, 2] - screen_z
    d = geometry.angular_disparity(depth, ipd, D)
    return _xp.to_float(d.max())


def calculate_facial_preservation(original_means3D, budgeted_means3D, is_face_mask) -> float:
    """Deprecated: returns the depth-relief ratio as a *percent*. Prefer
    :func:`depth_relief_ratio` (unit ratio)."""
    if _xp.count_true(is_face_mask) == 0:
        return 0.0
    return 100.0 * depth_relief_ratio(original_means3D, budgeted_means3D, is_face_mask)
