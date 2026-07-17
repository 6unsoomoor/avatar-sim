"""Stereo comfort-budget geometry — the closed forms behind the paper's derived numbers.

All depths are along the gaze axis, signed relative to the display plane: ``z > 0`` is in
*front* of the plane (toward the viewer, crossed disparity), ``z < 0`` is *behind* it
(uncrossed). Lengths are centimetres. This module is the single source of truth for
``supplemental`` Eq. (angular disparity) and the derived *two-sided* comfort bounds
``Z_front`` / ``Z_behind`` (Table ``tab:disp``) that delimit the budget the placement knob
seats the fixated face within (front = bounded pop-out, behind = max margin).
"""

from __future__ import annotations

import math
import os
from typing import Tuple

from . import _xp

# radians -> arc-minutes
RAD_TO_ARCMIN = (180.0 / math.pi) * 60.0
_RAD_TO_ARCMIN = RAD_TO_ARCMIN  # backward-compatible alias

# Default viewing geometry (overridden by experiments/configs/eval_sweep.yaml).
DEFAULT_IPD_CM = 6.3
DEFAULT_VIEW_DISTANCE_CM = 60.0
DEFAULT_DELTA_MAX_ARCMIN = 35.0
DEFAULT_PLANE_Z_CM = 0.0


def angular_disparity(z, ipd: float = DEFAULT_IPD_CM, D: float = DEFAULT_VIEW_DISTANCE_CM):
    r"""Perceptually relevant angular disparity :math:`\delta(z)` in arc-minutes.

    .. math::
        \delta(z) = 2\,\bigl|\arctan\tfrac{e}{2(D-z)} - \arctan\tfrac{e}{2D}\bigr|

    with IPD ``e`` and eye-to-plane distance ``D``. Monotonically increasing in ``|z|``.
    Accepts a Python scalar, NumPy array, or torch tensor (element-wise); returns the same
    kind. This is the exact form used for the paper's ``tab:disp`` and for the comfort
    criterion ``delta(z) <= delta_max``.
    """
    e = ipd
    plane_term = math.atan(e / (2.0 * D))  # scalar constant
    point_term = _xp.arctan(e / (2.0 * (D - z)))
    return 2.0 * _xp.abs_(point_term - plane_term) * _RAD_TO_ARCMIN


def comfort_bounds(
    ipd: float = DEFAULT_IPD_CM,
    D: float = DEFAULT_VIEW_DISTANCE_CM,
    delta_max_arcmin: float = DEFAULT_DELTA_MAX_ARCMIN,
) -> Tuple[float, float]:
    """Solve ``angular_disparity(z) == delta_max`` for both signs.

    Returns ``(Z_front, Z_behind)`` as positive magnitudes (cm). The zone is asymmetric:
    for the default constants (IPD 6.3, D 60, delta_max 35) this reproduces the paper's
    ``Z_front ~= 5.3`` and ``Z_behind ~= 6.46``.
    """

    def f(z: float) -> float:
        return float(angular_disparity(z, ipd, D)) - delta_max_arcmin

    z_front = _bisect(f, lo=1e-6, hi=D - 1e-3)           # z > 0 (in front)
    z_behind = _bisect(f, lo=-(D * 3.0), hi=-1e-6)        # z < 0 (behind)
    return z_front, abs(z_behind)


def _bisect(f, lo: float, hi: float, tol: float = 1e-9, max_iter: int = 200) -> float:
    """Robust bisection root-find (stdlib only; no scipy dependency required)."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise ValueError(
            f"angular_disparity does not bracket delta_max on [{lo}, {hi}] "
            f"(f(lo)={flo:.3f}, f(hi)={fhi:.3f}); check the constants."
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) < tol:
            return mid
        if flo * fmid < 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------- #
# Configuration loading — constants live only in experiments/configs/eval_sweep.yaml.
# --------------------------------------------------------------------------------------- #
def face_extent_at_yaw(face_means, yaw_deg: float) -> float:
    """Gaze-axis depth extent (cm) of the face region after rotating it by ``yaw_deg``.

    Yaw rotates the face's horizontal width into the gaze axis, increasing its depth extent.
    ``face_means`` is the face-region point cloud ``[Nf, 3]`` (already masked to the face).
    """
    import numpy as np

    P = _xp.to_numpy(face_means)
    a = math.radians(yaw_deg)
    ca, sa = math.cos(a), math.sin(a)
    R = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float64)  # about y
    z = (P @ R.T)[:, 2]
    return float(z.max() - z.min())


def yaw_safe(face_means, Z_behind: float, yaw_max: float = 60.0, step: float = 0.5) -> float:
    """Largest ``|yaw|`` (deg) for which the face depth extent still fits the behind-zone.

    ``yaw_safe = max{ |theta| : face_extent_at_yaw(theta) <= Z_behind }``. Scans from 0 upward
    and stops at the first yaw that exceeds the budget (extent grows monotonically with yaw).
    """
    import numpy as np

    P = _xp.to_numpy(face_means)
    safe = 0.0
    n = int(round(yaw_max / step))
    for k in range(n + 1):
        y = k * step
        if face_extent_at_yaw(P, y) <= Z_behind:
            safe = y
        else:
            break
    return safe


def _find_config() -> str:
    """Locate experiments/configs/eval_sweep.yaml by walking up from this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        cand = os.path.join(d, "experiments", "configs", "eval_sweep.yaml")
        if os.path.isfile(cand):
            return cand
        d = os.path.dirname(d)
    raise FileNotFoundError(
        "Could not locate experiments/configs/eval_sweep.yaml above " + here
    )


def load_geometry(path: str | None = None) -> dict:
    """Read the ``geometry`` block from eval_sweep.yaml and derive the comfort bounds.

    Returns a dict with keys ``ipd``, ``D``, ``delta_max``, ``plane_z``, ``Z_front``,
    ``Z_behind`` (all cm / arc-min). Falls back to the module defaults if PyYAML or the
    file is unavailable.
    """
    ipd, D = DEFAULT_IPD_CM, DEFAULT_VIEW_DISTANCE_CM
    delta_max, plane_z = DEFAULT_DELTA_MAX_ARCMIN, DEFAULT_PLANE_Z_CM
    try:
        import yaml  # optional

        cfg_path = path or _find_config()
        with open(cfg_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        g = cfg.get("geometry", {})
        ipd = float(g.get("ipd_cm", ipd))
        D = float(g.get("view_distance_cm", D))
        delta_max = float(g.get("delta_max_arcmin", delta_max))
        plane_z = float(g.get("plane_z_cm", plane_z))
    except Exception:
        # Device-free default; the tests pin these values explicitly anyway.
        pass

    z_front, z_behind = comfort_bounds(ipd, D, delta_max)
    return {
        "ipd": ipd,
        "D": D,
        "delta_max": delta_max,
        "plane_z": plane_z,
        "Z_front": z_front,
        "Z_behind": z_behind,
    }
