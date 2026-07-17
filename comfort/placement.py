"""Comfort-budget placement — the live method (paper Algorithm 1) and its baselines.

The contribution is a *frozen, calibrated, rigid* depth offset ``z0`` applied along the gaze
axis, with the unfixated tail left at identity. ``z0`` is a controllable depth-budget knob that
seats the fixated face at a chosen operating point within the *two-sided* comfort budget:
``z0 < 0`` recedes/behind the plane (uncrossed, maximum margin), ``z0 > 0`` is a bounded
pop-out/front (crossed, within ``Z_front``) for impact. Comfort is the constraint envelope; the
operating point is a deliberate choice on the impact<->comfort axis. Calibration observes the
FLAME nose-landmark depth over the first N frames (EMA-smoothed), takes its natural-sway band,
and — for the max-margin (receded) default — seats the band front at the plane *with a shell
margin* so the frontmost splat, not just the nose landmark, clears the plane; ``z0`` is then
frozen. Runtime is a pure ``means.z += z0``.

Three baselines isolate one failure axis each (paper Table ``tab:place``):
``centroid_at_plane`` (fixated face crossed / over budget), ``static_nose_at_plane``
(correct at neutral, drifts), ``uniform_compression`` (comfort by flattening the face).

Everything here is array-agnostic (NumPy for the device-free tests, torch for the pipeline)
and side-effect-free — ``apply`` never mutates its input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _xp, geometry


# --------------------------------------------------------------------------------------- #
# Placement results — each has a pure ``apply(means) -> means'``.
# --------------------------------------------------------------------------------------- #
@dataclass
class RigidPlacement:
    """A frozen rigid depth offset ``z0`` (cm) seating the fixated face at a chosen operating
    point in the two-sided comfort budget (``z0 < 0`` recedes behind the plane for max margin;
    ``z0 > 0`` is a bounded pop-out in front for impact)."""

    z0: float

    def apply(self, means):
        return apply_placement(means, self.z0)


@dataclass
class CompressionPlacement:
    """Uniform depth compression by factor ``k`` about the plane, after a rigid ``shift``.

    ``z' = plane + k * ((z + shift) - plane)``. Reaches comfort by scaling depth, which
    flattens facial relief (the face depth-relief ratio equals ``k``).
    """

    k: float
    plane_z: float
    shift: float

    def apply(self, means):
        out = _xp.clone(means)
        z = out[:, 2]
        out[:, 2] = self.plane_z + self.k * ((z + self.shift) - self.plane_z)
        return out


def apply_placement(means, z0: float):
    """Pure rigid shift along z: ``means.z += z0``. Identical to the B-side one-liner."""
    out = _xp.clone(means)
    out[:, 2] = out[:, 2] + z0
    return out


# --------------------------------------------------------------------------------------- #
# Calibration (paper Algorithm 1, "Calibration" block).
# --------------------------------------------------------------------------------------- #
class CalibrationAccumulator:
    """Accumulate the smoothed nose-anchor depth over calibration frames -> frozen ``z0``."""

    def __init__(
        self,
        geom: dict,
        ema_alpha: float = 0.15,
        band_q=(0.05, 0.95),
        central_x_cm: float = 3.0,
        margin_q: float = 0.999,
        extra_margin_cm: float = 0.0,
        nose_index: int = 30,
    ):
        self.geom = geom
        self.ema_alpha = float(ema_alpha)
        self.band_q = tuple(band_q)
        self.central_x_cm = float(central_x_cm)
        self.margin_q = float(margin_q)
        self.extra_margin_cm = float(extra_margin_cm)
        self.nose_index = int(nose_index)

        self._smoothed = None          # running EMA of the nose depth
        self._nose_hist: list[float] = []  # smoothed nose depths (for the sway band)
        self._shell_margin = 0.0       # max frontmost-splat-beyond-nose over the window
        self.z0 = None                 # frozen offset (set by finalize)

    def update(self, means, landmarks, is_face):
        nose_z = _xp.to_float(landmarks[self.nose_index, 2])
        if self._smoothed is None:
            self._smoothed = nose_z
        else:
            a = self.ema_alpha
            self._smoothed = a * nose_z + (1.0 - a) * self._smoothed
        self._nose_hist.append(self._smoothed)
        self._shell_margin = max(self._shell_margin, self._frontmost_margin(means, is_face, nose_z))
        return self._smoothed

    def _frontmost_margin(self, means, is_face, nose_z: float) -> float:
        """How far the frontmost central *face* splat sits ahead of the nose landmark."""
        x = means[:, 0]
        z = means[:, 2]
        central = is_face & (_xp.abs_(x) < self.central_x_cm)
        if _xp.count_true(central) == 0:
            return 0.0
        front_z = _xp.to_float(_xp.quantile(z[central], self.margin_q))
        return max(0.0, front_z - nose_z)

    def finalize(self) -> float:
        if not self._nose_hist:
            raise RuntimeError("CalibrationAccumulator.finalize() called with no frames")
        hist = np.asarray(self._nose_hist, dtype=np.float64)
        _q_lo, q_hi = np.quantile(hist, self.band_q)  # q_hi = most-forward sway of the nose
        plane_z = self.geom["plane_z"]
        margin = self._shell_margin + self.extra_margin_cm
        # Max-margin (receded) operating point: seat the forward sway (q_hi) plus the shell
        # margin at the plane, so even at forward sway the frontmost splat is at/behind the
        # plane (uncrossed). This is the maximum-margin default within the two-sided budget; a
        # bounded pop-out (front, within Z_front) is the other available operating point.
        self.z0 = float(plane_z - q_hi - margin)
        return self.z0


# --------------------------------------------------------------------------------------- #
# The four placements (each returns a *Placement* with ``.apply``).
# --------------------------------------------------------------------------------------- #
def ours_comfort_zone(frames, geom: dict, **kw) -> RigidPlacement:
    """Live method: calibrate ``z0`` over ``frames`` = iterable of (means, landmarks, is_face)."""
    acc = CalibrationAccumulator(geom, **kw)
    for means, landmarks, is_face in frames:
        acc.update(means, landmarks, is_face)
    return RigidPlacement(acc.finalize())


def centroid_at_plane(means, is_face, geom: dict) -> RigidPlacement:
    """Naive: put the whole-head centroid on the plane. Leaves the fixated face crossed."""
    z0 = geom["plane_z"] - _xp.to_float(_xp.mean(means[:, 2]))
    return RigidPlacement(z0)


def static_nose_at_plane(landmarks, geom: dict, nose_index: int = 30) -> RigidPlacement:
    """Pin the nose landmark to the plane from a single (neutral) frame; no sway band/margin."""
    nose_z = _xp.to_float(landmarks[nose_index, 2])
    return RigidPlacement(geom["plane_z"] - nose_z)


def uniform_compression(means, is_face, geom: dict) -> CompressionPlacement:
    """Shift the face centroid to the plane, then uniformly scale z until comfort is met.

    Finds the largest ``k in (0, 1]`` such that the max angular disparity over *all* Gaussians
    is within ``delta_max``. The face depth-relief ratio then equals ``k`` (< 1 => flattened).
    """
    plane_z = geom["plane_z"]
    ipd, D, delta_max = geom["ipd"], geom["D"], geom["delta_max"]
    z = means[:, 2]
    face_centroid = _xp.to_float(_xp.mean(z[is_face]))
    shift = plane_z - face_centroid
    zs = _xp.to_numpy(z) + shift  # numpy for the scalar k-search

    def max_disp(k: float) -> float:
        d = geometry.angular_disparity(plane_z + k * (zs - plane_z), ipd, D)
        return float(np.max(d))

    if max_disp(1.0) <= delta_max:
        k = 1.0
    else:
        lo, hi = 0.0, 1.0  # max_disp monotonically increasing in k
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if max_disp(mid) <= delta_max:
                lo = mid
            else:
                hi = mid
        k = lo
    return CompressionPlacement(k=float(k), plane_z=plane_z, shift=float(shift))
