"""Device-free evaluation sweeps that produce the paper's tables from a ``.pt`` sequence.

Everything here operates on a *sequence* of schema-v2 scenes (dicts of arrays), computes a
*frozen* placement, and characterises it over the sequence and a scripted observer/yaw sweep.
It is pure and array-agnostic, so it runs on the synthetic fixture (device-free CI) and on a
real exported sequence identically.

Tables produced:
  * ``placement_table``   -> main ``tab:place`` (4 placements x {worst face disparity, min relief})
  * ``containment_table`` -> supplemental ``tab:s2`` (containment vs calibration length N & margin)
  * ``observer_yaw_grid`` -> supplemental ``tab:s3`` (observer-angle x avatar-yaw disparity grid)
"""

from __future__ import annotations

import numpy as np

import metrics  # top-level module (code/avatar-python/metrics.py)

from . import _xp, fixtures, placement

PLACEMENT_ORDER = (
    "centroid_at_plane",
    "static_nose_at_plane",
    "uniform_compression",
    "ours_comfort_zone",
)


def _worst_disp(placed_means, is_face, observers, geom) -> float:
    d = metrics.max_face_disparity_batched(placed_means, is_face, observers, geom)
    return float(np.max(_xp.to_numpy(d)))


def calibrate_ours(frames, geom, N: int, extra_margin_cm: float = 0.0) -> placement.RigidPlacement:
    """Freeze ``z0`` from the first ``N`` frames (nose-sway band + shell margin + extra margin)."""
    acc = placement.CalibrationAccumulator(geom, extra_margin_cm=extra_margin_cm)
    for s in frames[:N]:
        acc.update(s["means3D"], s["landmarks3D"], s["is_face"])
    return placement.RigidPlacement(acc.finalize())


def frozen_placements(frames, geom, N: int, margin_cm: float = 0.5) -> dict:
    """Build the four *frozen* placements (calibrated on frame 0 / the first N frames).

    ``margin_cm`` is the paper's placement safety margin for ours ("seat the band at the zone
    front, with margin"); it absorbs the sway peak the smoothed band under-covers.
    """
    f0 = frames[0]
    return {
        "centroid_at_plane": placement.centroid_at_plane(f0["means3D"], f0["is_face"], geom),
        "static_nose_at_plane": placement.static_nose_at_plane(f0["landmarks3D"], geom),
        "uniform_compression": placement.uniform_compression(f0["means3D"], f0["is_face"], geom),
        "ours_comfort_zone": calibrate_ours(frames, geom, N, extra_margin_cm=margin_cm),
    }


def eval_frozen(frames, place_obj, geom, observers) -> dict:
    """Characterise a frozen placement over a sequence.

    Returns the worst-case (over frames x observers) fixated-face disparity, the min
    depth-relief ratio, whether the face ever crosses in front of the plane (``worst_front_z``
    = the most-forward placed face depth; ``> 0`` means crossed content), and a comfort flag.
    """
    worst_disp, min_relief, worst_front_z = 0.0, float("inf"), -float("inf")
    for s in frames:
        placed = place_obj.apply(s["means3D"])
        worst_disp = max(worst_disp, _worst_disp(placed, s["is_face"], observers, geom))
        min_relief = min(min_relief, metrics.depth_relief_ratio(s["means3D"], placed, s["is_face"]))
        face_z = _xp.to_numpy(placed[:, 2])[_xp.to_numpy(s["is_face"]).astype(bool)]
        worst_front_z = max(worst_front_z, float(face_z.max() - geom["plane_z"]))
    return {
        "worst_disp": worst_disp,
        "min_relief": min_relief,
        "worst_front_z": worst_front_z,          # > 0 => face crosses in front (uncomfortable)
        "comfortable": worst_disp <= geom["delta_max"],
        "uncrossed": worst_front_z <= 1e-3,
    }


def placement_table(frames, geom, observers, N: int, margin_cm: float = 0.5) -> dict:
    """main ``tab:place``: {placement -> {worst_disp, min_relief, worst_front_z, ...}}."""
    places = frozen_placements(frames, geom, N, margin_cm=margin_cm)
    return {name: eval_frozen(frames, places[name], geom, observers) for name in PLACEMENT_ORDER}


def containment_table(frames, geom, Ns, margins, observers) -> list[dict]:
    """supplemental ``tab:s2``: held-out containment vs calibration length N and margin."""
    rows = []
    for N in Ns:
        for margin in margins:
            place = calibrate_ours(frames, geom, N, extra_margin_cm=margin)
            in_zone = sum(
                _worst_disp(place.apply(s["means3D"]), s["is_face"], observers, geom) <= geom["delta_max"]
                for s in frames
            )
            rows.append({"N": N, "margin": margin, "containment": in_zone / len(frames)})
    return rows


def observer_yaw_grid(frames, geom, observer_angles, yaw_angles, N: int) -> dict:
    """supplemental ``tab:s3``: worst face disparity per (observer-angle, avatar-yaw) cell."""
    place = calibrate_ours(frames, geom, N)
    grid = {}
    for yaw in yaw_angles:
        for obs in observer_angles:
            worst = 0.0
            for s in frames:
                s_yaw = fixtures.make_yaw_scene(s, yaw_deg=yaw)
                placed = place.apply(s_yaw["means3D"])
                worst = max(worst, _worst_disp(placed, s_yaw["is_face"], [(obs, geom["D"])], geom))
            grid[(obs, yaw)] = worst
    return grid
