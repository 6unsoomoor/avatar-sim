"""The ``.pt`` schema (v2) — the single contract for A export <-> B load <-> harness.

Each per-frame ``.pt`` is a dict of the deformed avatar plus the placement anchor and the
static fixation mask, all in **centimetres**:

    means3D    [N, 3]  f32   deformed centres (cm)
    scales     [N, 3]  f32   activated/linear scales (cm)  -- NOT log
    rotations  [N, 4]  f32   quaternion (wxyz)
    opacities  [N, 1]  f32   activated (sigmoid) in [0, 1]
    shs        [N, K, 3] f32 (or ``colors`` [N, 3]) -- one mode per sequence
    landmarks3D[L, 3]  f32   >= the nose tip (anchor), cm
    is_face    [N]     bool  static FLAME fixation mask
    units      "cm"    metadata
    nose_index int     metadata (iBUG-68 nose tip = 30)

There is deliberately **no** placement offset in the ``.pt``: the frozen ``z0`` is calibrated
from the exported sequence by :mod:`comfort.placement` and written to a sidecar.
"""

from __future__ import annotations

import numpy as np

from . import _xp

REQUIRED_KEYS = ("means3D", "scales", "rotations", "opacities", "landmarks3D", "is_face")


def _shape(x):
    return tuple(int(s) for s in x.shape)


def validate_pt(d: dict, strict: bool = True, require_color: bool = True) -> list[str]:
    """Return a list of schema problems (empty == valid). Raise ``ValueError`` if ``strict``.

    Works on dicts whose arrays are NumPy or torch. A v1 dict (missing ``is_face`` / anchor /
    metadata, or log-space scales) is reported as not placement-ready.
    """
    problems: list[str] = []

    def has(k):
        return k in d and d[k] is not None

    for k in REQUIRED_KEYS:
        if not has(k):
            problems.append(f"missing required key '{k}'")
    if require_color and not (has("shs") or has("colors")):
        problems.append("missing colour: need 'shs' [N,K,3] or 'colors' [N,3]")

    N = None
    if has("means3D"):
        ms = _shape(d["means3D"])
        if len(ms) != 2 or ms[1] != 3:
            problems.append(f"means3D must be [N,3], got {ms}")
        else:
            N = ms[0]

    if N is not None:
        if has("scales"):
            ss = _shape(d["scales"])
            if ss != (N, 3):
                problems.append(f"scales must be [{N},3], got {ss}")
            else:
                smin = float(_xp.to_numpy(d["scales"]).min())
                if smin < 0.0:
                    problems.append(
                        f"scales look log-space (min {smin:.3g} < 0); export activated/linear cm"
                    )
        if has("rotations") and _shape(d["rotations"]) != (N, 4):
            problems.append(f"rotations must be [{N},4], got {_shape(d['rotations'])}")
        if has("opacities") and _shape(d["opacities"])[0] != N:
            problems.append(f"opacities first dim must be {N}, got {_shape(d['opacities'])}")
        if has("is_face"):
            fs = _shape(d["is_face"])
            if fs[0] != N:
                problems.append(f"is_face must have {N} entries, got {fs}")
        for ck in ("shs", "colors"):
            if has(ck) and _shape(d[ck])[0] != N:
                problems.append(f"{ck} first dim must be {N}, got {_shape(d[ck])}")

    if has("landmarks3D"):
        ls = _shape(d["landmarks3D"])
        if len(ls) != 2 or ls[1] != 3:
            problems.append(f"landmarks3D must be [L,3], got {ls}")
        else:
            arr = _xp.to_numpy(d["landmarks3D"])
            if not np.isfinite(arr).all() or np.allclose(arr, 0.0):
                problems.append("landmarks3D all-zero/non-finite (missing nose anchor)")

    if d.get("units") != "cm":
        problems.append(f"units must be 'cm', got {d.get('units')!r}")
    if "nose_index" not in d:
        problems.append("missing 'nose_index' metadata")

    if strict and problems:
        raise ValueError("invalid .pt (schema v2):\n  - " + "\n  - ".join(problems))
    return problems
