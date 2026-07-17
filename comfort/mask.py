"""Static per-Gaussian fixation (face) mask.

The fixation region is a model-space *definition*, not an image-space estimate: the FLAME
facial region is all mesh vertices except the neck and head-boundary lists
(``face = ALL \\ (neck u boundary)``). Each Gaussian is bound to a mesh triangle
(``binding_face_id``), so lifting the per-vertex indicator through that binding yields a
static per-Gaussian label ``is_face[i]`` (topology is fixed => computed once, shipped in the
``.pt``). The mask is used only to decide *what is characterised/anchored*, never to warp
anything.
"""

from __future__ import annotations

import numpy as np

from . import _xp


def load_flame_mask_indices(*paths: str) -> set[int]:
    """Read FLAME vertex-index lists (one or more ``neck.txt`` / ``boundary.txt``).

    Robust to whitespace- or comma-separated integers, one or many per line.
    """
    idx: set[int] = set()
    for path in paths:
        with open(path, "r") as fh:
            for line in fh:
                for tok in line.replace(",", " ").split():
                    try:
                        idx.add(int(tok))
                    except ValueError:
                        continue
    return idx


def build_is_face(binding_face_id, faces, excluded_vertices, require: str = "all"):
    """Lift the vertex-level face indicator to a per-Gaussian ``is_face`` mask.

    Parameters
    ----------
    binding_face_id : int array ``[N]`` — the triangle each Gaussian is bound to.
    faces           : int array ``[F, 3]`` — FLAME triangle -> vertex indices.
    excluded_vertices : iterable[int] — neck u boundary vertex indices (non-face).
    require : ``'all'`` (default) => triangle is face iff all 3 vertices are face;
              ``'majority'`` => >= 2 of 3.

    Returns a NumPy bool array ``[N]`` (computed once; ship it in the ``.pt``).
    """
    faces = _xp.to_numpy(faces).astype(np.int64)          # [F, 3]
    bfid = _xp.to_numpy(binding_face_id).astype(np.int64)  # [N]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be [F,3], got {faces.shape}")
    if bfid.size and (bfid.max() >= faces.shape[0] or bfid.min() < 0):
        raise ValueError("binding_face_id out of range for the given faces topology")

    n_verts = int(faces.max()) + 1
    is_face_vertex = np.ones(n_verts, dtype=bool)
    for v in excluded_vertices:
        v = int(v)
        if 0 <= v < n_verts:
            is_face_vertex[v] = False

    tri_vertex_face = is_face_vertex[faces]               # [F, 3]
    if require == "all":
        tri_is_face = tri_vertex_face.all(axis=1)
    elif require == "majority":
        tri_is_face = tri_vertex_face.sum(axis=1) >= 2
    else:
        raise ValueError("require must be 'all' or 'majority'")

    return tri_is_face[bfid]                               # [N] bool
