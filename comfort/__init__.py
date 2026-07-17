"""comfort — the live comfort-budget-placement library.

This package is the single source of truth for the method described in the paper
(``paper/main.tex`` Algorithm 1, ``paper/supplemental/supplemental.tex``): rendering a
deforming Gaussian head avatar on an autostereoscopic display by seating the fixated face,
with a frozen calibrated *rigid* depth offset, at a chosen operating point within the
*two-sided* stereo comfort budget (``z0 < 0`` recedes/behind for maximum margin; ``z0 > 0``
a bounded pop-out/front for impact). Comfort is the constraint envelope; the operating point
is a deliberate choice on the impact<->comfort axis.

Design notes
------------
* **Array-agnostic.** The numeric core works on either NumPy arrays or torch tensors, so the
  device-free tests run with NumPy alone (no torch/CUDA needed) while the real pipeline hands
  torch tensors (optionally on the GPU). See :mod:`comfort._xp`.
* **Pure functions.** Placement is a side-effect-free ``(means, z0) -> means'`` scalar shift,
  matching the one-line C++ operation in the SRD renderer.
* **Units are centimetres** everywhere on the A->B bridge (see the ``.pt`` schema in
  :mod:`comfort.schema`).
"""

from . import geometry  # noqa: F401
