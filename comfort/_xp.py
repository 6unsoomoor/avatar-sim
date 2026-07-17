"""Tiny array-namespace shim so the comfort core runs on NumPy *or* torch.

The device-free tests exercise the pure math with NumPy (always available); the real
pipeline (exporter, eval sweep, B-side parity) hands torch tensors, possibly on the GPU.
The only operations that differ by name between the two libraries are wrapped here.
"""

from __future__ import annotations

import numpy as np

try:  # torch is optional — absent in the device-free test/CI environment
    import torch as _torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - exercised only where torch is installed
    _torch = None
    _HAS_TORCH = False


def has_torch() -> bool:
    return _HAS_TORCH


def is_tensor(x) -> bool:
    return _HAS_TORCH and isinstance(x, _torch.Tensor)


def xp_of(x):
    """Return the array module (``numpy`` or ``torch``) for ``x``."""
    return _torch if is_tensor(x) else np


def arctan(x):
    return _torch.atan(x) if is_tensor(x) else np.arctan(x)


def sqrt(x):
    return _torch.sqrt(x) if is_tensor(x) else np.sqrt(x)


def abs_(x):
    return _torch.abs(x) if is_tensor(x) else np.abs(x)


def clip(x, lo, hi):
    return _torch.clamp(x, lo, hi) if is_tensor(x) else np.clip(x, lo, hi)


def quantile(x, q):
    if is_tensor(x):
        return _torch.quantile(x, float(q))
    return np.quantile(x, q)


def to_float(x) -> float:
    """Extract a Python float from a 0-d array / tensor / scalar."""
    if is_tensor(x):
        return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray):
        return float(x.reshape(()).item())
    return float(x)


def masked_select_col(arr, mask, col):
    """Return ``arr[mask, col]`` for either backend (bool-mask row selection)."""
    return arr[mask, col]


def clone(x):
    return x.clone() if is_tensor(x) else x.copy()


def to_numpy(x):
    if is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mean(x):
    return _torch.mean(x) if is_tensor(x) else np.mean(x)


def count_true(mask) -> int:
    """Number of True entries in a bool mask (either backend)."""
    return int(mask.sum().item()) if is_tensor(mask) else int(np.count_nonzero(mask))


def max_axis(x, axis: int):
    if is_tensor(x):
        return _torch.max(x, dim=axis).values
    return np.max(x, axis=axis)


def cos(x):
    return _torch.cos(x) if is_tensor(x) else np.cos(x)


def sin(x):
    return _torch.sin(x) if is_tensor(x) else np.sin(x)


def arccos(x):
    return _torch.acos(x) if is_tensor(x) else np.arccos(x)


def asarray_like(values, ref):
    """Build a 1-D array of ``values`` on the same backend/device/dtype as ``ref``."""
    if is_tensor(ref):
        return _torch.as_tensor(values, dtype=ref.dtype, device=ref.device)
    return np.asarray(values, dtype=np.float64)
