from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch


def _hilbert2d(x: np.ndarray, y: np.ndarray, bits: int) -> np.ndarray:
    n = 1 << int(bits)
    x = x.astype(np.int64, copy=True)
    y = y.astype(np.int64, copy=True)
    order = np.zeros_like(x, dtype=np.uint64)
    step = n // 2
    while step > 0:
        rx = ((x & step) > 0).astype(np.int64)
        ry = ((y & step) > 0).astype(np.int64)
        order += np.uint64(step * step) * ((3 * rx) ^ ry).astype(np.uint64)

        flip = ry == 0
        reverse = flip & (rx == 1)
        x_rev = (n - 1) - x
        y_rev = (n - 1) - y
        x = np.where(reverse, x_rev, x)
        y = np.where(reverse, y_rev, y)
        old_x = x.copy()
        x = np.where(flip, y, x)
        y = np.where(flip, old_x, y)
        step //= 2
    return order


def _morton2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    def expand(v: np.ndarray) -> np.ndarray:
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    return (expand(x) | (expand(y) << 1)).astype(np.uint32)


@lru_cache(maxsize=128)
def curve_indices(height: int, width: int, curve: str = "fractal") -> torch.Tensor:
    y, x = np.mgrid[0:height, 0:width]
    curve = str(curve).lower()
    if curve in {"fractal", "hilbert"}:
        bits = int(np.ceil(np.log2(max(height, width))))
        order = _hilbert2d(x.ravel(), y.ravel(), bits)
    elif curve == "morton":
        order = _morton2d(x.ravel(), y.ravel())
    elif curve == "raster":
        order = np.arange(height * width, dtype=np.int64)
    else:
        raise ValueError(f"unknown curve: {curve}")
    return torch.from_numpy(np.argsort(order).astype(np.int64))


def flatten_by_curve(x: torch.Tensor, curve: str = "fractal") -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = x.shape
    idx = curve_indices(height, width, curve).to(x.device)
    seq = x.view(batch, channels, -1)[:, :, idx]
    return seq.transpose(1, 2).contiguous(), idx


def unflatten_by_curve(seq: torch.Tensor, idx: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, _, channels = seq.shape
    inv = torch.argsort(idx)
    seq = seq[:, inv, :]
    return seq.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
