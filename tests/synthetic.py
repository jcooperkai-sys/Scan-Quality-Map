import numpy as np
from scipy import ndimage


def sheet_stack(size=96, spacing_vx=10.0, thickness_vx=3.0, tilt=0.15, noise=6.0, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((size, size, size), dtype=float)
    depth = y + tilt * x + 0.05 * z
    phase = np.mod(depth, spacing_vx)
    sheets = np.exp(-0.5 * ((phase - spacing_vx / 2) / (thickness_vx / 2)) ** 2)
    volume = 40 + 160 * sheets + rng.normal(0, noise, sheets.shape)
    return np.clip(volume, 1, 255).astype(np.uint8)


def hazy(volume, blur_vx=3.0, veil=0.5):
    data = volume.astype(float)
    blurred = ndimage.gaussian_filter(data, blur_vx)
    mixed = (1 - veil) * blurred + veil * blurred.mean()
    return np.clip(mixed, 1, 255).astype(np.uint8)


def foam(size=96, pores=900, radius_vx=2.5, seed=1):
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((size, size, size), dtype=float)
    volume = np.full((size, size, size), 190.0)
    for c in rng.uniform(0, size, size=(pores, 3)):
        d2 = (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2
        volume[d2 < radius_vx ** 2] = 45.0
    volume = ndimage.gaussian_filter(volume, 0.6) + rng.normal(0, 4, volume.shape)
    return np.clip(volume, 1, 255).astype(np.uint8)
