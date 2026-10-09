import numpy as np
from scipy import ndimage

FEATURES = ("coherence", "rhythm", "contrast", "sharpness")


def normalize(block):
    data = block.astype(np.float32)
    inside = data > 0
    if inside.sum() < 64:
        return None, inside
    lo, hi = np.percentile(data[inside], [1, 99])
    scaled = np.clip((data - lo) / max(hi - lo, 1e-6), 0, 1)
    scaled[~inside] = np.nan
    return scaled, inside


def fill_outside(scaled):
    filled = np.where(np.isnan(scaled), np.nanmean(scaled), scaled)
    return filled.astype(np.float32)


def structure_tensor(volume, gradient_sigma, window_sigma):
    gz = ndimage.gaussian_filter(volume, gradient_sigma, order=(1, 0, 0))
    gy = ndimage.gaussian_filter(volume, gradient_sigma, order=(0, 1, 0))
    gx = ndimage.gaussian_filter(volume, gradient_sigma, order=(0, 0, 1))
    pairs = {"zz": gz * gz, "yy": gy * gy, "xx": gx * gx, "zy": gz * gy, "zx": gz * gx, "yx": gy * gx}
    smooth = {k: ndimage.gaussian_filter(v, window_sigma) for k, v in pairs.items()}
    tensor = np.stack([
        np.stack([smooth["zz"], smooth["zy"], smooth["zx"]], axis=-1),
        np.stack([smooth["zy"], smooth["yy"], smooth["yx"]], axis=-1),
        np.stack([smooth["zx"], smooth["yx"], smooth["xx"]], axis=-1),
    ], axis=-2)
    return tensor, np.sqrt(gz * gz + gy * gy + gx * gx)


def coherence_and_normal(volume, inside, voxel_um, stride):
    tensor, gradient = structure_tensor(volume, max(0.7, 8.0 / voxel_um), max(1.5, 40.0 / voxel_um))
    sample = tuple(slice(None, None, stride) for _ in range(3))
    local = tensor[sample][inside[sample]]
    values = np.linalg.eigvalsh(local)
    largest, middle = values[:, 2], values[:, 1]
    weight = largest
    planarity = (largest - middle) / np.maximum(largest, 1e-12)
    coherence = float(np.sum(planarity * weight) / max(np.sum(weight), 1e-12))
    mean_tensor = local.mean(axis=0)
    normal = np.linalg.eigh(mean_tensor)[1][:, 2]
    return coherence, normal, gradient


def rhythm(volume, normal, voxel_um, period_um=(60.0, 400.0), lines=7):
    size = np.array(volume.shape, dtype=float)
    center = (size - 1) / 2
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    length = int(min(size) * 0.8)
    steps = np.arange(length) - length / 2
    offsets = np.linspace(-0.25, 0.25, lines) * min(size)
    trend_sigma = period_um[1] / voxel_um
    lag_lo = max(1, int(round(period_um[0] / voxel_um)))
    lag_hi = min(length // 2, int(round(period_um[1] / voxel_um)))
    if lag_hi <= lag_lo + 1:
        return 0.0
    peaks = []
    for a in offsets:
        for b in offsets:
            start = center + a * u + b * v
            points = start[:, None] + normal[:, None] * steps[None, :]
            profile = ndimage.map_coordinates(volume, points, order=1, mode="nearest")
            profile = profile - ndimage.gaussian_filter1d(profile, trend_sigma, mode="nearest")
            profile = profile - profile.mean()
            energy = np.dot(profile, profile)
            if energy <= 0:
                continue
            corr = np.array([np.dot(profile[:-k], profile[k:]) for k in range(1, lag_hi + 1)]) / energy
            peaks.append(corr[lag_lo - 1:lag_hi].max())
    return float(np.mean(peaks)) if peaks else 0.0


def contrast(scaled):
    values = scaled[~np.isnan(scaled)]
    hist, edges = np.histogram(values, bins=128, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    weights = np.cumsum(hist)
    total = weights[-1]
    sums = np.cumsum(hist * centers)
    mean_all = sums[-1] / total
    w0 = weights / total
    mean0 = sums / np.maximum(weights, 1)
    mean1 = (sums[-1] - sums) / np.maximum(total - weights, 1)
    between = w0 * (1 - w0) * (mean0 - mean1) ** 2
    variance = values.var()
    return float(between.max() / variance) if variance > 0 else 0.0


def sharpness(scaled, gradient, inside, voxel_um):
    values = scaled[inside]
    spread = values.std()
    if spread <= 0:
        return 0.0
    edge = np.percentile(gradient[inside], 90)
    return float(edge / spread * (voxel_um / 7.91))


def features(block, voxel_um, stride=2):
    scaled, inside = normalize(block)
    if scaled is None:
        return None
    volume = fill_outside(scaled)
    coherent, normal, gradient = coherence_and_normal(volume, inside, voxel_um, stride)
    return {
        "coherence": coherent,
        "rhythm": rhythm(volume, normal, voxel_um),
        "contrast": contrast(scaled),
        "sharpness": sharpness(scaled, gradient, inside, voxel_um),
    }
