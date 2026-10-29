import argparse
from pathlib import Path

import numpy as np
import scipy.stats as st
import tifffile
import torch
import torch.nn.functional as F
import zarr
from safetensors.torch import load_file
from timesformer_pytorch import TimeSformer

WINDOW = 64
FRAMES = 26
CLIP = 200


def gaussian_kernel(size, sigma):
    x = np.linspace(-sigma, sigma, size + 1)
    k1 = np.diff(st.norm.cdf(x))
    k2 = np.outer(k1, k1)
    return k2 / k2.max()


def load_model(weights, device):
    model = TimeSformer(dim=512, image_size=WINDOW, patch_size=16, num_frames=FRAMES, num_classes=16, channels=1,
                        depth=8, heads=6, dim_head=64, attn_dropout=0.1, ff_dropout=0.1)
    state = {k.removeprefix("backbone."): v for k, v in load_file(weights).items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(model, volume, stride, batch, device):
    _, height, width = volume.shape
    pad_h, pad_w = (-height) % WINDOW, (-width) % WINDOW
    volume = np.pad(volume, [(0, 0), (0, pad_h), (0, pad_w)])
    total = np.zeros(volume.shape[1:], np.float32)
    weight = np.zeros(volume.shape[1:], np.float32)
    kernel = torch.tensor(gaussian_kernel(WINDOW, 1), device=device, dtype=torch.float32)
    corners = [(y, x) for y in range(0, volume.shape[1] - WINDOW + 1, stride)
               for x in range(0, volume.shape[2] - WINDOW + 1, stride)]
    corners = [(y, x) for y, x in corners if volume[FRAMES // 2, y:y + WINDOW, x:x + WINDOW].any()]
    for i in range(0, len(corners), batch):
        part = corners[i:i + batch]
        windows = np.stack([volume[:, y:y + WINDOW, x:x + WINDOW] for y, x in part]).astype(np.float32)
        tensor = torch.from_numpy(np.minimum(windows, CLIP) / 255.0).to(device)[:, :, None]
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(tensor).view(-1, 1, 4, 4)
        maps = (F.interpolate(torch.sigmoid(logits.float()), scale_factor=16, mode="bilinear")[:, 0] * kernel).cpu().numpy()
        for (y, x), m in zip(part, maps):
            total[y:y + WINDOW, x:x + WINDOW] += m
            weight[y:y + WINDOW, x:x + WINDOW] += 1
    out = total / np.maximum(weight, 1)
    return out[:height, :width]


def main():
    parser = argparse.ArgumentParser(description="Run the 2023 Grand Prize TimeSformer ink model on a 66 layer render.")
    parser.add_argument("render", help="zarr group from vc_render_tifxyz with a (layers, rows, cols) array 0")
    parser.add_argument("weights", help="model.safetensors of scrollprize/timesformer_GP_scroll1")
    parser.add_argument("out", type=Path, help="output tif; the reversed layer order goes to <out>_reverse.tif")
    parser.add_argument("--layer-start", type=int, default=17)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    volume = np.asarray(zarr.open_group(args.render, mode="r")["0"][args.layer_start:args.layer_start + FRAMES])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, layers in (("", volume), ("_reverse", volume[::-1])):
        prediction = predict(model, np.ascontiguousarray(layers), args.stride, args.batch, device)
        target = args.out.with_name(args.out.stem + suffix + args.out.suffix)
        tifffile.imwrite(target, np.clip(prediction * 255, 0, 255).astype(np.uint8))
        print(f"{target} {prediction.shape} mean {prediction.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
