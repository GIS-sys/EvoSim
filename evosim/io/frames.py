"""Dependency-light frame rendering (numpy) for headless time-lapse; PNG via Pillow, else PPM-in-PNG-name fallback."""
from __future__ import annotations

import zlib

import numpy as np


def render_numpy(sim, size=512):
    c = sim.state.cells
    al = c.alive.numpy() != 0
    W, H = sim.state.dims.world
    img = np.zeros((size, size, 3), np.uint8)
    nut = sim.state.fields.nutrient.numpy()
    if nut.max() > 0:
        n = (np.clip(nut / max(nut.max(), 1e-9), 0, 1) * 60).astype(np.uint8)
        idx = (np.arange(size) * nut.shape[0] / size).astype(int)
        idy = (np.arange(size) * nut.shape[1] / size).astype(int)
        img[..., 1] = n[np.ix_(idx, idy)].T[::-1]
    if al.any():
        pos = c.pos.numpy()[al]
        lin = c.lineage_id.numpy()[al].astype(np.uint32)
        h = (lin * np.uint32(2654435761)) >> np.uint32(8)
        col = np.stack([(h & 255), ((h >> 8) & 255), ((h >> 16) & 255)], 1).astype(np.uint8) | 64
        x = np.clip((pos[:, 0] / W * size).astype(int), 0, size - 1)
        y = np.clip(size - 1 - (pos[:, 1] / H * size).astype(int), 0, size - 1)
        img[y, x] = col
    return img


def save_png(path, img):
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
        return
    except ImportError:
        pass
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(t, d):
        c = zlib.crc32(t + d) & 0xFFFFFFFF
        return len(d).to_bytes(4, "big") + t + d + c.to_bytes(4, "big")
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", w.to_bytes(4, "big") + h.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00") + \
        chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
