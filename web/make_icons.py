#!/usr/bin/env python3
"""Draws the viewer's home-screen icons (no image library needed): a gold spire on a dark ground.

    python3 web/make_icons.py      # writes web/icon-180.png (iPhone, iPad), icon-192.png and icon-512.png

The icon is drawn for the viewer (no game art). Each pixel is sampled 4 x 4 times for smooth edges.
"""
import os
import struct
import zlib

WEB = os.path.dirname(os.path.abspath(__file__))
BG_TOP, BG_BOTTOM = (34, 30, 46), (14, 14, 18)
GOLD, GOLD_DARK, RED = (232, 185, 35), (168, 120, 20), (210, 60, 52)


def inside(x, y):
    """The colour at (x, y) in the unit square, or None for the ground."""
    # A red gem high on the spire.
    if (x - 0.5) ** 2 + (y - 0.30) ** 2 <= 0.032 ** 2:
        return RED
    # Window slits.
    for wy in (0.48, 0.63):
        if abs(x - 0.5) <= 0.012 and wy <= y <= wy + 0.07:
            return BG_BOTTOM
    # The spire: slim, lit from the left (the right half in shade), on a stepped base.
    top, bottom, half = 0.07, 0.80, 0.12
    if top <= y <= bottom:
        w = half * (y - top) / (bottom - top)
        if abs(x - 0.5) <= w:
            return GOLD if x <= 0.5 else GOLD_DARK
    if 0.32 <= x <= 0.68 and 0.80 <= y <= 0.85:
        return GOLD_DARK
    if 0.26 <= x <= 0.74 and 0.85 <= y <= 0.89:
        return GOLD
    return None


def pixel(px, py, size, n=4):
    acc = [0.0, 0.0, 0.0]
    for i in range(n):
        for j in range(n):
            x, y = (px + (i + 0.5) / n) / size, (py + (j + 0.5) / n) / size
            c = inside(x, y)
            if c is None:
                c = tuple(BG_TOP[k] + (BG_BOTTOM[k] - BG_TOP[k]) * y for k in range(3))
            for k in range(3):
                acc[k] += c[k]
    return bytes(int(v / (n * n)) for v in acc)


def png(path, size):
    rows = b"".join(b"\x00" + b"".join(pixel(x, y, size) for x in range(size)) for y in range(size))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))  # noqa: E731
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


if __name__ == "__main__":
    for size in (180, 192, 512):
        png(os.path.join(WEB, f"icon-{size}.png"), size)
        print(f"wrote web/icon-{size}.png")
