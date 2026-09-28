"""Colour-vision check of the method palette (Python port of the core dataviz validator check).

Simulates protanopia / deuteranopia / tritanopia (Machado et al. 2009, severity 1.0, linear RGB),
converts to OKLab and reports the minimum pairwise Delta E (x100). Target >= 8; 6-8 needs secondary
encoding (here: x-position labels, marker shapes); < 6 is a hard fail.
Usage: python analysis/figures/check_palette.py
"""
from __future__ import annotations

import itertools

import numpy as np

from style import COLORS, OPTIMIZERS

MACHADO = {
    "normal": np.eye(3),
    "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}


def hex_to_linear(h: str) -> np.ndarray:
    c = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 1)
    lms = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]]) @ rgb
    lms = np.cbrt(lms)
    return np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]]) @ lms


def report() -> list[tuple]:
    rows = []
    for cond, m in MACHADO.items():
        labs = {k: linear_to_oklab(m @ hex_to_linear(COLORS[k])) for k in OPTIMIZERS}
        pairs = [(a, b, 100 * float(np.linalg.norm(labs[a] - labs[b]))) for a, b in itertools.combinations(OPTIMIZERS, 2)]
        a, b, d = min(pairs, key=lambda p: p[2])
        rows.append((cond, a, b, d))
    return rows


if __name__ == "__main__":
    print("condition | closest pair | min Delta E (OKLab x100) | verdict")
    for cond, a, b, d in report():
        verdict = "PASS (>= 8)" if d >= 8 else ("OK only with secondary encoding (6-8)" if d >= 6 else "FAIL (< 6)")
        print(f"{cond:9s} | {a} vs {b} | {d:5.1f} | {verdict}")
    L = {k: float(linear_to_oklab(hex_to_linear(COLORS[k]))[0]) for k in OPTIMIZERS}
    print("OKLab lightness:", {k: round(v, 2) for k, v in L.items()})
