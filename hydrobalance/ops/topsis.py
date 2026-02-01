"""TOPSIS selection from a Pareto front.

English:
    Select a compromise solution from Pareto set by closeness to ideal point.

日本語:
    パレート集合から妥協解を選ぶためのTOPSIS実装です。
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TopsisWeights:
    w1: float = 0.5
    w2: float = 0.5


def select(F: np.ndarray, weights: TopsisWeights = TopsisWeights()) -> int:
    # EN: F is assumed to be minimization objectives; ideal is min per axis.
    # JP: 目的関数は最小化と仮定。理想点=各軸の最小。
    F = np.asarray(F, dtype=float)
    # normalize
    denom = np.sqrt((F**2).sum(axis=0))
    denom = np.where(denom < 1e-12, 1.0, denom)
    R = F / denom
    W = np.array([weights.w1, weights.w2], dtype=float)
    V = R * W

    ideal = np.min(V, axis=0)
    nadir = np.max(V, axis=0)

    d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - nadir) ** 2).sum(axis=1))
    score = d_neg / np.maximum(d_pos + d_neg, 1e-12)
    return int(np.argmax(score))
