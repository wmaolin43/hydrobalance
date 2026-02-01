"""Hydropower approximation utilities.

English:
    Convert water release and head into power output.
    This repo uses a simple physics-based approximation:
        P = eta * rho * g * Q * H

    where:
        Q: discharge (m^3/s)
        H: head (m)
        eta: efficiency (0-1)

日本語:
    放流量と有効落差から発電出力を近似します。
    簡易式:
        P = eta * rho * g * Q * H
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HydroSpec:
    eta: float = 0.9
    rho: float = 1000.0
    g: float = 9.80665


def power_watts(Q_m3s: np.ndarray, H_m: np.ndarray, spec: HydroSpec = HydroSpec()) -> np.ndarray:
    Q_m3s = np.asarray(Q_m3s, dtype=float)
    H_m = np.asarray(H_m, dtype=float)
    return spec.eta * spec.rho * spec.g * Q_m3s * H_m


def power_mw(Q_m3s: np.ndarray, H_m: np.ndarray, spec: HydroSpec = HydroSpec()) -> np.ndarray:
    return power_watts(Q_m3s, H_m, spec) / 1e6
