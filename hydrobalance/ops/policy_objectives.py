"""Example objective for reservoir operation policy.

English:
    We illustrate a parametric monthly release policy:
        release = clamp(a0 + a1 * storage_norm + a2 * season_sin + a3 * season_cos)

    This objective is deliberately simple so that:
      - it can be optimized with NSGA-II
      - it is easy to explain in interviews

日本語:
    月次の放流ポリシーをパラメトリックに表現し、NSGA-IIで最適化する例です。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .hydropower import power_mw, HydroSpec


@dataclass(frozen=True)
class SimSpec:
    # EN: toy simulator parameters (for demo). Replace with a real mass-balance model if needed.
    # JP: デモ用の簡易シミュレータ。実運用では水収支モデルに置換してください。
    storage_min: float = 0.0
    storage_max: float = 1.0
    inflow_mean: float = 0.5
    inflow_amp: float = 0.15
    demand: float = 0.45
    head_m: float = 80.0
    Q_scale: float = 400.0  # convert normalized release -> m^3/s


def simulate(policy: np.ndarray, months: int = 240, spec: SimSpec = SimSpec(), seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate storage and power time series.

    Returns
    -------
    storage: [months]
    power_mw: [months]
    """
    rng = np.random.default_rng(seed)
    a0, a1, a2, a3 = policy.tolist()
    storage = np.zeros(months, dtype=float)
    power = np.zeros(months, dtype=float)
    storage[0] = 0.7 * (spec.storage_max - spec.storage_min)

    for t in range(1, months):
        season = 2 * np.pi * (t % 12) / 12.0
        inflow = spec.inflow_mean + spec.inflow_amp * np.sin(season) + 0.03 * rng.standard_normal()
        inflow = np.clip(inflow, 0.0, 1.0)

        storage_norm = (storage[t-1] - spec.storage_min) / (spec.storage_max - spec.storage_min + 1e-12)
        release = a0 + a1 * storage_norm + a2 * np.sin(season) + a3 * np.cos(season)
        release = np.clip(release, 0.0, 1.0)

        # EN: simple mass balance: S(t)=S(t-1)+inflow-release-demand
        # JP: 簡易水収支
        s = storage[t-1] + inflow - release - spec.demand
        storage[t] = np.clip(s, spec.storage_min, spec.storage_max)

        Q = release * spec.Q_scale
        power[t] = float(power_mw(Q, spec.head_m, HydroSpec())[()])
    return storage, power


def objectives(policy: np.ndarray) -> np.ndarray:
    """Two-objective minimization.

    EN:
        f1 = -mean(storage)  (maximize storage)
        f2 = -mean(power)    (maximize power)

    JP:
        f1 = -平均貯水量（貯水量を最大化）
        f2 = -平均発電量（発電量を最大化）
    """
    storage, power = simulate(policy)
    return np.array([-storage.mean(), -power.mean()], dtype=float)
