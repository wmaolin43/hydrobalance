"""Central configuration dataclasses.

English:
    Keep dataset / model / evaluation settings in one place.

日本語:
    データセット・モデル・評価の設定を一箇所で管理します。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class USGSQuery:
    """USGS query parameters.

    EN: Use site + parameter code (e.g., 00065 gage height, 00060 discharge).
    JP: 観測所(site)とパラメータコード（例：00065=水位, 00060=流量）を指定します。
    """

    site: str
    parameter_code: str
    start: str  # ISO date: YYYY-MM-DD
    end: str    # ISO date: YYYY-MM-DD
    service: str = "dv"  # "dv" (daily) or "iv" (instantaneous)


@dataclass(frozen=True)
class BacktestConfig:
    """Rolling-origin evaluation settings."""

    horizon: int = 30      # forecast horizon in steps (days if dv)
    step: int = 7          # move window by this many steps
    min_train: int = 365   # minimum training length
    metric: str = "rmse"   # primary metric


@dataclass(frozen=True)
class BOConfig:
    """Bayesian optimization settings."""

    n_init: int = 8
    n_iter: int = 25
    random_seed: int = 42
    acquisition: str = "ei"  # expected improvement
    exploration: float = 0.01
