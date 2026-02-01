"""Prediction intervals (simple conformal-style).

English:
    In many hydrology settings, point forecasts are not enough; we also want
    *uncertainty* estimates. A practical, assumption-light approach is to use
    conformal-style residual quantiles.

    This module provides helpers to build symmetric intervals:
        [yhat - q, yhat + q]
    where q is a high quantile of |y - yhat| computed on historical errors.

日本語:
    水文では点予測だけでなく不確実性も重要です。ここでは仮定の少ない
    残差分位点に基づく簡易コンフォーマル風の区間を提供します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class IntervalSpec:
    """Configuration for conformal-like intervals.

    EN:
        alpha: miscoverage level (0.1 -> 90% interval)
        method: 'abs' uses absolute residuals |e|.

    JP:
        alpha: 外れる確率（0.1なら90%区間）
        method: 'abs' は絶対残差 |e| を使用します。
    """

    alpha: float = 0.1
    method: str = "abs"


def conformal_radius(errors: Iterable[float], spec: IntervalSpec = IntervalSpec()) -> float:
    """Compute residual quantile radius q.

    EN: q = quantile(|e|, 1-alpha)
    JP: q = 分位点(|e|, 1-alpha)
    """
    e = np.asarray(list(errors), dtype=float)
    if e.size == 0:
        return float("nan")
    if spec.method != "abs":
        raise ValueError("Only method='abs' is supported in this simple helper")
    e = np.abs(e)
    q = float(np.quantile(e, 1.0 - spec.alpha))
    return q


def apply_symmetric_interval(yhat: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) arrays."""
    yhat = np.asarray(yhat, dtype=float)
    r = float(radius)
    return yhat - r, yhat + r
