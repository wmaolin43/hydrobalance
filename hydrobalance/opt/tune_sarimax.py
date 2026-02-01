"""Bayesian tuning for SARIMAX hyperparameters.

English:
    Tune (p,d,q,P,D,Q,s) in a bounded small space, using rolling-origin RMSE.

日本語:
    SARIMAXのハイパーパラメータをベイズ最適化で探索します（rolling-origin RMSE）。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

from ..config import BacktestConfig, BOConfig
from ..eval.backtest import rolling_origin
from ..models.sarimax import SarimaxSpec, fit_predict as sarimax_fit_predict
from .gp_bo import Bound, bayes_opt_minimize


@dataclass(frozen=True)
class SearchSpace:
    # EN: small integer ranges; kept tight for speed / stability.
    # JP: 探索範囲は小さめに固定（速度と安定性のため）。
    p_max: int = 3
    q_max: int = 3
    P_max: int = 2
    Q_max: int = 2
    d: int = 1
    D: int = 1
    s: int = 12


def tune_sarimax(
    series_df: pd.DataFrame,
    bt_cfg: BacktestConfig,
    bo_cfg: BOConfig,
    space: SearchSpace = SearchSpace(),
) -> Dict[str, Any]:
    bounds = [
        Bound(0, space.p_max + 0.999),
        Bound(0, space.q_max + 0.999),
        Bound(0, space.P_max + 0.999),
        Bound(0, space.Q_max + 0.999),
    ]

    def obj(x: np.ndarray) -> float:
        p = int(x[0]); q = int(x[1]); P = int(x[2]); Q = int(x[3])
        spec = SarimaxSpec(order=(p, space.d, q), seasonal_order=(P, space.D, Q, space.s))
        def fp(train_df, h):
            return sarimax_fit_predict(train_df, h, spec)
        out = rolling_origin(series_df, fp, bt_cfg)
        return float(out["mean_rmse"])

    bo = bayes_opt_minimize(
        obj, bounds=bounds,
        n_init=bo_cfg.n_init,
        n_iter=bo_cfg.n_iter,
        seed=bo_cfg.random_seed,
        xi=bo_cfg.exploration,
    )

    p, q, P, Q = [int(v) for v in bo["x_best"]]
    best_spec = SarimaxSpec(order=(p, space.d, q), seasonal_order=(P, space.D, Q, space.s))
    return {"best_spec": best_spec, "bo_trace": bo}
