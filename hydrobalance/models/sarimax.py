"""Seasonal ARIMA forecaster (SARIMAX).

English:
    We include SARIMAX as a transparent baseline and as a bridge to MCM-style
    solutions (Seasonal ARIMA is commonly used for reservoir level series).

日本語:
    透明性の高いベースラインとしてSARIMAXを実装します。
    MCMの典型解法（季節ARIMA）との接続点にもなります。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass(frozen=True)
class SarimaxSpec:
    order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 12)
    trend: str | None = "c"


def fit_predict(train_df: pd.DataFrame, horizon: int, spec: SarimaxSpec) -> np.ndarray:
    y = train_df["value"].to_numpy(dtype=float)
    model = SARIMAX(
        y,
        order=spec.order,
        seasonal_order=spec.seasonal_order,
        trend=spec.trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=horizon)
    return np.asarray(fc, dtype=float)
