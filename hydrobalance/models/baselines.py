"""Baselines for quick sanity checks.

English:
    - Last-value (persistence)
    - Seasonal naive (repeat last season)

日本語:
    - 直前値（persistence）
    - 季節ナイーブ（前季の繰り返し）
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def predict_last(train_df: pd.DataFrame, horizon: int) -> np.ndarray:
    last = float(train_df["value"].iloc[-1])
    return np.full(horizon, last, dtype=float)


def predict_seasonal_naive(train_df: pd.DataFrame, horizon: int, season: int = 365) -> np.ndarray:
    x = train_df["value"].to_numpy(dtype=float)
    if len(x) <= season:
        return predict_last(train_df, horizon)
    tail = x[-season:]
    rep = int(np.ceil(horizon / season))
    return np.tile(tail, rep)[:horizon].astype(float)
