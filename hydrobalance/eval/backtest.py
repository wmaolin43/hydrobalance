"""Rolling-origin backtesting.

English:
    Evaluate forecasters with walk-forward validation, which is more appropriate
    for time series than random CV.

日本語:
    時系列に適したウォークフォワード検証（rolling-origin）を実装します。
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Callable, Dict, Any, List
import numpy as np
import pandas as pd

from ..config import BacktestConfig
from .metrics import rmse, mae, smape, mase, peak_rmse


_METRICS = {"rmse": rmse, "mae": mae, "smape": smape, "mase": mase}


def rolling_origin(
    series_df: pd.DataFrame,
    fit_predict_fn: Callable[[pd.DataFrame, int], np.ndarray],
    cfg: BacktestConfig,
) -> Dict[str, Any]:
    """Run rolling-origin evaluation.

    Parameters
    ----------
    series_df:
        DataFrame with columns ['datetime', 'value'] and regular frequency.
    fit_predict_fn:
        Callable(train_df, horizon)->yhat array length horizon.
    cfg:
        BacktestConfig.

    Returns
    -------
    dict with per-fold metrics and global summary.
    """
    if cfg.metric not in _METRICS:
        raise ValueError(f"Unknown metric: {cfg.metric}")

    values = series_df["value"].to_numpy(dtype=float)
    n = len(values)

    folds = []
    start = cfg.min_train
    while start + cfg.horizon <= n:
        train_df = series_df.iloc[:start].copy()
        test_df = series_df.iloc[start : start + cfg.horizon].copy()

        yhat = fit_predict_fn(train_df, cfg.horizon)
        y = test_df["value"].to_numpy(dtype=float)

        fold = {
            "train_end": train_df["datetime"].iloc[-1],
            "test_start": test_df["datetime"].iloc[0],
            "test_end": test_df["datetime"].iloc[-1],
            "rmse": rmse(y, yhat),
            "mae": mae(y, yhat),
            "smape": smape(y, yhat),
            # EN: scale with a random-walk baseline on train split.
            # JP: 学習データ上のナイーブ誤差でスケールします。
            "mase": mase(y, yhat, y_insample=train_df["value"].to_numpy(dtype=float), seasonality=1),
            # EN/JP: peak-focused metric (optional but informative for floods).
            "peak_rmse": peak_rmse(y, yhat, q=0.95),
        }
        folds.append(fold)
        start += cfg.step

    out = {
        "config": asdict(cfg),
        "n_folds": len(folds),
        "folds": folds,
        "mean_rmse": float(np.mean([f["rmse"] for f in folds])) if folds else None,
        "mean_mae": float(np.mean([f["mae"] for f in folds])) if folds else None,
        "mean_smape": float(np.mean([f["smape"] for f in folds])) if folds else None,
        "mean_mase": float(np.mean([f["mase"] for f in folds])) if folds else None,
        "mean_peak_rmse": float(np.nanmean([f["peak_rmse"] for f in folds])) if folds else None,
    }
    return out
