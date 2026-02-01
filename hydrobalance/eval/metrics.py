"""Forecasting metrics.

English:
    Provide common error metrics for regression / forecasting.

日本語:
    回帰・予測の代表的な誤差指標を実装します。
"""

from __future__ import annotations
import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def mase(
    y_true,
    y_pred,
    y_insample,
    seasonality: int = 1,
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Scaled Error (MASE).

    English:
        MASE scales the MAE by the MAE of a naive (seasonal) random-walk
        forecaster computed on the training (in-sample) data.

    日本語:
        学習データ上の季節ナイーブのMAEで割ることで、スケールに依存しない
        誤差指標(MASE)を計算します。

    Notes / 注意:
        - seasonality=1 gives the standard random-walk scaling.
        - for daily series with yearly seasonality, you can set seasonality=365.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_insample = np.asarray(y_insample, dtype=float)

    if len(y_insample) <= seasonality:
        return float("nan")

    # Naive seasonal in-sample forecast error
    scale = np.mean(np.abs(y_insample[seasonality:] - y_insample[:-seasonality]))
    scale = float(max(scale, eps))
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def peak_rmse(y_true, y_pred, q: float = 0.95) -> float:
    """RMSE computed on the top-q quantile of y_true (flood/peak focus).

    EN: Useful for hydrology where peak levels matter.
    JP: 洪水などピーク水位を重視した評価に有用です。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    thr = np.quantile(y_true, q)
    mask = y_true >= thr
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
