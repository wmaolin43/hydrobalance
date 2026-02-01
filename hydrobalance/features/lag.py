"""Lag/rolling/time features for time-series ML models.

English:
    Convert a univariate time series into a supervised learning dataset.
    All features are built using information available up to time t to avoid
    data leakage.

日本語:
    1変量時系列を教師あり学習のデータセットに変換します。
    特徴量は必ず時刻tまでの情報のみから作成し、リークを防止します。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LagSpec:
    """Configuration for feature generation.

    English:
        lags:
            Past offsets used as lag features (e.g., 1, 2, 7).
        rolling_windows:
            Window sizes for rolling mean/std computed from shifted series.
        rolling_median_windows:
            Window sizes for rolling median and IQR.
        diff_lags:
            Differences value(t) - value(t-k) using shifted series.
        ewm_spans:
            Exponential moving average features (span).
        use_cyclical_time:
            If True, encode calendar fields using sin/cos.

    日本語:
        lags:
            ラグ特徴（例: 1, 2, 7）。
        rolling_windows:
            移動平均/標準偏差（shift後の系列から計算）。
        rolling_median_windows:
            移動中央値とIQR。
        diff_lags:
            差分特徴 value(t) - value(t-k)。
        ewm_spans:
            指数移動平均（EMA）。
        use_cyclical_time:
            Trueなら周期特徴をsin/cosで表現。
    """

    lags: List[int]
    rolling_windows: Optional[List[int]] = None
    rolling_median_windows: Optional[List[int]] = None
    diff_lags: Optional[List[int]] = None
    ewm_spans: Optional[List[int]] = None
    use_cyclical_time: bool = True


def _cyclic(v: pd.Series, period: float, prefix: str) -> pd.DataFrame:
    """Cyclic encoding.

    EN: represent periodic variable with sin/cos.
    JP: 周期変数をsin/cosで表現。
    """
    x = 2.0 * np.pi * (v.astype(float) / period)
    return pd.DataFrame({f"{prefix}_sin": np.sin(x), f"{prefix}_cos": np.cos(x)})


def _add_time_features(x: pd.DataFrame, dt: pd.Series, use_cyclical: bool) -> pd.DataFrame:
    """Add calendar features.

    EN:
        We include both coarse and cyclic encodings. This improves tree/NN models
        without relying on external covariates.

    JP:
        外生変数がなくてもモデルが季節性を学べるように、カレンダー特徴を追加します。
    """
    dt = pd.to_datetime(dt)
    # Always provide raw integer features (useful for linear models)
    x["year"] = dt.dt.year
    x["month"] = dt.dt.month
    x["dow"] = dt.dt.dayofweek
    x["dayofyear"] = dt.dt.dayofyear

    # Hour features only if available (e.g., 15min / hourly data)
    if hasattr(dt.dt, "hour"):
        x["hour"] = dt.dt.hour

    if use_cyclical:
        x = pd.concat(
            [
                x,
                _cyclic(x["month"], 12.0, "month"),
                _cyclic(x["dow"], 7.0, "dow"),
                _cyclic(x["dayofyear"], 365.25, "doy"),
            ],
            axis=1,
        )
        if "hour" in x:
            x = pd.concat([x, _cyclic(x["hour"], 24.0, "hour")], axis=1)

    return x


def make_supervised(df: pd.DataFrame, spec: LagSpec, horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
    """Build X, y for forecasting horizon steps ahead.

    EN:
        y(t) = value(t + horizon)

    JP:
        目的変数: y(t) = value(t + horizon)

    Notes / 注意:
        - Feature columns are created from value(t), value(t-1), ... only.
        - Rolling stats are computed on value shifted by 1 step.
    """
    if "datetime" not in df or "value" not in df:
        raise ValueError("df must have columns ['datetime', 'value']")

    x = pd.DataFrame({"datetime": df["datetime"]})

    # Lag features
    for k in spec.lags:
        x[f"lag_{k}"] = df["value"].shift(k)

    # Difference features (robustly capturing changes)
    if spec.diff_lags:
        for k in spec.diff_lags:
            x[f"diff_{k}"] = df["value"] - df["value"].shift(k)

    # Rolling mean/std (use shift(1) to avoid peeking)
    if spec.rolling_windows:
        base = df["value"].shift(1)
        for w in spec.rolling_windows:
            x[f"roll_mean_{w}"] = base.rolling(w).mean()
            x[f"roll_std_{w}"] = base.rolling(w).std()

    # Rolling median/IQR (more robust to spikes)
    if spec.rolling_median_windows:
        base = df["value"].shift(1)
        for w in spec.rolling_median_windows:
            med = base.rolling(w).median()
            q75 = base.rolling(w).quantile(0.75)
            q25 = base.rolling(w).quantile(0.25)
            x[f"roll_median_{w}"] = med
            x[f"roll_iqr_{w}"] = (q75 - q25)

    # Exponentially-weighted mean (EMA)
    if spec.ewm_spans:
        base = df["value"].shift(1)
        for s in spec.ewm_spans:
            x[f"ewm_mean_{s}"] = base.ewm(span=s, adjust=False).mean()

    # Calendar features
    x = _add_time_features(x, df["datetime"], use_cyclical=spec.use_cyclical_time)

    # Target
    y = df["value"].shift(-horizon)

    # Keep only rows with complete features and target
    feat_cols = [c for c in x.columns if c != "datetime"]
    mask = x[feat_cols].notna().all(axis=1) & y.notna()
    x = x.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return x, y
