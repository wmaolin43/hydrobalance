"""Gradient-boosted tree forecaster.

English:
    A strong non-linear baseline for tabular lag features.

日本語:
    ラグ特徴量の表形式データに強い非線形ベースラインです。
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from ..features.lag import LagSpec, make_supervised


@dataclass(frozen=True)
class GBMSpec:
    # EN: A slightly richer feature set to improve stability.
    # JP: 安定性を上げるため、特徴量を少し増やします。
    lags: list[int] = (1, 2, 3, 7, 14, 30, 60)
    rolling: list[int] = (7, 14, 30)
    rolling_median: list[int] = (7, 30)
    diff_lags: list[int] = (1, 7)
    ewm_spans: list[int] = (7, 30)
    # EN: We set a large upper bound and rely on early stopping.
    # JP: 木の本数は上限を大きくし、早期停止で自動調整します。
    n_estimators: int = 3000
    learning_rate: float = 0.03
    num_leaves: int = 63
    max_depth: int = -1
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 42
    early_stopping_rounds: int = 100
    valid_fraction: float = 0.15


def fit_predict(train_df: pd.DataFrame, horizon: int, spec: GBMSpec) -> np.ndarray:
    """Fit a one-step model and forecast recursively.

    English:
        We *train only once* (1-step ahead) and then roll forward.
        This is significantly faster than re-fitting the model for each
        horizon step.

    日本語:
        1ステップ先のモデルを一度だけ学習し、その後は再帰的に
        未来を予測します。各ステップで再学習しないため高速です。
    """

    lag_spec = LagSpec(
        lags=list(spec.lags),
        rolling_windows=list(spec.rolling),
        rolling_median_windows=list(spec.rolling_median),
        diff_lags=list(spec.diff_lags),
        ewm_spans=list(spec.ewm_spans),
        use_cyclical_time=True,
    )

    df_hist = train_df.copy().reset_index(drop=True)
    X, y = make_supervised(df_hist, lag_spec, horizon=1)

    X_num = X.drop(columns=["datetime"])
    # Time-aware split for early stopping
    n = len(X_num)
    n_valid = max(1, int(n * spec.valid_fraction))
    X_tr, X_va = X_num.iloc[:-n_valid], X_num.iloc[-n_valid:]
    y_tr, y_va = y.iloc[:-n_valid], y.iloc[-n_valid:]

    model = LGBMRegressor(
        n_estimators=spec.n_estimators,
        learning_rate=spec.learning_rate,
        num_leaves=spec.num_leaves,
        max_depth=spec.max_depth,
        subsample=spec.subsample,
        colsample_bytree=spec.colsample_bytree,
        random_state=spec.random_state,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="l2",
        callbacks=[
            # EN: stop when validation no longer improves.
            # JP: 検証データが改善しなくなったら早期停止。
            early_stopping(spec.early_stopping_rounds),
            # EN/JP: keep output quiet in scripts; set to 1 for debugging.
            log_evaluation(period=0),
        ],
    )

    # Recursive forecasting using updated history
    # EN: We generate the next timestamp by using the median timestep.
    # JP: 次の時刻はデータの代表的な時間差（median）で進めます。
    dt_series = pd.to_datetime(df_hist["datetime"])
    if len(dt_series) >= 2:
        step = (dt_series.diff().dropna().median()).to_pytimedelta()
    else:
        step = pd.Timedelta(days=1)

    preds: list[float] = []
    for _ in range(horizon):
        next_dt = pd.to_datetime(df_hist["datetime"].iloc[-1]) + step
        df_tmp = pd.concat(
            [df_hist, pd.DataFrame({"datetime": [next_dt], "value": [np.nan]})],
            ignore_index=True,
        )
        Xn, _ = make_supervised(df_tmp, lag_spec, horizon=1)
        x_last = Xn.drop(columns=["datetime"]).iloc[[-1]]
        p = float(model.predict(x_last)[0])
        preds.append(p)
        df_hist = pd.concat(
            [df_hist, pd.DataFrame({"datetime": [next_dt], "value": [p]})],
            ignore_index=True,
        )

    return np.asarray(preds, dtype=float)
