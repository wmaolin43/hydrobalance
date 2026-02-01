"""Preprocessing utilities.

English:
    - resample to regular frequency
    - fill missing values (forward fill / interpolation)
    - optionally winsorize outliers

日本語:
    - 規則的な頻度にリサンプリング
    - 欠損補完（前方補完 / 補間）
    - 任意で外れ値をwinsorize
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanSpec:
    freq: str = "D"
    fill: Literal["ffill", "interpolate", "none"] = "interpolate"
    winsor_p: float = 0.0  # 0 disables


def regularize(df: pd.DataFrame, spec: CleanSpec) -> pd.DataFrame:
    """Regularize to a fixed frequency.

    Expected input columns: datetime, value
    """
    if "datetime" not in df or "value" not in df:
        raise ValueError("df must have columns ['datetime', 'value']")

    s = df.set_index("datetime")["value"].sort_index()
    s = s.resample(spec.freq).mean()

    if spec.fill == "ffill":
        s = s.ffill()
    elif spec.fill == "interpolate":
        s = s.interpolate(limit_direction="both")
    elif spec.fill == "none":
        pass
    else:
        raise ValueError("fill must be one of: ffill, interpolate, none")

    if spec.winsor_p and spec.winsor_p > 0:
        lo = np.nanquantile(s.values, spec.winsor_p)
        hi = np.nanquantile(s.values, 1 - spec.winsor_p)
        s = s.clip(lo, hi)

    out = s.to_frame(name="value").reset_index()
    return out
