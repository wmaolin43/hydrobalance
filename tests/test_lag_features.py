from hydrobalance.features.lag import LagSpec, make_supervised
import pandas as pd
import numpy as np

def test_make_supervised_shapes():
    n = 200
    df = pd.DataFrame({
        "datetime": pd.date_range("2020-01-01", periods=n, freq="D"),
        "value": np.arange(n, dtype=float)
    })
    X, y = make_supervised(df, LagSpec(lags=[1,2,7], rolling_windows=[7]), horizon=1)
    assert len(X) == len(y)
    assert "lag_1" in X.columns
