#!/usr/bin/env python3
"""Train and backtest forecasters.

This script demonstrates how to:
  - load a CSV produced by fetch_usgs.py
  - regularize it
  - run rolling-origin evaluation for multiple models

Example:
  python scripts/train_forecaster.py --csv data/raw/usgs.csv --model sarimax --horizon 30
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd

# EN: Allow `python scripts/...` without requiring `pip install -e .` first.
# JP: `pip install -e .` 前でも `python scripts/...` が動くようにパスを追加。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrobalance.data.preprocess import regularize, CleanSpec
from hydrobalance.config import BacktestConfig
from hydrobalance.eval.backtest import rolling_origin
from hydrobalance.models.baselines import predict_last, predict_seasonal_naive
from hydrobalance.models.sarimax import fit_predict as sarimax_fit_predict, SarimaxSpec
from hydrobalance.models.gbm import fit_predict as gbm_fit_predict, GBMSpec
from hydrobalance.models.tcn import fit_predict as tcn_fit_predict, TCNSpec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--model", required=True, choices=["last", "seasonal", "sarimax", "gbm", "tcn"])
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--step", type=int, default=7)
    p.add_argument("--min-train", type=int, default=365)
    p.add_argument("--out", default="artifacts/backtest.json")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[["datetime", "value"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = regularize(df, CleanSpec(freq="D", fill="interpolate", winsor_p=0.0))

    cfg = BacktestConfig(horizon=args.horizon, step=args.step, min_train=args.min_train)

    if args.model == "last":
        fp = lambda train_df, h: predict_last(train_df, h)
    elif args.model == "seasonal":
        fp = lambda train_df, h: predict_seasonal_naive(train_df, h, season=365)
    elif args.model == "sarimax":
        spec = SarimaxSpec()
        fp = lambda train_df, h: sarimax_fit_predict(train_df, h, spec)
    elif args.model == "gbm":
        spec = GBMSpec()
        fp = lambda train_df, h: gbm_fit_predict(train_df, h, spec)
    elif args.model == "tcn":
        spec = TCNSpec(epochs=10)  # keep default light in example
        fp = lambda train_df, h: tcn_fit_predict(train_df, h, spec)
    else:
        raise ValueError(args.model)

    out = rolling_origin(df, fp, cfg)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"Saved backtest report: {args.out}")
    print("Summary:", {k: out[k] for k in ["n_folds", "mean_rmse", "mean_mae", "mean_smape"]})


if __name__ == "__main__":
    main()
