#!/usr/bin/env python3
"""Bayesian Optimization for SARIMAX.

Example:
  python scripts/tune_sarimax_bo.py --csv data/raw/usgs.csv --out artifacts/sarimax_bo.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd

# EN: Allow running as a script without install.
# JP: インストール前でも実行できるようにパスを追加。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrobalance.data.preprocess import regularize, CleanSpec
from hydrobalance.config import BacktestConfig, BOConfig
from hydrobalance.opt.tune_sarimax import tune_sarimax


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="artifacts/sarimax_bo.json")
    p.add_argument("--horizon", type=int, default=30)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[["datetime", "value"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = regularize(df, CleanSpec(freq="D", fill="interpolate"))

    bt = BacktestConfig(horizon=args.horizon, step=7, min_train=365)
    bo = BOConfig(n_init=6, n_iter=18, random_seed=42, exploration=0.01)

    result = tune_sarimax(df, bt, bo)
    serializable = {
        "best_spec": {
            "order": result["best_spec"].order,
            "seasonal_order": result["best_spec"].seasonal_order,
            "trend": result["best_spec"].trend,
        },
        "bo_trace": result["bo_trace"],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved: {args.out}")
    print("Best:", serializable["best_spec"])


if __name__ == "__main__":
    main()
