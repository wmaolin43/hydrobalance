#!/usr/bin/env python3
"""NSGA-II demo: optimize a simple reservoir release policy.

Example:
  python scripts/optimize_policy_nsga2.py --out artifacts/policy_pareto.npz
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np

# EN: Allow running as a script without install.
# JP: インストール前でも実行できるようにパスを追加。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrobalance.ops.nsga2 import nsga2, NSGA2Config
from hydrobalance.ops.policy_objectives import objectives
from hydrobalance.ops.topsis import select, TopsisWeights


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="artifacts/policy_pareto.npz")
    p.add_argument("--w1", type=float, default=0.5)
    p.add_argument("--w2", type=float, default=0.5)
    args = p.parse_args()

    bounds = [
        (0.0, 1.0),   # a0
        (-1.0, 1.0),  # a1
        (-1.0, 1.0),  # a2
        (-1.0, 1.0),  # a3
    ]
    cfg = NSGA2Config(pop_size=80, generations=80, seed=42)

    result = nsga2(objectives, bounds=bounds, cfg=cfg)
    Xp = result["pareto_X"]
    Fp = result["pareto_F"]

    idx = select(Fp, TopsisWeights(w1=args.w1, w2=args.w2))
    rec_x = Xp[idx]
    rec_f = Fp[idx]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, pareto_X=Xp, pareto_F=Fp, recommend_x=rec_x, recommend_F=rec_f)
    print(f"Saved Pareto set: {args.out}")
    print("Recommended policy params:", rec_x)
    print("Objectives (min):", rec_f)


if __name__ == "__main__":
    main()
