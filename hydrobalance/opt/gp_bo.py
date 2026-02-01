"""Simple Bayesian Optimization with Gaussian Processes.

English:
    This is a compact educational implementation of Bayesian Optimization:
    - GaussianProcessRegressor surrogate
    - Expected Improvement acquisition
    - Random sampling in bounded box (works well for low-dimensional spaces)

日本語:
    ガウス過程による簡易ベイズ最適化の自作実装です。
    - GaussianProcessRegressor（サロゲート）
    - Expected Improvement（獲得関数）
    - 境界ボックス内でのランダムサンプリング（低次元向け）

Why custom?
    We avoid extra dependencies (Optuna/skopt) so that the repo is easy to read.

Disclaimer:
    For serious research, use a mature BO library.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm


@dataclass(frozen=True)
class Bound:
    low: float
    high: float


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    # EN: EI for minimization (lower is better).
    # JP: 最小化問題のEI（小さいほど良い）を計算します。
    imp = best - mu - xi
    z = imp / np.maximum(sigma, 1e-9)
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma < 1e-9] = 0.0
    return ei


def bayes_opt_minimize(
    objective: Callable[[np.ndarray], float],
    bounds: List[Bound],
    n_init: int = 8,
    n_iter: int = 25,
    n_candidates: int = 2048,
    seed: int = 42,
    xi: float = 0.01,
) -> Dict[str, object]:
    """Minimize objective(x) with Bayesian optimization.

    Parameters
    ----------
    objective:
        function taking x (shape [d]) and returning scalar loss.
    bounds:
        list of Bound(low, high) for each dimension.
    n_init:
        number of random warm-start points.
    n_iter:
        number of BO iterations.
    n_candidates:
        random candidate points used to maximize acquisition.
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)

    def sample(n: int) -> np.ndarray:
        xs = np.empty((n, d), dtype=float)
        for j, b in enumerate(bounds):
            xs[:, j] = rng.uniform(b.low, b.high, size=n)
        return xs

    X = sample(n_init)
    y = np.array([objective(x) for x in X], dtype=float)

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)

    for _ in range(n_iter):
        gp.fit(X, y)
        cand = sample(n_candidates)
        mu, std = gp.predict(cand, return_std=True)
        best = float(np.min(y))
        ei = _expected_improvement(mu, std, best=best, xi=xi)
        x_next = cand[int(np.argmax(ei))]
        y_next = float(objective(x_next))

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)

    best_idx = int(np.argmin(y))
    return {
        "x_best": X[best_idx].tolist(),
        "y_best": float(y[best_idx]),
        "X": X.tolist(),
        "y": y.tolist(),
    }
