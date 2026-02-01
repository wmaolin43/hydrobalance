"""Minimal NSGA-II implementation (2 objectives).

English:
    This is a compact implementation intended for portfolio-quality code, not for
    maximum performance. It demonstrates:
      - fast non-dominated sorting
      - crowding distance
      - elitist selection
      - polynomial mutation + SBX crossover (simple versions)

日本語:
    読みやすさを重視したNSGA-IIの簡易実装です。
    - 非支配ソート
    - 混雑距離
    - エリート選択
    - 交叉/突然変異（簡易版）

Use case in this repo:
    Optimize a parametric reservoir release policy with two objectives:
      1) maximize storage (minimize shortage risk)
      2) maximize hydropower generation

We return a Pareto set and optionally a TOPSIS-based recommended point.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any
import numpy as np


@dataclass(frozen=True)
class NSGA2Config:
    pop_size: int = 80
    generations: int = 80
    crossover_prob: float = 0.9
    mutation_prob: float = 0.2
    eta_c: float = 10.0
    eta_m: float = 20.0
    seed: int = 42


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    # EN/JP: Minimization dominance: a dominates b if <= in all and < in at least one.
    return np.all(a <= b) and np.any(a < b)


def fast_non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    n = len(F)
    S = [[] for _ in range(n)]
    n_dom = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(F: np.ndarray, idxs: List[int]) -> np.ndarray:
    # EN: Compute crowding distance for a front.
    # JP: フロント内の混雑距離を計算します。
    m = F.shape[1]
    dist = np.zeros(len(idxs), dtype=float)
    if len(idxs) == 0:
        return dist
    front = F[idxs]
    for j in range(m):
        order = np.argsort(front[:, j])
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = front[order[0], j], front[order[-1], j]
        if fmax - fmin < 1e-12:
            continue
        for k in range(1, len(idxs) - 1):
            dist[order[k]] += (front[order[k + 1], j] - front[order[k - 1], j]) / (fmax - fmin)
    return dist


def sbx_crossover(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, bounds: np.ndarray, eta_c: float) -> Tuple[np.ndarray, np.ndarray]:
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() > 0.5:
            continue
        x1, x2 = p1[i], p2[i]
        if abs(x1 - x2) < 1e-12:
            continue
        low, high = bounds[i]
        if x1 > x2:
            x1, x2 = x2, x1
        u = rng.random()
        beta = 1.0 + (2.0 * (x1 - low) / (x2 - x1))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if u <= 1.0 / alpha:
            betaq = (u * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
        child1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (high - x2) / (x2 - x1))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if u <= 1.0 / alpha:
            betaq = (u * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
        child2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

        c1[i] = np.clip(child1, low, high)
        c2[i] = np.clip(child2, low, high)
    return c1, c2


def poly_mutation(rng: np.random.Generator, x: np.ndarray, bounds: np.ndarray, eta_m: float, p_mut: float) -> np.ndarray:
    y = x.copy()
    for i in range(len(x)):
        if rng.random() > p_mut:
            continue
        low, high = bounds[i]
        if high - low < 1e-12:
            continue
        u = rng.random()
        delta1 = (y[i] - low) / (high - low)
        delta2 = (high - y[i]) / (high - low)
        if u < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1.0))
            deltaq = val ** (1.0 / (eta_m + 1.0)) - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1.0))
            deltaq = 1.0 - val ** (1.0 / (eta_m + 1.0))
        y[i] = np.clip(y[i] + deltaq * (high - low), low, high)
    return y


def nsga2(
    objective: Callable[[np.ndarray], np.ndarray],
    bounds: List[Tuple[float, float]],
    cfg: NSGA2Config = NSGA2Config(),
) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    bounds_arr = np.asarray(bounds, dtype=float)
    d = len(bounds)

    def sample(n: int) -> np.ndarray:
        X = np.empty((n, d), dtype=float)
        for j, (lo, hi) in enumerate(bounds):
            X[:, j] = rng.uniform(lo, hi, size=n)
        return X

    pop = sample(cfg.pop_size)
    F = np.vstack([objective(x) for x in pop])

    for _ in range(cfg.generations):
        # Selection (binary tournament by rank + crowding)
        fronts = fast_non_dominated_sort(F)
        rank = np.empty(cfg.pop_size, dtype=int)
        crowd = np.zeros(cfg.pop_size, dtype=float)
        for r, idxs in enumerate(fronts):
            for i in idxs:
                rank[i] = r
            cd = crowding_distance(F, idxs)
            for k, i in enumerate(idxs):
                crowd[i] = cd[k]

        def tournament():
            i, j = rng.integers(0, cfg.pop_size, size=2)
            if rank[i] < rank[j]:
                return i
            if rank[j] < rank[i]:
                return j
            return i if crowd[i] > crowd[j] else j

        parents = np.array([pop[tournament()] for _ in range(cfg.pop_size)], dtype=float)

        # Variation
        children = []
        i = 0
        while i < cfg.pop_size:
            p1 = parents[i]
            p2 = parents[(i + 1) % cfg.pop_size]
            if rng.random() < cfg.crossover_prob:
                c1, c2 = sbx_crossover(rng, p1, p2, bounds_arr, cfg.eta_c)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = poly_mutation(rng, c1, bounds_arr, cfg.eta_m, cfg.mutation_prob)
            c2 = poly_mutation(rng, c2, bounds_arr, cfg.eta_m, cfg.mutation_prob)
            children.append(c1); children.append(c2)
            i += 2
        children = np.asarray(children[: cfg.pop_size], dtype=float)

        pop2 = np.vstack([pop, children])
        F2 = np.vstack([F, np.vstack([objective(x) for x in children])])

        # Elitist survivor selection
        fronts2 = fast_non_dominated_sort(F2)
        new_pop, new_F = [], []
        for idxs in fronts2:
            if len(new_pop) + len(idxs) <= cfg.pop_size:
                new_pop.extend(pop2[idxs])
                new_F.extend(F2[idxs])
            else:
                # pick by crowding distance
                cd = crowding_distance(F2, idxs)
                order = np.argsort(-cd)  # descending
                need = cfg.pop_size - len(new_pop)
                chosen = [idxs[o] for o in order[:need]]
                new_pop.extend(pop2[chosen])
                new_F.extend(F2[chosen])
                break
        pop = np.asarray(new_pop, dtype=float)
        F = np.asarray(new_F, dtype=float)

    fronts = fast_non_dominated_sort(F)
    pareto_idxs = fronts[0]
    return {
        "pop": pop,
        "F": F,
        "pareto_X": pop[pareto_idxs],
        "pareto_F": F[pareto_idxs],
    }
