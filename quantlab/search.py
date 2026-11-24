"""Budgeted, reproducible random and genetic parameter search.

Only uncached objective calls consume ``budget``. The history contains every
candidate request, including duplicate candidates served by the cache. Fully
fixed bounds have a single possible candidate and therefore require one call.
"""

from collections.abc import Mapping
import math
import sys

import numpy as np


_WORST_SCORE = -sys.float_info.max
_POPULATION_SIZE = 32


def optimize(objective, bounds, method="ga", budget=256, seed=0):
    """Maximize ``objective(params)`` within finite, inclusive ``bounds``.

    ``bounds`` preserves parameter order and maps names to ``(lower, upper)``.
    Both methods spend exactly ``budget`` unique evaluations for ordinary
    continuous search spaces. If the floating-point search space is exhausted,
    fewer evaluations are returned instead of retrying duplicates indefinitely.
    Nonfinite objective values are replaced with the lowest finite float.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")
    if method not in ("ga", "random"):
        raise ValueError("method must be 'ga' or 'random'")
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise TypeError("budget must be an integer")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if not isinstance(bounds, Mapping) or not bounds:
        raise ValueError("bounds must be a nonempty mapping")

    names = list(bounds)
    if not all(isinstance(name, str) for name in names):
        raise TypeError("parameter names must be strings")
    endpoints = []
    for name in names:
        pair = bounds[name]
        if not isinstance(pair, (tuple, list, np.ndarray)) or len(pair) != 2:
            raise ValueError(f"bounds for {name!r} must contain two endpoints")
        lower, upper = float(pair[0]), float(pair[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError(f"bounds for {name!r} must be finite and ordered")
        if not math.isfinite(upper - lower):
            raise ValueError(f"bounds for {name!r} have an overflowing width")
        endpoints.append((lower, upper))

    lower = np.array([pair[0] for pair in endpoints], dtype=float)
    upper = np.array([pair[1] for pair in endpoints], dtype=float)
    width = upper - lower
    rng = np.random.default_rng(seed)
    cache = {}
    history = []
    best_params = None
    best_score = None
    cache_hits = 0
    duplicate_streak = 0
    # A finite-float interval may contain fewer distinct points than the budget.
    # The guard also makes highly degenerate searches terminate predictably.
    collision_limit = max(1024, int(budget) * 100)

    def sample():
        return rng.uniform(lower, upper)

    def evaluate(candidate):
        nonlocal best_params, best_score, cache_hits, duplicate_streak
        vector = np.clip(np.asarray(candidate, dtype=float), lower, upper)
        key = tuple(float(value) for value in vector)
        params = dict(zip(names, key))
        cache_hit = key in cache
        if cache_hit:
            score = cache[key]
            cache_hits += 1
            duplicate_streak += 1
        else:
            # Give the objective a copy so mutation cannot alter the audit log.
            score = float(objective(params.copy()))
            if not math.isfinite(score):
                score = _WORST_SCORE
            cache[key] = score
            duplicate_streak = 0
            if best_score is None or score > best_score:
                best_params, best_score = params.copy(), score
        history.append({"params": params, "score": score, "cache_hit": cache_hit})
        return vector, score

    fixed_space = not bool(np.any(width > 0))
    if budget and fixed_space:
        evaluate(lower)
    elif method == "random":
        while len(cache) < budget and duplicate_streak < collision_limit:
            evaluate(sample())
    elif budget:
        population = []
        scores = []
        while len(population) < min(_POPULATION_SIZE, budget):
            before = len(cache)
            candidate, score = evaluate(sample())
            if len(cache) > before:
                population.append(candidate)
                scores.append(score)
            if duplicate_streak >= collision_limit:
                break

        def tournament():
            contestants = rng.choice(len(population), size=min(3, len(population)), replace=False)
            winner = max(contestants, key=lambda index: scores[int(index)])
            return population[int(winner)]

        while len(cache) < budget and duplicate_streak < collision_limit:
            ranked = sorted(range(len(population)), key=lambda index: scores[index], reverse=True)
            elites = ranked[:min(2, len(ranked))]
            next_population = [population[index].copy() for index in elites]
            next_scores = [scores[index] for index in elites]
            while len(next_population) < _POPULATION_SIZE and len(cache) < budget:
                if duplicate_streak >= collision_limit:
                    break
                # Random immigrants prevent cached clones from stalling search.
                if duplicate_streak >= _POPULATION_SIZE:
                    child = sample()
                else:
                    parent_a, parent_b = tournament(), tournament()
                    child = np.where(rng.random(len(names)) < 0.5, parent_a, parent_b)
                    mutation_mask = rng.random(len(names)) < 0.2
                    child = child + mutation_mask * rng.normal(0.0, 0.1, len(names)) * width
                child, score = evaluate(child)
                next_population.append(child)
                next_scores.append(score)
            population, scores = next_population, next_scores

    return {
        "best_params": best_params,
        "best_score": best_score,
        "evaluations": len(cache),
        "requests": len(history),
        "cache_hits": cache_hits,
        "history": history,
    }
