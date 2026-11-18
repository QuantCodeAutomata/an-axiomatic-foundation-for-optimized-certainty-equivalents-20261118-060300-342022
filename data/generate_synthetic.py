"""Synthetic random-variable generators for the experiments in exp/.

The paper works on an atomless probability space (Assumption 1) and verifies
its axioms with constructed random variables (normal, uniform, Bernoulli,
discrete). No market data is used anywhere in this replication: every
experiment numerically illustrates a theorem or proposition using samples
drawn from explicitly specified laws.
"""

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """A NumPy random generator seeded for reproducibility."""
    return np.random.default_rng(seed)


def sample_normal(n: int, mean: float, std: float, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from Normal(mean, std^2)."""
    if std < 0:
        raise ValueError("std must be non-negative")
    return rng.normal(loc=mean, scale=std, size=n)


def sample_uniform(n: int, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from Uniform(low, high)."""
    if low > high:
        raise ValueError("low must not exceed high")
    return rng.uniform(low=low, high=high, size=n)


def sample_bernoulli(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """n iid draws from a coin with probability p, values in {0, 1}."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    return (rng.random(n) < p).astype(float)


def sample_discrete(n: int, values: np.ndarray, probs: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """n iid draws from a finite discrete law on `values` with probabilities `probs`."""
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if values.shape != probs.shape:
        raise ValueError("values and probs must have the same shape")
    if not np.isclose(probs.sum(), 1.0, atol=1e-8):
        raise ValueError("probs must sum to 1")
    return rng.choice(values, size=n, p=probs)


def fair_coin(n: int, rng: np.random.Generator) -> np.ndarray:
    """n iid draws of a fair coin (a coin with probability 1/2), values in {0, 1}."""
    return sample_bernoulli(n, 0.5, rng)
