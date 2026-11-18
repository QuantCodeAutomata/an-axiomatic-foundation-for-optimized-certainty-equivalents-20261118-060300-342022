"""Unit tests for src/axioms.py: axiom-checking utilities on small fixtures."""

import numpy as np
import pytest

from src.axioms import (
    check_equal_mean_randomization,
    check_fair_coin_compensation,
    check_fair_coin_independence,
    check_monotonicity,
    check_normalization,
    check_positive_homogeneity,
    check_prudence,
    check_sra,
    convex_order_holds,
    fair_coin,
    fair_coin_mixture,
)
from src.risk_measures import entropic_risk_measure, expectation, mean_variance


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_fair_coin_is_half_half(rng):
    b = fair_coin(200_000, rng)
    assert set(np.unique(b)) <= {0.0, 1.0}
    assert abs(np.mean(b) - 0.5) < 1e-2


def test_check_normalization_holds_for_expectation(rng):
    result = check_normalization(expectation, [0.0, 1.0, -1.0, 5.0])
    assert all(r["holds"] for r in result.values())


def test_convex_order_constant_le_bernoulli():
    """X = E[Y] (constant) is smaller in convex order than Y ~ Bernoulli(p): E[phi(X)] <= E[phi(Y)].

    X is set to the *sample* mean of Y (not the theoretical p) so that the
    necessary condition E[X] = E[Y] holds exactly on the finite sample.
    """
    rng = np.random.default_rng(3)
    p = 0.3
    y = (rng.random(200_000) < p).astype(float)
    x = np.full_like(y, np.mean(y))
    test_funcs = [lambda z: z**2, lambda z: np.maximum(z, 0.0), np.exp]
    assert convex_order_holds(x, y, test_funcs, tol=1e-3)


def test_check_sra_holds_for_expectation_under_convex_order(rng):
    p = 0.4
    y = (rng.random(100_000) < p).astype(float)
    x = np.full_like(y, np.mean(y))
    result = check_sra(expectation, x, y, tol=1e-9)
    assert result["holds"]


def test_fair_coin_mixture_has_expected_total(rng):
    x = np.full(1000, 1.0)
    y = np.full(1000, 3.0)
    w = fair_coin_mixture(x, y, a=0.5, rng=rng)
    # Either x+a=1.5 or y-a=2.5 in every state.
    assert set(np.round(np.unique(w), 6)) <= {1.5, 2.5}


def test_check_fair_coin_compensation_holds_for_expectation(rng):
    x = rng.normal(0.0, 1.0, size=20_000)
    y = rng.normal(1.0, 2.0, size=20_000)
    a_grid = np.linspace(-5, 5, 101)
    result = check_fair_coin_compensation(expectation, x, y, a_grid, rng, tol=5e-2)
    assert result["holds"]
    assert result["target_average"] == pytest.approx(0.5 * (np.mean(x) + np.mean(y)), abs=1e-2)


def test_check_monotonicity_holds_for_expectation(rng):
    x = rng.normal(size=5000)
    y = x + 1.0  # surely larger
    result = check_monotonicity(expectation, x, y)
    assert result["holds"]


def test_check_equal_mean_randomization_equality_for_mean_variance(rng):
    mu = 0.0
    x = rng.normal(mu, 1.0, size=100_000)
    y = rng.normal(mu, 3.0, size=100_000)
    rho = lambda s: mean_variance(s, kappa=0.5)
    result = check_equal_mean_randomization(rho, x, y, rng, tol=5e-2)
    assert result["holds_inequality"]
    assert result["holds_equality"]


def test_check_fair_coin_independence_holds_for_entropic(rng):
    x = rng.normal(0.0, 1.0, size=50_000)
    y = rng.normal(0.5, 1.0, size=50_000)
    z = rng.normal(0.2, 1.5, size=50_000)
    rho = lambda s: entropic_risk_measure(s, gamma=0.5)
    result = check_fair_coin_independence(rho, x, y, z, rng)
    assert result["holds"]


def test_check_positive_homogeneity_holds_for_expectation(rng):
    x = rng.normal(1.0, 2.0, size=10_000)
    result = check_positive_homogeneity(expectation, x, [0.5, 2.0, 10.0])
    assert all(r["holds"] for r in result.values())


def test_check_prudence_holds_for_expectation(rng):
    x = rng.normal(size=20_000)

    def perturbation(base, n, gen):
        z = gen.uniform(-3.0, 3.0, size=base.shape)
        return base + z / n

    result = check_prudence(expectation, x, perturbation, [2, 10, 100, 1000], rng)
    assert result["holds"]
