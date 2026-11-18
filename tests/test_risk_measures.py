"""Unit tests for src/risk_measures.py."""

import numpy as np
import pytest

from src.risk_measures import (
    chi_risk_measure,
    double_expectation,
    entropic_risk_measure,
    exact_es_discrete,
    exact_var_discrete,
    expectation,
    expected_shortfall,
    mean_variance,
    value_at_risk,
    xi_risk_measure,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_expectation_matches_numpy_mean(rng):
    x = rng.normal(size=10_000)
    assert expectation(x) == pytest.approx(np.mean(x))


def test_value_at_risk_is_between_min_and_max(rng):
    x = rng.normal(size=10_000)
    var = value_at_risk(x, 0.9)
    assert x.min() <= var <= x.max()


def test_expected_shortfall_exceeds_value_at_risk(rng):
    """ES_p(X) >= VaR_p(X) always, since ES averages the upper tail beyond VaR."""
    x = rng.normal(size=20_000)
    for p in [0.5, 0.9, 0.99]:
        assert expected_shortfall(x, p) >= value_at_risk(x, p) - 1e-6


def test_expected_shortfall_monotone_in_level(rng):
    """ES_p(X) is nondecreasing in p (a stricter level averages a smaller, more extreme tail)."""
    x = rng.normal(size=50_000)
    levels = [0.5, 0.75, 0.9, 0.99]
    es_values = [expected_shortfall(x, p) for p in levels]
    assert all(a <= b + 1e-6 for a, b in zip(es_values, es_values[1:]))


def test_entropic_risk_measure_limit_is_expectation(rng):
    """ER_0 = E[X] by definition (paper Proposition 5)."""
    x = rng.normal(1.0, 2.0, size=10_000)
    assert entropic_risk_measure(x, 0.0) == pytest.approx(np.mean(x))


def test_entropic_risk_measure_increasing_in_gamma(rng):
    """A more risk-averse (larger gamma) entropic measure gives a higher risk evaluation."""
    x = rng.normal(size=10_000)
    gammas = [0.01, 0.1, 1.0, 5.0]
    values = [entropic_risk_measure(x, g) for g in gammas]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))


def test_mean_variance_reduces_to_expectation_at_kappa_zero(rng):
    x = rng.normal(2.0, 1.0, size=10_000)
    assert mean_variance(x, kappa=0.0) == pytest.approx(np.mean(x))


def test_mean_variance_exceeds_expectation_for_positive_kappa(rng):
    x = rng.normal(size=10_000)
    assert mean_variance(x, kappa=1.0) >= expectation(x)


def test_chi_matches_three_quarter_bound_on_discrete_example():
    """chi(W) <= 3/4 for W = 2*1_{E2}, P(E2)=1/4 (paper Appendix A, proof of Prop. 3)."""
    rng = np.random.default_rng(2)
    n = 400_000
    e1 = rng.random(n) < 0.5
    e2 = e1 & (rng.random(n) < 0.5)
    w = 2.0 * e2.astype(float)
    assert chi_risk_measure(w) <= 0.75 + 1e-2


def test_double_expectation_is_twice_expectation(rng):
    x = rng.normal(size=1000)
    assert double_expectation(x) == pytest.approx(2 * np.mean(x))


def test_xi_is_average_of_two_expected_shortfalls(rng):
    x = rng.normal(size=50_000)
    expected = 0.5 * expected_shortfall(x, 0.5) + 0.5 * expected_shortfall(x, 0.75)
    assert xi_risk_measure(x) == pytest.approx(expected)


def test_exact_var_discrete_uniform_support():
    """VaR_p for Y ~ Uniform{0,1,2}: VaR_{1/2}=1, VaR_{3/4}=2 (paper Example 4)."""
    values = np.array([0.0, 1.0, 2.0])
    probs = np.array([1 / 3, 1 / 3, 1 / 3])
    assert exact_var_discrete(values, probs, 0.5) == pytest.approx(1.0)
    assert exact_var_discrete(values, probs, 0.75) == pytest.approx(2.0)


def test_exact_es_discrete_matches_paper_example_4():
    """ES_{1/2}(Y) = 5/3 and ES_{3/4}(Y) = 2 for Y ~ Uniform{0,1,2} (paper Example 4)."""
    values = np.array([0.0, 1.0, 2.0])
    probs = np.array([1 / 3, 1 / 3, 1 / 3])
    assert exact_es_discrete(values, probs, 0.5) == pytest.approx(5 / 3, abs=1e-9)
    assert exact_es_discrete(values, probs, 0.75) == pytest.approx(2.0, abs=1e-9)


def test_exact_es_discrete_rejects_invalid_level():
    values = np.array([0.0, 1.0])
    probs = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        exact_es_discrete(values, probs, 1.5)
