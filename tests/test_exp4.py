"""Tests for exp/exp_4.py: Proposition 4 / Corollary 1 -- Expected Shortfall,
Positive Homogeneity and Prudence."""

import pytest

from data.generate_synthetic import make_rng, sample_normal
from exp.exp_4 import mixture_rho
from src.axioms import check_positive_homogeneity, fatou_prudence_sequence
from src.risk_measures import expectation, expected_shortfall


@pytest.fixture
def rng():
    return make_rng(123)


@pytest.fixture
def X(rng):
    return sample_normal(5000, 0.0, 1.0, rng)


def test_mixture_lambda_zero_is_pure_es(X):
    p = 0.9
    assert mixture_rho(X, 0.0, p) == expected_shortfall(X, p)


def test_mixture_lambda_one_is_pure_expectation(X):
    p = 0.9
    assert mixture_rho(X, 1.0, p) == pytest.approx(expectation(X), abs=1e-9)


def test_positive_homogeneity_holds_for_es(X):
    p = 0.9
    result = check_positive_homogeneity(lambda s: expected_shortfall(s, p), X, [2.0], tol=1e-2)
    assert result[2.0]["holds"]
    assert result[2.0]["rho_tX"] == pytest.approx(result[2.0]["t_rho_X"], rel=1e-2)


def test_fatou_sequence_violates_prudence_for_expectation_but_not_es():
    rng = make_rng(7)
    sequence, limit = fatou_prudence_sequence([10, 1000], 200_000, rng)

    e_limit = expectation(limit)
    e_large_n = expectation(sequence[1000])
    # liminf E[X_n] = -1 < 0 = E[X]: Prudence is violated by plain expectation.
    assert e_limit == pytest.approx(0.0, abs=1e-9)
    assert e_large_n < e_limit - 0.5
    assert e_large_n == pytest.approx(-1.0, abs=0.2)

    # n=1000 > 1/(1-p)=10 for p=0.9, so ES_p no longer sees the rare spike.
    p = 0.9
    es_limit = expected_shortfall(limit, p)
    es_large_n = expected_shortfall(sequence[1000], p)
    assert es_limit == pytest.approx(0.0, abs=1e-9)
    assert es_large_n == pytest.approx(es_limit, abs=1e-2)
