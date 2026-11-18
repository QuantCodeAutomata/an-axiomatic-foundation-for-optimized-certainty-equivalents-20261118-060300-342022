"""Unit tests for src/oce.py: the OCE minimization and its loss functions."""

import numpy as np
import pytest

from src.oce import (
    loss_chi,
    loss_entropic,
    loss_expectation,
    loss_expected_shortfall,
    loss_quadratic,
    oce,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_loss_functions_vanish_at_zero():
    """Every loss function must satisfy ell(0) = 0 (paper Sec. 2)."""
    assert loss_expectation(np.array([0.0]))[0] == 0.0
    assert loss_quadratic(np.array([0.0]), kappa=2.0)[0] == 0.0
    assert loss_expected_shortfall(np.array([0.0]), p=0.9)[0] == 0.0
    assert loss_entropic(np.array([0.0]), gamma=1.0)[0] == 0.0
    assert loss_chi(np.array([0.0]))[0] == 0.0


def test_loss_functions_dominate_identity():
    """Every loss function used in this paper satisfies ell(x) >= x (paper Sec. 2)."""
    x = np.linspace(-5, 5, 101)
    for loss in [
        loss_expectation,
        lambda z: loss_quadratic(z, kappa=0.3),
        lambda z: loss_expected_shortfall(z, p=0.8),
        lambda z: loss_entropic(z, gamma=0.5),
        loss_chi,
    ]:
        assert np.all(loss(x) >= x - 1e-9)


def test_oce_expectation_matches_sample_mean(rng):
    """For ell(x)=x, rho_ell(X) = E[X] exactly (the infimum is attained at every m)."""
    x = rng.normal(2.0, 3.0, size=50_000)
    assert oce(x, loss_expectation) == pytest.approx(np.mean(x), abs=1e-2)


def test_oce_quadratic_matches_closed_form_mean_variance(rng):
    """For ell(x)=x+kappa*x^2, rho_ell(X) = E[X] + kappa*Var(X) (paper Example 1)."""
    x = rng.normal(1.0, 2.0, size=50_000)
    kappa = 0.7
    closed_form = np.mean(x) + kappa * np.var(x)
    assert oce(x, lambda z: loss_quadratic(z, kappa)) == pytest.approx(closed_form, rel=1e-2)


def test_oce_expected_shortfall_matches_quantile_formula(rng):
    """For ell(x)=x_+/(1-p), rho_ell(X) equals the Rockafellar-Uryasev ES_p (paper Example 2)."""
    x = rng.normal(0.0, 1.0, size=100_000)
    p = 0.9
    var = np.quantile(x, p)
    closed_form = np.mean(x[x >= var])
    assert oce(x, lambda z: loss_expected_shortfall(z, p)) == pytest.approx(closed_form, rel=2e-2)


def test_oce_entropic_matches_log_sum_exp(rng):
    """For ell(x)=(e^{gamma x}-1)/gamma, rho_ell(X) = (1/gamma) log E[exp(gamma X)] (Eq. 3)."""
    x = rng.normal(0.0, 1.0, size=50_000)
    gamma = 0.5
    closed_form = np.log(np.mean(np.exp(gamma * x))) / gamma
    assert oce(x, lambda z: loss_entropic(z, gamma)) == pytest.approx(closed_form, rel=1e-2)


def test_oce_rejects_empty_sample():
    with pytest.raises(ValueError):
        oce(np.array([]), loss_expectation)


def test_loss_quadratic_rejects_negative_kappa():
    with pytest.raises(ValueError):
        loss_quadratic(np.array([1.0]), kappa=-1.0)


def test_loss_expected_shortfall_rejects_invalid_p():
    with pytest.raises(ValueError):
        loss_expected_shortfall(np.array([1.0]), p=1.5)


def test_loss_entropic_rejects_nonpositive_gamma():
    with pytest.raises(ValueError):
        loss_entropic(np.array([1.0]), gamma=0.0)
