"""Tests for exp/exp_3.py: Mean-Variance criterion and Equal-Mean Randomization."""

import numpy as np
import pytest

from data.generate_synthetic import make_rng, sample_normal
from exp.exp_3 import (
    check_oce_matches_closed_form,
    run_monotonicity_violation,
)
from src.axioms import check_equal_mean_randomization
from src.risk_measures import mean_variance
from functools import partial


@pytest.fixture
def rng():
    return make_rng(123)


def test_oce_matches_closed_form_quadratic_loss(rng):
    x = sample_normal(100_000, 0.5, 2.0, rng)
    for kappa in [0.0, 0.1, 1.0, 5.0]:
        result = check_oce_matches_closed_form(x, kappa, tol=5e-3)
        assert result["holds"], result


def test_equal_mean_randomization_holds_with_equality(rng):
    mu = 0.0
    x = sample_normal(100_000, mu, 1.0, rng)
    y = sample_normal(100_000, mu, 4.0, rng)
    rho = partial(mean_variance, kappa=1.0)
    result = check_equal_mean_randomization(rho, x, y, rng, tol=0.05)
    assert result["holds_inequality"]
    assert result["holds_equality"]
    # Var(BX + (1-B)Y) = (Var(X) + Var(Y)) / 2 when E[X] = E[Y] (paper Sec. 4.1).
    var_mix_expected = 0.5 * (np.var(x) + np.var(y))
    assert abs((result["rho_mixture"] - mu) - var_mix_expected) < 0.1


def test_monotonicity_violated_for_positive_kappa():
    result = run_monotonicity_violation()
    assert result["monotonicity_violated"]
    assert result["rho_X"] > result["rho_Y"]
    assert abs(result["rho_X"] - result["expected_rho_X"]) < 1e-2
    assert result["rho_Y"] == pytest.approx(4.0)
