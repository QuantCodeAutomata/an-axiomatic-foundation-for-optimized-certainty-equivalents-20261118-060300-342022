"""Tests for exp/exp_5.py: Entropic Risk Measure and Fair-Coin Independence."""

import numpy as np
import pytest

from data.generate_synthetic import make_rng, sample_normal
from exp.exp_5 import er_gamma_mixture_closed
from src.axioms import check_fair_coin_independence, fair_coin
from src.risk_measures import entropic_risk_measure, expectation


@pytest.fixture
def rng():
    return make_rng(7)


def test_closed_form_mixture_matches_monte_carlo(rng):
    gamma = 1.0
    n = 20_000
    x = sample_normal(n, 0.0, 1.0, rng)
    z = sample_normal(n, 0.2, 1.5, rng)
    rho_x = entropic_risk_measure(x, gamma)
    rho_z = entropic_risk_measure(z, gamma)

    b = fair_coin(n, rng)
    mix = b * x + (1.0 - b) * z
    rho_mix_mc = entropic_risk_measure(mix, gamma)
    rho_mix_closed = er_gamma_mixture_closed(rho_x, rho_z, gamma)

    rel_diff = abs(rho_mix_closed - rho_mix_mc) / abs(rho_mix_mc)
    assert rel_diff < 2e-2


def test_entropic_gamma_zero_is_expectation(rng):
    x = sample_normal(5_000, 0.3, 2.0, rng)
    assert entropic_risk_measure(x, 0.0) == expectation(x)


def test_fci_holds_for_entropic_risk_measure(rng):
    n = 20_000
    gamma = 1.0
    x = sample_normal(n, 0.0, 1.0, rng)
    y = sample_normal(n, 0.5, 1.0, rng)
    z = sample_normal(n, 0.2, 1.5, rng)
    result = check_fair_coin_independence(
        lambda s: entropic_risk_measure(s, gamma), x, y, z, rng
    )
    assert result["holds"]


def test_closed_form_mixture_strictly_increasing_in_rho_x():
    gamma = 1.0
    rho_z = 0.5
    rho_x_values = [-1.0, 0.0, 0.5, 1.0, 2.0]
    mixture_values = [er_gamma_mixture_closed(rx, rho_z, gamma) for rx in rho_x_values]
    diffs = np.diff(mixture_values)
    assert np.all(diffs > 0)
