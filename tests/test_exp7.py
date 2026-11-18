"""Tests for exp/exp_7.py: fairness premium of ES and its average (Example 4)."""

import numpy as np
import pytest

from exp.exp_7 import es_fairness_premium, xi_premium_curve
from src import config


def test_es_1_2_premium_reaches_zero_at_half():
    result = es_fairness_premium(0.5, 0.5)
    assert abs(result["premium"]) < 1e-9


def test_es_3_4_premium_reaches_zero_at_one():
    result = es_fairness_premium(0.75, 1.0)
    assert abs(result["premium"]) < 1e-9


def test_es_1_2_premium_positive_away_from_minimizer():
    assert es_fairness_premium(0.5, 0.0)["premium"] > 0.0
    assert es_fairness_premium(0.5, 1.5)["premium"] > 0.0


def test_xi_premium_minimum_is_one_sixth():
    a_grid = np.linspace(-1.0, 2.5, 3501)
    curve = xi_premium_curve(a_grid)
    assert np.min(curve) == pytest.approx(1.0 / 6.0, abs=1e-3)
    # xi violates Fair-Coin Compensation: the premium never reaches zero.
    assert np.all(curve > -1e-9)


def test_es_y_values_match_closed_form():
    es_1_2 = es_fairness_premium(0.5, 0.0)["es_Y"]
    es_3_4 = es_fairness_premium(0.75, 0.0)["es_Y"]
    assert es_1_2 == pytest.approx(5.0 / 3.0)
    assert es_3_4 == pytest.approx(2.0)
