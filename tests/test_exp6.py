"""Tests for exp/exp_6.py: Proposition 2 -- VaR_p violates Fair-Coin
Compensation (FC)."""

import pytest

from exp.exp_6 import var_fc_violation
from src.config import EXP6_P_VALUES


def test_key_inequality_holds_for_p_09():
    assert var_fc_violation(0.9)["key_inequality_holds"] is True


def test_fc_violation_holds_for_p_09():
    assert var_fc_violation(0.9)["fc_violation"] is True


def test_var_values_for_p_09():
    result = var_fc_violation(0.9)
    assert result["var_X"] == 0.0
    assert result["var_Y"] == 1.0


@pytest.mark.parametrize("p", EXP6_P_VALUES)
def test_fc_violation_and_key_inequality_for_all_p(p):
    result = var_fc_violation(p)
    assert result["fc_violation"] is True
    assert result["key_inequality_holds"] is True


@pytest.mark.parametrize("p", [0.01, 0.1, 0.3, 0.5, 0.75, 0.9, 0.99])
def test_algebraic_identity(p):
    assert abs(((1.0 + p**2) / 2.0 - p) - ((1.0 - p)**2 / 2.0)) < 1e-9
