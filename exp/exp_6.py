"""Experiment 6: Proposition 2 -- Value-at-Risk violates Fair-Coin
Compensation (FC).

Reproduces the paper's explicit counterexample: X = 0 (a.s.) and Y a
two-point {0, 1} law with P(Y=0) = p^2, P(Y=1) = 1 - p^2. For every
p in (0, 1), VaR_p(X) = 0 and VaR_p(Y) = 1, while the fair-coin mixture
W_1(a) = B(X+a) + (1-B)(Y-a) satisfies VaR_p(W_1(0)) = 0 because
P(W_1(0) <= 0) = (1+p^2)/2 > p for every p in (0, 1) (since
(1+p^2)/2 - p = (1-p)^2/2 > 0). Hence
inf_a VaR_p(B(X+a)+(1-B)(Y-a)) <= 0 < 1/2 = (VaR_p(X)+VaR_p(Y))/2,
so Axiom (FC) fails for VaR_p for every p in (0, 1). Run as
``python -m exp.exp_6``.
"""

import json

import numpy as np
import pandas as pd

from data.generate_synthetic import make_rng
from src import config
from src.risk_measures import exact_var_discrete, mix_discrete_fair_coin, value_at_risk

RESULTS_DIR = "results"

A_GRID = np.linspace(-0.5, 0.5, 21)


def var_fc_violation(p: float) -> dict:
    """Exact, closed-form check of the Proposition 2 counterexample for a
    given VaR level p in (0, 1)."""
    x_values = np.array([0.0])
    x_probs = np.array([1.0])
    y_values = np.array([0.0, 1.0])
    y_probs = np.array([p**2, 1.0 - p**2])

    var_x = exact_var_discrete(x_values, x_probs, p)
    var_y = exact_var_discrete(y_values, y_probs, p)

    w_values, w_probs = mix_discrete_fair_coin(x_values, x_probs, y_values, y_probs, a=0.0)
    var_w_at_zero = exact_var_discrete(w_values, w_probs, p)

    target_average = 0.5 * (var_x + var_y)
    p_w_leq_0 = float(w_probs[w_values <= 1e-9].sum())
    key_inequality_holds = bool(p_w_leq_0 > p)
    fc_violation = bool(var_w_at_zero < target_average - 1e-9)

    # var_w_at_zero is VaR_p at the single point a=0 of the transfer grid, so
    # it is only an upper bound on inf_a VaR_p(W_1(a)); per the paper's proof
    # this single point already demonstrates strict violation of (FC), since
    # VaR_p(X)=0 is already the infimum value of VaR_p over the real line, so
    # the true infimum cannot exceed this upper bound and the strict
    # inequality below suffices.
    inf_premium = 2.0 * var_w_at_zero - var_x - var_y

    return {
        "p": p,
        "p_squared": p**2,
        "var_X": var_x,
        "var_Y": var_y,
        "prob_W_leq_0": p_w_leq_0,
        "prob_threshold_p": p,
        "key_inequality_holds": key_inequality_holds,
        "var_W_at_a0": var_w_at_zero,
        "target_average": target_average,
        "fc_violation": fc_violation,
        "inf_premium": inf_premium,
    }


def premium_vs_a(p: float) -> pd.DataFrame:
    """Scans a grid of transfer amounts a around 0 and reports the premium
    2*VaR_p(W_1(a)) - VaR_p(X) - VaR_p(Y), confirming its minimum over the
    grid stays at or below the a=0 value computed in `var_fc_violation`."""
    x_values = np.array([0.0])
    x_probs = np.array([1.0])
    y_values = np.array([0.0, 1.0])
    y_probs = np.array([p**2, 1.0 - p**2])
    var_x = exact_var_discrete(x_values, x_probs, p)
    var_y = exact_var_discrete(y_values, y_probs, p)

    rows = []
    for a in A_GRID:
        w_values, w_probs = mix_discrete_fair_coin(x_values, x_probs, y_values, y_probs, a=float(a))
        var_w = exact_var_discrete(w_values, w_probs, p)
        premium = 2.0 * var_w - var_x - var_y
        rows.append({"p": p, "a": float(a), "var_W": var_w, "premium": premium})
    return pd.DataFrame(rows)


def monte_carlo_cross_check(p: float, n: int, rng: np.random.Generator) -> dict:
    """Validates the exact discrete VaR_p(W_1(0)) against a Monte Carlo
    estimate of VaR_p obtained by resampling the exact law of W_1(0)."""
    x_values = np.array([0.0])
    x_probs = np.array([1.0])
    y_values = np.array([0.0, 1.0])
    y_probs = np.array([p**2, 1.0 - p**2])
    w_values, w_probs = mix_discrete_fair_coin(x_values, x_probs, y_values, y_probs, a=0.0)
    exact_var = exact_var_discrete(w_values, w_probs, p)

    samples = rng.choice(w_values, size=n, p=w_probs)
    mc_var = value_at_risk(samples, p)

    return {
        "p": p,
        "exact_var": exact_var,
        "monte_carlo_var": mc_var,
        "match": bool(abs(exact_var - mc_var) < 1e-2),
    }


def main() -> None:
    rng = make_rng(config.RANDOM_SEED)

    per_p_results = [var_fc_violation(p) for p in config.EXP6_P_VALUES]
    violation_df = pd.DataFrame(per_p_results)

    premium_df = pd.concat([premium_vs_a(p) for p in config.EXP6_P_VALUES], ignore_index=True)

    mc_check = monte_carlo_cross_check(0.9, config.EXP6_SAMPLE_SIZE, rng)

    violation_df.to_csv(f"{RESULTS_DIR}/exp6_var_violates_fc.csv", index=False)
    premium_df.to_csv(f"{RESULTS_DIR}/exp6_premium_vs_a.csv", index=False)

    summary = {
        "fc_violation_holds_for_all_p": bool(violation_df["fc_violation"].all()),
        "key_inequality_holds_for_all_p": bool(violation_df["key_inequality_holds"].all()),
        "per_p": {str(r["p"]): r for r in per_p_results},
        "monte_carlo_cross_check": mc_check,
    }
    with open(f"{RESULTS_DIR}/exp6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("exp_6 summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
