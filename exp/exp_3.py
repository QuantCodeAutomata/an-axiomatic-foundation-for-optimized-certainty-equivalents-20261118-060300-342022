"""Experiment 3: Mean-Variance criterion and Equal-Mean Randomization (Theorem 3).

Reproduces the paper's Section 4.1 result that the Mean-Variance criterion
rho_kappa(X) = E[X] + kappa*Var(X) is an OCE (loss ell(x) = x + kappa*x^2,
Example 1) and satisfies the Equal-Mean Randomization (ER) property with
equality: for X, Y with E[X] = E[Y], mixing them with an independent fair
coin B gives rho(BX + (1-B)Y) = (rho(X) + rho(Y)) / 2 (paper Sec. 4.1,
Appendix A). It also reproduces the Monotonicity violation of Mean-Variance
(paper Sec. 4.1, Appendix A) that marks it as a non-standard OCE.
"""

import json
from functools import partial

import numpy as np
import pandas as pd

from data.generate_synthetic import (
    make_rng,
    sample_discrete,
    sample_normal,
    sample_uniform,
)
from src import config
from src.axioms import check_equal_mean_randomization, check_monotonicity
from src.oce import loss_quadratic, oce
from src.risk_measures import mean_variance


def check_oce_matches_closed_form(samples: np.ndarray, kappa: float, tol: float = 1e-3) -> dict:
    """Verifies that inf_m {m + E[ell(X-m)]} with ell(x) = x + kappa*x^2
    (paper Example 1) equals the closed-form E[X] + kappa*Var(X) (Sec. 4.1)."""
    oce_value = oce(samples, partial(loss_quadratic, kappa=kappa))
    closed_form = mean_variance(samples, kappa)
    return {
        "kappa": kappa,
        "oce_value": oce_value,
        "closed_form": closed_form,
        "abs_diff": abs(oce_value - closed_form),
        "holds": bool(abs(oce_value - closed_form) <= tol),
    }


def run_closed_form_checks(rng: np.random.Generator) -> pd.DataFrame:
    """Section 1 of the methodology: OCE minimization vs. closed form, for
    several kappa and a representative sample."""
    rows = []
    x = sample_normal(config.N_SAMPLES, 0.0, 1.5, rng)
    for kappa in config.EXP3_KAPPA_VALUES:
        rows.append(check_oce_matches_closed_form(x, kappa))
    return pd.DataFrame(rows)


def build_equal_mean_pairs(rng: np.random.Generator) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Pairs (X, Y) with E[X] = E[Y] = mu, from Normal, Uniform and discrete
    laws (distributions not prescribed by the paper; see config EXP3_*)."""
    mu = config.EXP3_EQUAL_MEAN
    pairs = []
    for std_x, std_y in config.EXP3_NORMAL_STDS:
        x = sample_normal(config.N_SAMPLES, mu, std_x, rng)
        y = sample_normal(config.N_SAMPLES, mu, std_y, rng)
        pairs.append((f"Normal(std={std_x} vs {std_y})", x, y))
    for half_x, half_y in config.EXP3_UNIFORM_HALF_WIDTHS:
        x = sample_uniform(config.N_SAMPLES, -half_x, half_x, rng)
        y = sample_uniform(config.N_SAMPLES, -half_y, half_y, rng)
        pairs.append((f"Uniform(half-width={half_x} vs {half_y})", x, y))
    x_discrete = sample_discrete(config.N_SAMPLES, config.EXP3_DISCRETE_X_VALUES,
                                  config.EXP3_DISCRETE_X_PROBS, rng)
    y_constant = np.full(config.N_SAMPLES, config.EXP3_DISCRETE_Y_CONSTANT)
    pairs.append(("Discrete{0,4} vs constant 2", x_discrete, y_constant))
    return pairs


def run_equal_mean_randomization(rng: np.random.Generator) -> pd.DataFrame:
    """Section 3-4 of the methodology: ER property, checked for equality
    (not just the weak inequality), for several kappa and distribution pairs."""
    rows = []
    pairs = build_equal_mean_pairs(rng)
    for kappa in config.EXP3_KAPPA_VALUES:
        rho = partial(mean_variance, kappa=kappa)
        # The ER identity rho(mixture) = average scales with kappa*Var(.), so the
        # Monte Carlo sampling-noise tolerance is scaled with kappa as well.
        tol = config.MC_PREMIUM_TOL * max(1.0, kappa)
        for name, x, y in pairs:
            result = check_equal_mean_randomization(rho, x, y, rng, tol=tol)
            var_x, var_y = float(np.var(x)), float(np.var(y))
            rows.append({
                "kappa": kappa,
                "pair": name,
                **result,
                "var_X": var_x,
                "var_Y": var_y,
                "variance_identity_rhs": 0.5 * (var_x + var_y),
            })
    return pd.DataFrame(rows)


def run_monotonicity_violation() -> dict:
    """Section 5 of the methodology: X ~ {0,4} equally likely satisfies X <= 4
    a.s., but rho_kappa(X) = 2 + 4*kappa > 4 = rho_kappa(4) for kappa > 0
    (paper Sec. 4.1, Appendix A), so Mean-Variance is not a standard OCE."""
    kappa = config.EXP3_MONOTONICITY_KAPPA
    rho = partial(mean_variance, kappa=kappa)
    x = config.EXP3_DISCRETE_X_VALUES  # exact two-point support {0, 4}, equal weights
    probs = config.EXP3_DISCRETE_X_PROBS
    samples_x = np.repeat(x, (probs * 1_000_000).astype(int))
    samples_y = np.full(samples_x.shape, 4.0)  # Y = 4 a.s., the a.s. upper bound of X
    result = check_monotonicity(rho, samples_x, samples_y)
    result["kappa"] = kappa
    result["expected_rho_X"] = 2.0 + 4.0 * kappa
    result["expected_rho_Y"] = 4.0
    result["monotonicity_violated"] = bool(not result["holds"])
    return result


def main() -> None:
    rng = make_rng(config.RANDOM_SEED + 3)

    closed_form_df = run_closed_form_checks(rng)
    closed_form_df.to_csv("results/exp3_closed_form.csv", index=False)

    er_df = run_equal_mean_randomization(rng)
    er_df.to_csv("results/exp3_equal_mean_randomization.csv", index=False)

    monotonicity = run_monotonicity_violation()

    summary = {
        "closed_form_all_match": bool(closed_form_df["holds"].all()),
        "er_inequality_all_hold": bool(er_df["holds_inequality"].all()),
        "er_equality_all_hold": bool(er_df["holds_equality"].all()),
        "monotonicity_violated_for_kappa_positive": monotonicity["monotonicity_violated"],
        "monotonicity_rho_X": monotonicity["rho_X"],
        "monotonicity_rho_Y": monotonicity["rho_Y"],
    }
    with open("results/exp3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("exp_3 summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
