"""Experiment 2: Proposition 3 -- independence of SRA, FC, N and M.

Each of the four risk measures below satisfies exactly three of the four
axioms (Strong Risk Aversion, Fair-Coin Compensation, Normalization,
Monotonicity) and violates the fourth, which the paper uses to show that no
one axiom is implied by the other three (Proposition 3, Sec. 3.1 and the
proof in Appendix A). Run as ``python -m exp.exp_2``.
"""

import json

import numpy as np
import pandas as pd

from data.generate_synthetic import make_rng, sample_bernoulli, sample_normal
from src import config
from src.axioms import (
    check_fair_coin_compensation,
    check_monotonicity,
    check_normalization,
    check_sra,
    convex_order_holds,
)
from src.risk_measures import chi_risk_measure, double_expectation, mean_variance, xi_risk_measure

RESULTS_DIR = "results"


def _exact_uniform_support(values: np.ndarray, probs: np.ndarray, n: int) -> np.ndarray:
    """An exact (non-Monte-Carlo) empirical sample of a finite discrete law,
    built by deterministic repetition so the resulting empirical quantiles
    and Expected-Shortfall values match the paper's closed-form numbers
    (Example 4) rather than approximating them with sampling noise."""
    counts = np.round(probs * n).astype(int)
    counts[-1] = n - counts[:-1].sum()
    return np.repeat(values, counts)


def _chi_counterexample(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """W = 2*1_{E2} and X = E[W|G], G = sigma(E1), E2 subset E1 (paper Appendix
    A, proof of Prop. 3). P(E1)=EXP2_CHI_P_E1, P(E2)=EXP2_CHI_P_E2.

    E[W|G] is computed exactly (not estimated): on E1, E[W|E1] = 2*P(E2|E1);
    off E1, W=0 so E[W|not E1]=0. With the paper's values 2*P(E2|E1) = 1, so
    X = 1_{E1} exactly -- used directly rather than re-estimated from the
    sample, avoiding Monte Carlo noise in the conditional expectation itself.
    """
    p_e1, p_e2 = config.EXP2_CHI_P_E1, config.EXP2_CHI_P_E2
    e1 = rng.random(n) < p_e1
    e2 = e1 & (rng.random(n) < p_e2 / p_e1)
    w = 2.0 * e2.astype(float)
    cond_exp_w_given_g = (2.0 * (p_e2 / p_e1)) * e1.astype(float)
    return cond_exp_w_given_g, w


def run_xi_violates_fc(rng: np.random.Generator) -> dict:
    """xi = (1/2) ES_{1/2} + (1/2) ES_{3/4} satisfies SRA, N and M, but
    violates Fair-Coin Compensation (paper Example 4 / Proposition 3)."""
    n = 300_000
    y = _exact_uniform_support(config.EXP2_XI_Y_VALUES, config.EXP2_XI_Y_PROBS, n)
    x_zero = np.zeros(n)

    fc_result = check_fair_coin_compensation(
        xi_risk_measure, x_zero, y, config.EXP2_XI_FC_TRANSFER_GRID, rng, tol=config.MC_PREMIUM_TOL
    )

    normalization = check_normalization(xi_risk_measure, config.EXP1_NORMALIZATION_CONSTANTS, tol=1e-3)
    normalization_holds = all(r["holds"] for r in normalization.values())

    p = config.EXP1_SRA_BERNOULLI_P[2]  # p=0.5, a representative level
    y_bernoulli = sample_bernoulli(config.N_SAMPLES, p, rng)
    x_const = np.full(config.N_SAMPLES, float(p))
    test_functions = [lambda z: z**2, lambda z: np.maximum(z, 0.0), lambda z: np.exp(z)]
    cx_holds = convex_order_holds(x_const, y_bernoulli, test_functions, tol=config.MC_PREMIUM_TOL)
    sra_result = check_sra(xi_risk_measure, x_const, y_bernoulli, tol=config.MC_PREMIUM_TOL)

    u = sample_normal(config.N_SAMPLES, 0.0, 1.0, rng)
    shift = np.abs(sample_normal(config.N_SAMPLES, 1.0, 0.2, rng))
    monotonicity_result = check_monotonicity(xi_risk_measure, u, u + shift, tol=config.MC_PREMIUM_TOL)

    return {
        "xi(Y)": fc_result["rho_Y"],
        "xi(X=0)": fc_result["rho_X"],
        "target_average": fc_result["target_average"],
        "inf_premium": fc_result["inf_premium"],
        "a_star": fc_result["a_star"],
        "fc_holds": fc_result["holds"],
        "normalization_holds": normalization_holds,
        "sra_convex_order_holds": cx_holds,
        "sra_holds": sra_result["holds"],
        "monotonicity_holds": monotonicity_result["holds"],
        "violates": "FC",
    }


def run_chi_violates_sra(rng: np.random.Generator) -> dict:
    """chi(X) = inf_m {m + E[(X-m)_+] + P(X>m)} satisfies FC, N and M, but
    violates Strong Risk Aversion (paper Appendix A, proof of Prop. 3)."""
    n = 400_000
    x_cond, w = _chi_counterexample(n, rng)
    test_functions = [lambda z: z**2, lambda z: np.maximum(z, 0.0), lambda z: np.exp(z)]
    cx_holds = convex_order_holds(x_cond, w, test_functions, tol=config.MC_PREMIUM_TOL)
    sra_result = check_sra(chi_risk_measure, x_cond, w, tol=1e-2)

    normalization = check_normalization(chi_risk_measure, config.EXP1_NORMALIZATION_CONSTANTS, tol=1e-2)
    normalization_holds = all(r["holds"] for r in normalization.values())

    x_fc = sample_normal(config.N_SAMPLES, 0.0, 1.0, rng)
    y_fc = sample_normal(config.N_SAMPLES, 1.0, 1.5, rng)
    fc_result = check_fair_coin_compensation(
        chi_risk_measure, x_fc, y_fc, config.EXP2_CHI_M_GRID, rng, tol=0.05
    )

    u = sample_normal(config.N_SAMPLES, 0.0, 1.0, rng)
    shift = np.abs(sample_normal(config.N_SAMPLES, 1.0, 0.2, rng))
    monotonicity_result = check_monotonicity(chi_risk_measure, u, u + shift, tol=1e-2)

    return {
        "chi(X=E[W|G])": sra_result["rho_X"],
        "chi(W)": sra_result["rho_Y"],
        "sra_convex_order_holds": cx_holds,
        "sra_holds": sra_result["holds"],
        "normalization_holds": normalization_holds,
        "fc_holds": fc_result["holds"],
        "fc_inf_premium": fc_result["inf_premium"],
        "monotonicity_holds": monotonicity_result["holds"],
        "violates": "SRA",
    }


def run_double_expectation_violates_n(rng: np.random.Generator) -> dict:
    """rho(X) = 2*E[X] satisfies SRA, FC and M, but violates Normalization
    (paper Proposition 3)."""
    normalization = check_normalization(double_expectation, config.EXP1_NORMALIZATION_CONSTANTS, tol=1e-6)
    normalization_holds = all(r["holds"] for r in normalization.values())
    nonzero_violation = all(
        (not r["holds"]) for c, r in normalization.items() if c != 0.0
    )

    p = config.EXP1_SRA_BERNOULLI_P[2]
    y_bernoulli = sample_bernoulli(config.N_SAMPLES, p, rng)
    x_const = np.full(config.N_SAMPLES, float(p))
    test_functions = [lambda z: z**2, lambda z: np.maximum(z, 0.0), lambda z: np.exp(z)]
    cx_holds = convex_order_holds(x_const, y_bernoulli, test_functions, tol=config.MC_PREMIUM_TOL)
    sra_result = check_sra(double_expectation, x_const, y_bernoulli, tol=config.MC_PREMIUM_TOL)

    (mean_x, std_x), (mean_y, std_y) = config.EXP2_2E_NORMAL_PARAMS
    x_fc = sample_normal(config.N_SAMPLES, mean_x, std_x, rng)
    y_fc = sample_normal(config.N_SAMPLES, mean_y, std_y, rng)
    fc_result = check_fair_coin_compensation(
        double_expectation, x_fc, y_fc, config.EXP1_FC_TRANSFER_GRID, rng, tol=config.MC_PREMIUM_TOL
    )

    u = sample_normal(config.N_SAMPLES, 0.0, 1.0, rng)
    shift = np.abs(sample_normal(config.N_SAMPLES, 1.0, 0.2, rng))
    monotonicity_result = check_monotonicity(double_expectation, u, u + shift, tol=config.MC_PREMIUM_TOL)

    return {
        "normalization_holds_at_zero": normalization[0.0]["holds"],
        "normalization_violated_elsewhere": nonzero_violation,
        "normalization_holds": normalization_holds,
        "sra_convex_order_holds": cx_holds,
        "sra_holds": sra_result["holds"],
        "fc_holds": fc_result["holds"],
        "fc_inf_premium": fc_result["inf_premium"],
        "monotonicity_holds": monotonicity_result["holds"],
        "violates": "N",
    }


def run_mean_variance_violates_m(rng: np.random.Generator) -> dict:
    """rho_MV(X) = E[X] + kappa*Var(X), kappa=EXP2_MV_KAPPA, satisfies SRA, FC
    and N (it is an OCE, Example 1), but violates Monotonicity: X ~ {0,4}
    equal probability satisfies X <= 4 a.s., yet rho_MV(X) = 2 + 4*kappa > 4 =
    rho_MV(4) for kappa > 0 (paper Sec. 4.1)."""
    kappa = config.EXP2_MV_KAPPA
    mv_rho = lambda x: mean_variance(x, kappa)
    n = 400_000
    x_discrete = _exact_uniform_support(config.EXP2_MV_X_VALUES, config.EXP2_MV_X_PROBS, n)
    y_constant = np.full(n, 4.0)
    monotonicity_result = check_monotonicity(mv_rho, x_discrete, y_constant, tol=1e-6)

    normalization = check_normalization(mv_rho, config.EXP1_NORMALIZATION_CONSTANTS, tol=1e-6)
    normalization_holds = all(r["holds"] for r in normalization.values())

    p = config.EXP1_SRA_BERNOULLI_P[2]
    y_bernoulli = sample_bernoulli(config.N_SAMPLES, p, rng)
    x_const = np.full(config.N_SAMPLES, float(p))
    test_functions = [lambda z: z**2, lambda z: np.maximum(z, 0.0), lambda z: np.exp(z)]
    cx_holds = convex_order_holds(x_const, y_bernoulli, test_functions, tol=config.MC_PREMIUM_TOL)
    sra_result = check_sra(mv_rho, x_const, y_bernoulli, tol=config.MC_PREMIUM_TOL)

    x_fc = sample_normal(config.N_SAMPLES, 0.0, 1.0, rng)
    y_fc = sample_normal(config.N_SAMPLES, 1.0, 1.5, rng)
    fc_result = check_fair_coin_compensation(
        mv_rho, x_fc, y_fc, config.EXP1_FC_TRANSFER_GRID, rng, tol=config.MC_PREMIUM_TOL
    )

    return {
        "rho_MV(X~{0,4})": monotonicity_result["rho_X"],
        "rho_MV(4)": monotonicity_result["rho_Y"],
        "monotonicity_holds": monotonicity_result["holds"],
        "normalization_holds": normalization_holds,
        "sra_convex_order_holds": cx_holds,
        "sra_holds": sra_result["holds"],
        "fc_holds": fc_result["holds"],
        "fc_inf_premium": fc_result["inf_premium"],
        "violates": "M",
    }


def main() -> None:
    rng = make_rng(config.RANDOM_SEED + 2)

    xi_res = run_xi_violates_fc(rng)
    chi_res = run_chi_violates_sra(rng)
    e2_res = run_double_expectation_violates_n(rng)
    mv_res = run_mean_variance_violates_m(rng)

    table = pd.DataFrame([
        {"family": "xi = (ES_1/2+ES_3/4)/2", **xi_res},
        {"family": "chi (Prop. 3 construction)", **chi_res},
        {"family": "2*E[.]", **e2_res},
        {"family": f"Mean-Variance kappa={config.EXP2_MV_KAPPA}", **mv_res},
    ])
    table.to_csv(f"{RESULTS_DIR}/exp2_independence_of_axioms.csv", index=False)

    summary = {
        "xi_violates_FC_only": bool(
            (not xi_res["fc_holds"]) and xi_res["normalization_holds"]
            and xi_res["sra_holds"] and xi_res["monotonicity_holds"]
        ),
        "chi_violates_SRA_only": bool(
            (not chi_res["sra_holds"]) and chi_res["normalization_holds"]
            and chi_res["fc_holds"] and chi_res["monotonicity_holds"]
        ),
        "double_expectation_violates_N_only": bool(
            (not e2_res["normalization_holds"]) and e2_res["sra_holds"]
            and e2_res["fc_holds"] and e2_res["monotonicity_holds"]
        ),
        "mean_variance_violates_M_only": bool(
            (not mv_res["monotonicity_holds"]) and mv_res["normalization_holds"]
            and mv_res["sra_holds"] and mv_res["fc_holds"]
        ),
        "xi_inf_premium": xi_res["inf_premium"],
        "chi_SRA_gap": chi_res["chi(X=E[W|G])"] - chi_res["chi(W)"],
        "mean_variance_M_gap": mv_res["rho_MV(X~{0,4})"] - mv_res["rho_MV(4)"],
    }
    with open(f"{RESULTS_DIR}/exp2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("exp_2 summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
