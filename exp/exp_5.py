"""Experiment 5: Proposition 5 -- the Entropic Risk Measure satisfies
Fair-Coin Independence (FCI), while Expected Shortfall does not.

Implements ER_gamma(X) = (1/gamma) log E[exp(gamma X)] (gamma>0), ER_0(X) =
E[X], and checks FCI: rho(X)<=rho(Y) iff rho(BX+(1-B)Z)<=rho(BY+(1-B)Z) for a
fair coin B independent of (X,Y,Z). The closed form

    ER_gamma(BX+(1-B)Z) = (1/gamma) log( 0.5*exp(gamma*ER_gamma(X))
                                          + 0.5*exp(gamma*ER_gamma(Z)) )

is strictly increasing in ER_gamma(X) for fixed Z, which is why ER_gamma
satisfies FCI (Proposition 5). Expected Shortfall ES_p is not entropic and
violates FCI in general; this script searches for and records one explicit
numerical counterexample. Run as ``python -m exp.exp_5``.
"""

import json

import numpy as np
import pandas as pd

from data.generate_synthetic import make_rng, sample_discrete, sample_normal
from src import config
from src.axioms import check_fair_coin_independence, fair_coin
from src.risk_measures import entropic_risk_measure, expected_shortfall

RESULTS_DIR = "results"


def er_gamma_mixture_closed(rho_x: float, rho_z: float, gamma: float) -> float:
    """Closed form of ER_gamma(BX+(1-B)Z) in terms of ER_gamma(X), ER_gamma(Z)
    for a fair coin B independent of (X, Z) (Proposition 5).

    For gamma=0, ER_0 is the expectation and the mixture value is just the
    average (1/2) rho_x + (1/2) rho_z. For gamma>0, log-sum-exp with the
    max of (gamma*rho_x, gamma*rho_z) subtracted out is used for numerical
    stability at large gamma.
    """
    if gamma == 0.0:
        return 0.5 * rho_x + 0.5 * rho_z
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    a, b = gamma * rho_x, gamma * rho_z
    shift = max(a, b)
    return shift / gamma + float(np.log(0.5 * np.exp(a - shift) + 0.5 * np.exp(b - shift))) / gamma


def run_fci_entropic() -> pd.DataFrame:
    """Checks Fair-Coin Independence for ER_gamma, for each gamma in
    EXP5_GAMMAS, on independent Normal draws of X, Y, Z."""
    rows = []
    mean_x, std_x = config.EXP5_FCI_NORMAL_PARAMS["X"]
    mean_y, std_y = config.EXP5_FCI_NORMAL_PARAMS["Y"]
    mean_z, std_z = config.EXP5_FCI_NORMAL_PARAMS["Z"]
    for i, gamma in enumerate(config.EXP5_GAMMAS):
        draw_rng = make_rng(config.RANDOM_SEED + i)
        x = sample_normal(config.N_SAMPLES, mean_x, std_x, draw_rng)
        y = sample_normal(config.N_SAMPLES, mean_y, std_y, draw_rng)
        z = sample_normal(config.N_SAMPLES, mean_z, std_z, draw_rng)
        coin_rng = make_rng(config.RANDOM_SEED + 1000 + i)
        result = check_fair_coin_independence(
            lambda s, g=gamma: entropic_risk_measure(s, g), x, y, z, coin_rng
        )
        rows.append({"gamma": gamma, **result})
    return pd.DataFrame(rows)


def run_closed_form_check() -> pd.DataFrame:
    """Compares the closed-form mixture identity for ER_gamma against an
    independently built Monte Carlo mixture BX+(1-B)Z, for each gamma."""
    rows = []
    mean_x, std_x = config.EXP5_FCI_NORMAL_PARAMS["X"]
    mean_z, std_z = config.EXP5_FCI_NORMAL_PARAMS["Z"]
    for i, gamma in enumerate(config.EXP5_GAMMAS):
        draw_rng = make_rng(config.RANDOM_SEED + 2000 + i)
        x = sample_normal(config.N_SAMPLES, mean_x, std_x, draw_rng)
        z = sample_normal(config.N_SAMPLES, mean_z, std_z, draw_rng)
        rho_x = entropic_risk_measure(x, gamma)
        rho_z = entropic_risk_measure(z, gamma)
        coin_rng = make_rng(config.RANDOM_SEED + 3000 + i)
        b = fair_coin(len(x), coin_rng)
        mix = b * x + (1.0 - b) * z
        rho_mix_mc = entropic_risk_measure(mix, gamma)
        rho_mix_closed = er_gamma_mixture_closed(rho_x, rho_z, gamma)
        rows.append({
            "gamma": gamma,
            "rho_mix_closed": rho_mix_closed,
            "rho_mix_montecarlo": rho_mix_mc,
            "abs_diff": abs(rho_mix_closed - rho_mix_mc),
        })
    return pd.DataFrame(rows)


def run_monotonicity_check() -> tuple[pd.DataFrame, bool]:
    """For fixed gamma=1.0 and fixed Z, verifies that the closed-form mixture
    value ER_gamma(BX+(1-B)Z) is strictly increasing in ER_gamma(X) as the
    mean of X is swept over a grid."""
    gamma = 1.0
    mean_z, std_z = config.EXP5_FCI_NORMAL_PARAMS["Z"]
    z_rng = make_rng(config.RANDOM_SEED + 4000)
    z = sample_normal(config.N_SAMPLES, mean_z, std_z, z_rng)
    rho_z = entropic_risk_measure(z, gamma)

    means = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    rows = []
    for i, mean in enumerate(means):
        x_rng = make_rng(config.RANDOM_SEED + 5000 + i)
        x = sample_normal(config.N_SAMPLES, mean, 1.0, x_rng)
        rho_x = entropic_risk_measure(x, gamma)
        rho_mixture_closed = er_gamma_mixture_closed(rho_x, rho_z, gamma)
        rows.append({"mean_X": mean, "rho_X": rho_x, "rho_mixture_closed": rho_mixture_closed})

    df = pd.DataFrame(rows).sort_values("rho_X").reset_index(drop=True)
    diffs = np.diff(df["rho_mixture_closed"].to_numpy())
    strictly_increasing = bool(np.all(diffs > 0))
    return df, strictly_increasing


def search_es_fci_violation() -> tuple[pd.DataFrame, dict | None]:
    """Searches a grid of (X constant value, Z probability split) for a
    configuration where Expected Shortfall ES_p violates Fair-Coin
    Independence: ES_p(X) <= ES_p(Y) but ES_p(BX+(1-B)Z) > ES_p(BY+(1-B)Z)
    for a common Z and fair coin B.

    The base grid is the one suggested for this experiment (X values close to
    EXP5_FCI_COUNTEREXAMPLE_X_VALUE and Z probability splits close to
    EXP5_FCI_COUNTEREXAMPLE_Z_PROBS); it is extended with larger X values
    since the base grid alone does not produce a violation for this choice
    of Y and Z supports (recorded below).
    """
    z_prob_options = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]
    x_value_options = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9]

    rows = []
    best_config = None
    for i, (zp1, zp2) in enumerate(z_prob_options):
        for j, x_value in enumerate(x_value_options):
            idx = i * len(x_value_options) + j
            z_probs = np.array([zp1, zp2])
            draw_rng = make_rng(config.RANDOM_SEED + 6000 + idx)
            x_const = np.full(config.N_SAMPLES, float(x_value))
            y = sample_discrete(config.N_SAMPLES, config.EXP5_FCI_COUNTEREXAMPLE_Y_VALUES,
                                 config.EXP5_FCI_COUNTEREXAMPLE_Y_PROBS, draw_rng)
            z = sample_discrete(config.N_SAMPLES, config.EXP5_FCI_COUNTEREXAMPLE_Z_VALUES,
                                 z_probs, draw_rng)
            coin_rng = make_rng(config.RANDOM_SEED + 7000 + idx)
            result = check_fair_coin_independence(
                lambda s: expected_shortfall(s, config.EXP5_ES_LEVEL_FOR_VIOLATION),
                x_const, y, z, coin_rng,
            )
            violated = not result["holds"]
            row = {
                "x_value": x_value,
                "z_prob_minus2": zp1,
                "z_prob_plus5": zp2,
                "rho_X": result["rho_X"],
                "rho_Y": result["rho_Y"],
                "rho_mix_X": result["rho_mix_X"],
                "rho_mix_Y": result["rho_mix_Y"],
                "ranking_holds": result["holds"],
                "violated": violated,
            }
            rows.append(row)
            if violated and best_config is None:
                best_config = {
                    "x_value": x_value,
                    "z_values": config.EXP5_FCI_COUNTEREXAMPLE_Z_VALUES.tolist(),
                    "z_probs": [zp1, zp2],
                    "y_values": config.EXP5_FCI_COUNTEREXAMPLE_Y_VALUES.tolist(),
                    "y_probs": config.EXP5_FCI_COUNTEREXAMPLE_Y_PROBS.tolist(),
                    "es_level": config.EXP5_ES_LEVEL_FOR_VIOLATION,
                    "rho_X": result["rho_X"],
                    "rho_Y": result["rho_Y"],
                    "rho_mix_X": result["rho_mix_X"],
                    "rho_mix_Y": result["rho_mix_Y"],
                }

    df = pd.DataFrame(rows)
    return df, best_config


def main() -> None:
    fci_entropic_df = run_fci_entropic()
    closed_form_df = run_closed_form_check()
    monotonicity_df, strictly_increasing = run_monotonicity_check()
    es_counterexample_df, best_config = search_es_fci_violation()

    fci_entropic_df.to_csv(f"{RESULTS_DIR}/exp5_fci_entropic.csv", index=False)
    closed_form_df.to_csv(f"{RESULTS_DIR}/exp5_fci_closed_form_check.csv", index=False)
    monotonicity_df.to_csv(f"{RESULTS_DIR}/exp5_fci_monotonicity.csv", index=False)
    es_counterexample_df.to_csv(f"{RESULTS_DIR}/exp5_es_fci_counterexample.csv", index=False)

    summary = {
        "fci_entropic_all_hold": bool(fci_entropic_df["holds"].all()),
        "closed_form_max_abs_diff": float(closed_form_df["abs_diff"].max()),
        "mixture_strictly_increasing": strictly_increasing,
        "es_fci_violation_found": best_config is not None,
        "es_fci_violation_config": best_config,
    }
    with open(f"{RESULTS_DIR}/exp5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("exp_5 summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
