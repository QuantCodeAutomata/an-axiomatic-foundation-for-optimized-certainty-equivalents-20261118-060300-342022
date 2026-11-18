"""Experiment 1: Theorem 1 -- OCEs satisfy Strong Risk Aversion (SRA),
Normalization (N), and Fair-Coin Compensation (FC).

Implements the OCE formula rho_ell(X) = inf_m { m + E[ell(X-m)] } (Eq. 1) for
the quadratic (Mean-Variance), Expected-Shortfall and entropic loss families
of Examples 1-3, and checks the three axioms of Theorem 1 on each resulting
risk measure. Run as ``python -m exp.exp_1``.
"""

import json
from functools import partial

import numpy as np
import pandas as pd

from data.generate_synthetic import make_rng, sample_bernoulli, sample_normal
from src import config
from src.axioms import check_fair_coin_compensation, check_normalization, check_sra, convex_order_holds
from src.oce import loss_entropic, loss_expected_shortfall, loss_expectation, loss_quadratic, oce
from src.risk_measures import entropic_risk_measure, expectation, expected_shortfall, mean_variance

RESULTS_DIR = "results"


def build_families() -> dict[str, dict]:
    """The OCE risk measures tested, each with its loss and closed-form check."""
    families: dict[str, dict] = {
        "E": {
            "rho": lambda x: oce(x, loss_expectation),
            "closed_form": expectation,
        }
    }
    for kappa in config.EXP1_KAPPA_VALUES:
        families[f"MV_kappa={kappa}"] = {
            "rho": lambda x, k=kappa: oce(x, partial(loss_quadratic, kappa=k)),
            "closed_form": lambda x, k=kappa: mean_variance(x, k),
        }
    for p in config.EXP1_ES_LEVELS:
        families[f"ES_p={p}"] = {
            "rho": lambda x, pp=p: oce(x, partial(loss_expected_shortfall, p=pp)),
            "closed_form": lambda x, pp=p: expected_shortfall(x, pp),
        }
    for gamma in config.EXP1_ENTROPIC_GAMMAS:
        families[f"ER_gamma={gamma}"] = {
            "rho": lambda x, g=gamma: oce(x, partial(loss_entropic, gamma=g)),
            "closed_form": lambda x, g=gamma: entropic_risk_measure(x, g),
        }
    return families


def check_closed_form_match(families: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Verifies the generic OCE minimization recovers the closed-form value of
    each risk measure, across increasing sample sizes (convergence check)."""
    rows = []
    for n in config.EXP1_CONVERGENCE_SAMPLE_SIZES:
        x = sample_normal(n, 0.0, 1.0, rng)
        for name, fam in families.items():
            oce_val = fam["rho"](x)
            closed_val = fam["closed_form"](x)
            rows.append({
                "family": name, "n": n, "oce_value": oce_val, "closed_form_value": closed_val,
                "abs_diff": abs(oce_val - closed_val),
                "matches": bool(abs(oce_val - closed_val) <= 10.0 * config.TOLERANCE),
            })
    return pd.DataFrame(rows)


def check_normalization_all(families: dict) -> pd.DataFrame:
    """Axiom (N): rho(c) = c for every constant c, for every family."""
    rows = []
    for name, fam in families.items():
        result = check_normalization(fam["rho"], config.EXP1_NORMALIZATION_CONSTANTS, tol=1e-4)
        for c, r in result.items():
            rows.append({"family": name, "c": c, "rho_c": r["rho_c"], "holds": r["holds"]})
    return pd.DataFrame(rows)


def check_sra_all(families: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Axiom (SRA): rho(X) <= rho(Y) when X <=_cx Y, for X=E[Y] constant and
    Y ~ Bernoulli(p); a constant equal to the mean is always dominated in
    convex order by Jensen's inequality, which is verified explicitly."""
    rows = []
    test_functions = [lambda x: x**2, lambda x: np.maximum(x, 0.0), lambda x: np.exp(x)]
    for p in config.EXP1_SRA_BERNOULLI_P:
        y = sample_bernoulli(config.N_SAMPLES, p, rng)
        x = np.full(config.N_SAMPLES, float(p))
        cx_holds = convex_order_holds(x, y, test_functions, tol=config.MC_PREMIUM_TOL)
        for name, fam in families.items():
            res = check_sra(fam["rho"], x, y, tol=config.MC_PREMIUM_TOL)
            rows.append({"family": name, "p": p, "convex_order_holds": cx_holds,
                         "rho_X": res["rho_X"], "rho_Y": res["rho_Y"], "holds": res["holds"]})
    return pd.DataFrame(rows)


def check_fc_all(families: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Axiom (FC): inf_a rho(B(X+a)+(1-B)(Y-a)) = (rho(X)+rho(Y))/2 (Eq. 2)."""
    rows = []
    mean_x, mean_y = config.EXP1_FC_PAIR_NORMAL_MEANS
    std_x, std_y = config.EXP1_FC_PAIR_NORMAL_STDS
    x = sample_normal(config.N_SAMPLES, mean_x, std_x, rng)
    y = sample_normal(config.N_SAMPLES, mean_y, std_y, rng)
    for name, fam in families.items():
        if name in config.EXP1_FC_EXCLUDE_FAMILIES:
            continue
        res = check_fair_coin_compensation(fam["rho"], x, y, config.EXP1_FC_TRANSFER_GRID, rng,
                                            tol=config.MC_PREMIUM_TOL)
        rows.append({"family": name, **res})
    return pd.DataFrame(rows)


def main() -> None:
    rng = make_rng(config.RANDOM_SEED)
    families = build_families()

    convergence_df = check_closed_form_match(families, rng)
    normalization_df = check_normalization_all(families)
    sra_df = check_sra_all(families, rng)
    fc_df = check_fc_all(families, rng)

    convergence_df.to_csv(f"{RESULTS_DIR}/exp1_convergence.csv", index=False)
    normalization_df.to_csv(f"{RESULTS_DIR}/exp1_normalization.csv", index=False)
    sra_df.to_csv(f"{RESULTS_DIR}/exp1_sra.csv", index=False)
    fc_df.to_csv(f"{RESULTS_DIR}/exp1_fair_coin_compensation.csv", index=False)

    summary = {
        "n_families": len(families),
        "normalization_all_hold": bool(normalization_df["holds"].all()),
        "sra_all_hold": bool(sra_df["holds"].all()),
        "sra_all_convex_order_verified": bool(sra_df["convex_order_holds"].all()),
        "fc_all_hold": bool(fc_df["holds"].all()),
        "convergence_final_n_all_match": bool(
            convergence_df[convergence_df["n"] == max(config.EXP1_CONVERGENCE_SAMPLE_SIZES)]["matches"].all()
        ),
    }
    with open(f"{RESULTS_DIR}/exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("exp_1 summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
