"""Experiment 4: Proposition 4, Corollary 1 -- Expected Shortfall is
characterized, among the mixtures rho_{lambda,p}(X) = lambda*E[X] +
(1-lambda)*ES_p(X) with lambda in [0,1], by Positive Homogeneity (PH, holds
for every lambda) and Prudence (P, holds only for lambda=0, i.e. pure ES_p).

Positive Homogeneity: rho(tX) = t*rho(X) for all t > 0 (paper Sec. 4.2).
Prudence: X_n -> X a.s. implies liminf_n rho(X_n) >= rho(X) (paper Sec. 4.2).

This script checks PH for pure ES_p and for the mixture (both hold), checks
Prudence for pure ES_p under a bounded a.s.-converging perturbation (holds),
and exhibits the paper's Fatou-type counterexample X_n = -n*1{U<1/n} -> 0
a.s. on which plain expectation violates Prudence while ES_p does not -- and
shows that mixing in any positive lambda reintroduces the violation,
confirming Corollary 1 isolates lambda=0. Run as ``python -m exp.exp_4``.
"""

import json
from functools import partial

import numpy as np
import pandas as pd

from data.generate_synthetic import make_rng, sample_normal, sample_uniform
from src import config
from src.axioms import check_positive_homogeneity, check_prudence, fatou_prudence_sequence
from src.oce import loss_expected_shortfall, oce
from src.risk_measures import expectation, expected_shortfall

RESULTS_DIR = "results"


def mixture_rho(samples: np.ndarray, lam: float, p: float) -> float:
    """rho_{lambda,p}(X) = lambda*E[X] + (1-lambda)*ES_p(X), lambda in [0,1]."""
    return lam * expectation(samples) + (1.0 - lam) * expected_shortfall(samples, p)


def verify_oce_matches_es(samples: np.ndarray, p: float, tol: float = 1e-2) -> dict:
    """At lambda=0 the mixture is pure ES_p; cross-checks this against the
    generic OCE minimization with the Expected-Shortfall loss (Example 2)."""
    oce_value = oce(samples, partial(loss_expected_shortfall, p=p))
    es_value = expected_shortfall(samples, p)
    mixture_value = mixture_rho(samples, 0.0, p)
    rel_diff = abs(oce_value - es_value) / max(1.0, abs(es_value))
    return {
        "p": p,
        "oce_value": oce_value,
        "es_value": es_value,
        "mixture_lambda0_value": mixture_value,
        "matches": bool(rel_diff <= tol),
    }


def check_ph_es(X: np.ndarray) -> pd.DataFrame:
    """Axiom (PH) for pure ES_p, for every level p in EXP4_ES_LEVELS."""
    rows = []
    for p in config.EXP4_ES_LEVELS:
        result = check_positive_homogeneity(
            lambda s, pp=p: expected_shortfall(s, pp), X, config.EXP4_PH_SCALARS
        )
        for t, r in result.items():
            rows.append({"p": p, "t": t, "rho_tX": r["rho_tX"], "t_rho_X": r["t_rho_X"], "holds": r["holds"]})
    return pd.DataFrame(rows)


def check_ph_mixture(X: np.ndarray) -> pd.DataFrame:
    """Axiom (PH) for the mixture lambda*E + (1-lambda)*ES_p, for every
    lambda in EXP4_MIXTURE_LAMBDAS (p=EXP4_MIXTURE_P fixed); a mixture of two
    PH risk measures is PH, so this is expected to hold for every lambda."""
    rows = []
    for lam in config.EXP4_MIXTURE_LAMBDAS:
        result = check_positive_homogeneity(
            lambda s, lam=lam: mixture_rho(s, lam, config.EXP4_MIXTURE_P), X, config.EXP4_PH_SCALARS
        )
        for t, r in result.items():
            rows.append({"lam": lam, "t": t, "rho_tX": r["rho_tX"], "t_rho_X": r["t_rho_X"], "holds": r["holds"]})
    return pd.DataFrame(rows)


def perturbation_fn(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """X_n = X + Z/n, Z ~ Uniform(-bound, bound) independent of X: X_n -> X
    a.s. as n -> infinity for any fixed, bounded Z (paper Sec. 4.2)."""
    z = sample_uniform(len(X), -config.EXP4_PRUDENCE_Z_BOUND, config.EXP4_PRUDENCE_Z_BOUND, rng)
    return X + z / n


def check_prudence_es(X: np.ndarray) -> pd.DataFrame:
    """Axiom (P) for pure ES_p under the bounded perturbation above, one row
    per (p, n) in long format."""
    rows = []
    for p in config.EXP4_ES_LEVELS:
        rng = make_rng(config.RANDOM_SEED + 1)
        result = check_prudence(
            lambda s, pp=p: expected_shortfall(s, pp), X, perturbation_fn,
            config.EXP4_PRUDENCE_N_VALUES, rng,
        )
        for n, rho_xn in zip(config.EXP4_PRUDENCE_N_VALUES, result["sequence"]):
            rows.append({
                "p": p, "n": n, "rho_Xn": rho_xn,
                "rho_X": result["rho_X"], "liminf_estimate": result["liminf_estimate"],
                "holds": result["holds"],
            })
    return pd.DataFrame(rows)


def check_prudence_fatou_es_vs_expectation(sequence: dict, limit: np.ndarray,
                                            tol: float = 1e-2) -> pd.DataFrame:
    """Prudence on the Fatou-type sequence (Corollary 1): plain expectation
    violates Prudence (liminf E[X_n] = -1 < 0 = E[X]) while ES_p does not,
    once the rare spike's probability 1/n is below the level it integrates."""
    rows = []
    rho_limit_e = expectation(limit)
    rho_limit_by_p = {p: expected_shortfall(limit, p) for p in config.EXP4_ES_LEVELS}
    for n, x_n in sequence.items():
        rho_n = expectation(x_n)
        rows.append({
            "n": n, "measure": "E", "rho_Xn": rho_n, "rho_limit": rho_limit_e,
            "satisfies_prudence_here": bool(rho_n >= rho_limit_e - tol),
        })
        for p in config.EXP4_ES_LEVELS:
            rho_np = expected_shortfall(x_n, p)
            rows.append({
                "n": n, "measure": f"ES_p={p}", "rho_Xn": rho_np, "rho_limit": rho_limit_by_p[p],
                "satisfies_prudence_here": bool(rho_np >= rho_limit_by_p[p] - tol),
            })
    return pd.DataFrame(rows)


def check_prudence_fatou_mixture(sequence: dict, limit: np.ndarray, tol: float = 1e-2) -> pd.DataFrame:
    """Prudence on the same Fatou-type sequence for the mixture
    lambda*E + (1-lambda)*ES_p: expected to fail whenever lambda > 0, since
    the lambda*E term then dominates with liminf lambda*(-1) < 0 (Corollary 1
    isolates lambda=0, i.e. pure ES_p, as the only Prudent mixture)."""
    rows = []
    rho_limit_by_lam = {lam: mixture_rho(limit, lam, config.EXP4_MIXTURE_P) for lam in config.EXP4_MIXTURE_LAMBDAS}
    for lam in config.EXP4_MIXTURE_LAMBDAS:
        rho_limit = rho_limit_by_lam[lam]
        for n, x_n in sequence.items():
            rho_n = mixture_rho(x_n, lam, config.EXP4_MIXTURE_P)
            rows.append({
                "lam": lam, "n": n, "rho_Xn": rho_n, "rho_limit": rho_limit,
                "satisfies_prudence_here": bool(rho_n >= rho_limit - tol),
            })
    return pd.DataFrame(rows)


def main() -> None:
    rng = make_rng(config.RANDOM_SEED)
    X = sample_normal(config.N_SAMPLES, config.EXP4_BASE_NORMAL_MEAN, config.EXP4_BASE_NORMAL_STD, rng)

    oce_matches = [verify_oce_matches_es(X, p) for p in config.EXP4_ES_LEVELS]

    ph_es_df = check_ph_es(X)
    ph_mixture_df = check_ph_mixture(X)
    prudence_es_df = check_prudence_es(X)

    fatou_rng = make_rng(config.RANDOM_SEED + 2)
    sequence, limit = fatou_prudence_sequence(config.EXP4_FATOU_N_VALUES, config.EXP4_FATOU_SAMPLE_SIZE, fatou_rng)
    fatou_es_vs_e_df = check_prudence_fatou_es_vs_expectation(sequence, limit)
    fatou_mixture_df = check_prudence_fatou_mixture(sequence, limit)

    ph_es_df.to_csv(f"{RESULTS_DIR}/exp4_positive_homogeneity_es.csv", index=False)
    ph_mixture_df.to_csv(f"{RESULTS_DIR}/exp4_positive_homogeneity_mixture.csv", index=False)
    prudence_es_df.to_csv(f"{RESULTS_DIR}/exp4_prudence_es_uniform_perturbation.csv", index=False)
    fatou_es_vs_e_df.to_csv(f"{RESULTS_DIR}/exp4_prudence_fatou_es_vs_expectation.csv", index=False)
    fatou_mixture_df.to_csv(f"{RESULTS_DIR}/exp4_prudence_fatou_mixture.csv", index=False)

    max_n = max(config.EXP4_FATOU_N_VALUES)
    e_at_max_n = fatou_es_vs_e_df[(fatou_es_vs_e_df["measure"] == "E") & (fatou_es_vs_e_df["n"] == max_n)]
    fatou_expectation_violates_prudence = bool(not e_at_max_n["satisfies_prudence_here"].iloc[0])

    es_rows = fatou_es_vs_e_df[fatou_es_vs_e_df["measure"].str.startswith("ES_p=")].copy()
    es_rows["p"] = es_rows["measure"].str.replace("ES_p=", "", regex=False).astype(float)
    es_large_n = es_rows[es_rows["n"] > 1.0 / (1.0 - es_rows["p"])]
    fatou_es_all_satisfy_prudence_for_large_n = bool(es_large_n["satisfies_prudence_here"].all())

    p_mix = config.EXP4_MIXTURE_P
    mixture_large_n = fatou_mixture_df[fatou_mixture_df["n"] > 1.0 / (1.0 - p_mix)]
    positive_lambda_rows = mixture_large_n[mixture_large_n["lam"] > 0]
    fatou_mixture_violates_prudence_for_lambda_positive = bool(
        (~positive_lambda_rows["satisfies_prudence_here"]).all()
    )

    summary = {
        "ph_es_all_hold": bool(ph_es_df["holds"].all()),
        "ph_mixture_all_hold": bool(ph_mixture_df["holds"].all()),
        "prudence_es_all_hold": bool(prudence_es_df.groupby("p")["holds"].first().all()),
        "fatou_expectation_violates_prudence": fatou_expectation_violates_prudence,
        "fatou_es_all_satisfy_prudence_for_large_n": fatou_es_all_satisfy_prudence_for_large_n,
        "fatou_mixture_violates_prudence_for_lambda_positive": fatou_mixture_violates_prudence_for_lambda_positive,
        "oce_matches_es_lambda0": oce_matches,
        "oce_matches_es_lambda0_all": bool(all(m["matches"] for m in oce_matches)),
    }
    with open(f"{RESULTS_DIR}/exp4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("exp_4 summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
