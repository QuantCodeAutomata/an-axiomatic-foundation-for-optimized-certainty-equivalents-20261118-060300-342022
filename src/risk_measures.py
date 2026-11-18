"""Specific risk measures used throughout the paper's examples and
propositions: Expected Shortfall, the Entropic Risk Measure, the
Mean-Variance criterion, Value-at-Risk, and the two risk measures chi and
xi used to prove the independence of the axioms (Proposition 3).

Monte Carlo variants (`value_at_risk`, `expected_shortfall`, ...) operate on
an iid sample and are used for the continuous-law checks in exp_1, exp_4 and
exp_5. Exact variants (`exact_var_discrete`, `exact_es_discrete`) operate on
an explicit finite support/probability pair and reproduce the closed-form
numbers quoted in the paper's examples (exp_2, exp_6, exp_7).

Library check: `scipy.stats` provides generic quantile functions but not the
Rockafellar-Uryasev Expected-Shortfall integral or the OCE-specific risk
measures of this paper, so these are implemented directly from the paper's
formulas (Examples 1-4, Appendix A).
"""

import numpy as np


def expectation(samples: np.ndarray) -> float:
    """E[X]."""
    return float(np.mean(samples))


def value_at_risk(samples: np.ndarray, p: float) -> float:
    """Monte Carlo Value-at-Risk VaR_p(X) = inf{x : P(X<=x) >= p} (paper Sec. 3.2).

    Uses the "higher" interpolation so the estimate satisfies the inf-quantile
    definition exactly on the empirical distribution (no interpolation bias).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return float(np.quantile(samples, p, method="higher"))


def expected_shortfall(samples: np.ndarray, p: float) -> float:
    """Expected Shortfall via the Rockafellar-Uryasev formula
    ES_p(X) = VaR_p(X) + E[(X - VaR_p(X))_+] / (1-p) (Example 2), i.e. the OCE
    inf_m {m + E[(X-m)_+]/(1-p)} evaluated at its minimizer m = VaR_p(X).

    This equals (1/(1-p)) E[X 1{X>=VaR_p(X)}] only when the law has no atom
    at VaR_p(X) (the continuous case); the Rockafellar-Uryasev form used here
    is exact for atomic/discrete laws as well, since it does not depend on
    how the probability mass exactly at the quantile is split between the
    tail and its complement.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    var = value_at_risk(samples, p)
    tail_excess = float(np.mean(np.maximum(samples - var, 0.0)))
    return var + tail_excess / (1.0 - p)


def entropic_risk_measure(samples: np.ndarray, gamma: float) -> float:
    """Entropic Risk Measure ER_gamma(X) = (1/gamma) log E[exp(gamma X)] (Eq. 3).

    ER_0 is defined as the limit E[X] (paper Proposition 5).
    """
    if gamma == 0.0:
        return expectation(samples)
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    # log-sum-exp for numerical stability: (1/gamma) * (gamma*max + log(mean(exp(gamma*(x-max)))))
    shift = float(np.max(samples))
    return shift + float(np.log(np.mean(np.exp(gamma * (samples - shift))))) / gamma


def mean_variance(samples: np.ndarray, kappa: float) -> float:
    """Mean-Variance criterion rho_MV(X) = E[X] + kappa * Var(X), kappa >= 0 (Example 1)."""
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    return float(np.mean(samples) + kappa * np.var(samples))


def chi_risk_measure(samples: np.ndarray) -> float:
    """chi(X) = inf_m { m + E[(X-m)_+] + P(X>m) } (Proposition 3, second example).

    The objective is piecewise linear in m with kinks only at the sample
    values, so its infimum over the empirical distribution is attained at one
    of them. Evaluated in O(n log n) via a single sort and prefix sums (the
    direct pairwise formula is O(n^2) in memory and does not scale to the
    sample sizes used elsewhere in this replication).
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    sorted_x = np.sort(samples)
    candidates = np.unique(sorted_x)
    prefix_sum = np.concatenate(([0.0], np.cumsum(sorted_x)))
    total_sum = prefix_sum[-1]
    idx = np.searchsorted(sorted_x, candidates, side="right")  # count of x_i <= candidate
    sum_below = prefix_sum[idx]
    count_above = n - idx
    sum_above = total_sum - sum_below
    objective = candidates + (sum_above - candidates * count_above) / n + count_above / n
    return float(np.min(objective))


def double_expectation(samples: np.ndarray) -> float:
    """The risk measure 2*E[X], which violates Normalization (Proposition 3)."""
    return float(2.0 * np.mean(samples))


def xi_risk_measure(samples: np.ndarray) -> float:
    """xi = (1/2) ES_{1/2} + (1/2) ES_{3/4} (Example 4 / Proposition 3)."""
    return 0.5 * expected_shortfall(samples, 0.5) + 0.5 * expected_shortfall(samples, 0.75)


def exact_var_discrete(values: np.ndarray, probs: np.ndarray, p: float) -> float:
    """Exact VaR_p for a finite discrete law given by (values, probs) (Sec. 3.2).

    VaR_p(X) = inf{x : P(X<=x) >= p}, computed from the exact cdf.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    order = np.argsort(values)
    v, pr = values[order], probs[order]
    cdf = np.cumsum(pr)
    idx = int(np.searchsorted(cdf, p, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def mix_discrete_fair_coin(x_values: np.ndarray, x_probs: np.ndarray,
                            y_values: np.ndarray, y_probs: np.ndarray,
                            a: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact law of W_1(a) = B(X+a) + (1-B)(Y-a) for a fair coin B independent
    of the finite discrete laws of X and Y (paper Sec. 3.2, Example 4).

    Returns the merged (values, probs) of the finite mixture: the branch
    B=1 shifts X's support by +a with weight 1/2, the branch B=0 shifts Y's
    support by -a with weight 1/2; values that coincide are merged.
    """
    branch_x = x_values + a
    branch_y = y_values - a
    values = np.concatenate([branch_x, branch_y])
    probs = np.concatenate([0.5 * x_probs, 0.5 * y_probs])
    unique_values, inverse = np.unique(values, return_inverse=True)
    merged_probs = np.zeros(unique_values.shape[0])
    np.add.at(merged_probs, inverse, probs)
    return unique_values, merged_probs


def exact_es_discrete(values: np.ndarray, probs: np.ndarray, p: float) -> float:
    """Exact ES_p for a finite discrete law via the quantile integral
    ES_p(X) = (1/(1-p)) * integral_p^1 VaR_s(X) ds (paper Sec. 3.2, Example 2).

    The integral is evaluated piece by piece over the probability mass
    intervals of the (sorted) support, which is exact for a finite law.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    order = np.argsort(values)
    v, pr = values[order], probs[order]
    cdf = np.cumsum(pr)
    lower = np.concatenate(([0.0], cdf[:-1]))
    # Overlap of each mass interval [lower_i, cdf_i] with the integration range [p, 1].
    overlap = np.clip(cdf, p, 1.0) - np.clip(lower, p, 1.0)
    overlap = np.clip(overlap, 0.0, None)
    integral = float(np.sum(overlap * v))
    return integral / (1.0 - p)
