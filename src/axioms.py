"""Numerical verification of the axioms and properties of
Blier-Wong, Lauzier and Van Oosten (2026).

Every check takes a risk-measure callable `rho: np.ndarray -> float` (an
empirical sample represents a random variable X in L^infty) and returns a
dictionary with the computed quantities and a boolean `holds` flag, so that
exp/ scripts can tabulate results without duplicating the verification logic.

Custom implementation: these axioms (SRA, N, FC, M, ER, PH, Prudence, FCI) are
specific to this paper; no library implements them.
"""

from typing import Callable

import numpy as np

RiskMeasure = Callable[[np.ndarray], float]


def fair_coin(n: int, rng: np.random.Generator) -> np.ndarray:
    """A fair coin B in {0,1} with P(B=1) = 1/2, independent of any sample it
    multiplies. Uses an exactly balanced split (n // 2 heads, randomly
    permuted) rather than i.i.d. Bernoulli(0.5) draws: an i.i.d. coin has an
    empirical mean deviating from 1/2 by O(1/sqrt(n)), which, multiplied by a
    transfer a, biases the fairness-premium estimate at the edges of a wide
    grid of a (Monte Carlo variance-reduction choice, not specified in the
    paper)."""
    b = np.zeros(n)
    b[: n // 2] = 1.0
    rng.shuffle(b)
    return b


def check_normalization(rho: RiskMeasure, constants: list[float], n_samples: int = 1000,
                         tol: float = 1e-6) -> dict:
    """Axiom (N): rho(c) = c for every constant c (paper Sec. 3.1)."""
    results = {}
    for c in constants:
        sample = np.full(n_samples, float(c))
        rho_c = rho(sample)
        results[c] = {"rho_c": rho_c, "holds": bool(abs(rho_c - c) <= tol)}
    return results


def convex_order_holds(X: np.ndarray, Y: np.ndarray,
                        test_functions: list[Callable[[np.ndarray], np.ndarray]],
                        tol: float = 1e-6) -> bool:
    """Checks X <=_cx Y via E[phi(X)] <= E[phi(Y)] for convex test functions phi (paper Sec. 2).

    `tol` should be widened for Monte Carlo samples to absorb sampling noise.
    """
    for phi in test_functions:
        if np.mean(phi(X)) > np.mean(phi(Y)) + tol:
            return False
    return True


def check_sra(rho: RiskMeasure, X: np.ndarray, Y: np.ndarray, tol: float = 1e-6) -> dict:
    """Axiom (SRA): rho(X) <= rho(Y) whenever X <=_cx Y (paper Sec. 3.1).

    The convex-order premise must be verified independently (see
    `convex_order_holds`); this function only checks the risk-measure
    inequality that the axiom requires.
    """
    rho_x, rho_y = rho(X), rho(Y)
    return {"rho_X": rho_x, "rho_Y": rho_y, "holds": bool(rho_x <= rho_y + tol)}


def fair_coin_mixture(X: np.ndarray, Y: np.ndarray, a: float,
                       rng: np.random.Generator) -> np.ndarray:
    """A transfer on the coin: B(X+a) + (1-B)(Y-a), B a fair coin independent of (X,Y)
    (paper Sec. 3.1, Axiom (FC)). X and Y are resampled to equal length n if needed."""
    n = max(len(X), len(Y))
    x = rng.choice(X, size=n, replace=True) if len(X) != n else X
    y = rng.choice(Y, size=n, replace=True) if len(Y) != n else Y
    b = fair_coin(n, rng)
    return b * (x + a) + (1.0 - b) * (y - a)


def fairness_premium(rho: RiskMeasure, X: np.ndarray, Y: np.ndarray, a_grid: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """pi_rho(a) = R(a) - rho(X) - rho(Y) = 2*rho(W_1(a)) - rho(X) - rho(Y) (paper Sec. 3.2).

    The underlying realizations of X, Y and the fair coin B are drawn once and
    reused for every a in the grid, since (X, Y, B) are fixed random variables
    in the paper's definition of Axiom (FC) and only the transfer a varies;
    redrawing B independently at each a would add spurious a-dependent sampling
    noise (the empirical mean of B deviates from 1/2 by O(1/sqrt(n)), which,
    multiplied by a, grows at the edges of the grid).
    """
    rho_x, rho_y = rho(X), rho(Y)
    n = max(len(X), len(Y))
    x = rng.choice(X, size=n, replace=True) if len(X) != n else X
    y = rng.choice(Y, size=n, replace=True) if len(Y) != n else Y
    b = fair_coin(n, rng)
    premiums = np.empty(len(a_grid))
    for i, a in enumerate(a_grid):
        w = b * (x + float(a)) + (1.0 - b) * (y - float(a))
        premiums[i] = 2.0 * rho(w) - rho_x - rho_y
    return premiums


def check_fair_coin_compensation(rho: RiskMeasure, X: np.ndarray, Y: np.ndarray,
                                  a_grid: np.ndarray, rng: np.random.Generator,
                                  tol: float = 1e-2) -> dict:
    """Axiom (FC): inf_a rho(B(X+a)+(1-B)(Y-a)) = (rho(X)+rho(Y))/2 (paper Eq. 2)."""
    rho_x, rho_y = rho(X), rho(Y)
    target = 0.5 * (rho_x + rho_y)
    premiums = fairness_premium(rho, X, Y, a_grid, rng)
    inf_idx = int(np.argmin(premiums))
    inf_premium = float(premiums[inf_idx])
    a_star = float(a_grid[inf_idx])
    return {
        "rho_X": rho_x,
        "rho_Y": rho_y,
        "target_average": target,
        "inf_premium": inf_premium,
        "a_star": a_star,
        "holds": bool(abs(inf_premium) <= tol),
    }


def check_monotonicity(rho: RiskMeasure, X: np.ndarray, Y: np.ndarray, tol: float = 1e-6) -> dict:
    """Axiom (M): X <= Y a.s. implies rho(X) <= rho(Y) (paper Sec. 3.1)."""
    rho_x, rho_y = rho(X), rho(Y)
    return {"rho_X": rho_x, "rho_Y": rho_y, "holds": bool(rho_x <= rho_y + tol)}


def check_equal_mean_randomization(rho: RiskMeasure, X: np.ndarray, Y: np.ndarray,
                                    rng: np.random.Generator, tol: float = 1e-2) -> dict:
    """Property (ER): rho(BX+(1-B)Y) <= (rho(X)+rho(Y))/2 when E[X]=E[Y] (paper Sec. 4.1).

    For Mean-Variance this holds with equality, which is also checked here.
    """
    mean_x, mean_y = float(np.mean(X)), float(np.mean(Y))
    mixture = fair_coin_mixture(X, Y, 0.0, rng)
    rho_x, rho_y, rho_mix = rho(X), rho(Y), rho(mixture)
    average = 0.5 * (rho_x + rho_y)
    return {
        "mean_X": mean_x,
        "mean_Y": mean_y,
        "rho_X": rho_x,
        "rho_Y": rho_y,
        "rho_mixture": rho_mix,
        "average": average,
        "holds_inequality": bool(rho_mix <= average + tol),
        "holds_equality": bool(abs(rho_mix - average) <= tol),
    }


def check_fair_coin_independence(rho: RiskMeasure, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                  rng: np.random.Generator, tol: float = 1e-2) -> dict:
    """Property (FI): rho(X)<=rho(Y) iff rho(BX+(1-B)Z)<=rho(BY+(1-B)Z) (paper Sec. 4.3).

    Uses the same coin realization B for both mixtures, as in Axiom (FC)."""
    n = max(len(X), len(Y), len(Z))
    x = rng.choice(X, size=n, replace=True) if len(X) != n else X
    y = rng.choice(Y, size=n, replace=True) if len(Y) != n else Y
    z = rng.choice(Z, size=n, replace=True) if len(Z) != n else Z
    b = fair_coin(n, rng)
    mix_x = b * x + (1.0 - b) * z
    mix_y = b * y + (1.0 - b) * z
    rho_x, rho_y = rho(X), rho(Y)
    rho_mix_x, rho_mix_y = rho(mix_x), rho(mix_y)
    ranking_before = rho_x <= rho_y
    ranking_after = rho_mix_x <= rho_mix_y
    return {
        "rho_X": rho_x,
        "rho_Y": rho_y,
        "rho_mix_X": rho_mix_x,
        "rho_mix_Y": rho_mix_y,
        "holds": bool(ranking_before == ranking_after),
    }


def fatou_prudence_sequence(n_values: list[int], sample_size: int,
                             rng: np.random.Generator) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """A Fatou-type sequence X_n -> 0 a.s. that is not uniformly bounded in n.

    Custom implementation (paper Sec. 4.2, Corollary 1): Corollary 1 states
    that Prudence isolates pure Expected Shortfall among the mixtures
    lambda*E + (1-lambda)*ES_p. A single uniform draw U is reused for every
    n, and A_n = {U < 1/n} is nested decreasing (A_n shrinks as n grows), so
    X_n = -n * 1{U<1/n} converges to X=0 almost surely: for almost every
    draw of U there is a finite n0(U) after which U >= 1/n. However
    E[X_n] = -n * P(U<1/n) = -1 for every n, so liminf E[X_n] = -1 < 0 =
    E[X]: plain expectation is not lower semicontinuous under a.s.
    convergence without a uniform bound, the classical failure mode of
    Fatou's lemma absent domination. Expected Shortfall at level p does not
    see this spike once 1/n < 1-p, because it only integrates the upper
    (1-p) tail of the loss distribution, which stays prudent on this
    sequence.
    """
    u = rng.random(sample_size)
    sequence = {n: -float(n) * (u < 1.0 / n).astype(float) for n in n_values}
    limit = np.zeros(sample_size)
    return sequence, limit


def check_positive_homogeneity(rho: RiskMeasure, X: np.ndarray, t_values: list[float],
                                tol: float = 1e-2) -> dict:
    """Property (PH): rho(tX) = t*rho(X) for all t > 0 (paper Sec. 4.2)."""
    rho_x = rho(X)
    results = {}
    for t in t_values:
        rho_tx = rho(t * X)
        results[t] = {
            "rho_tX": rho_tx,
            "t_rho_X": t * rho_x,
            "holds": bool(abs(rho_tx - t * rho_x) <= tol * max(1.0, abs(t * rho_x))),
        }
    return results


def check_prudence(rho: RiskMeasure, X: np.ndarray,
                    perturbation_fn: Callable[[np.ndarray, int, np.random.Generator], np.ndarray],
                    n_values: list[int], rng: np.random.Generator, tol: float = 2e-2) -> dict:
    """Property (P) Prudence: for X_n -> X a.s., liminf rho(X_n) >= rho(X) (paper Sec. 4.2).

    `perturbation_fn(X, n, rng)` must build a sequence converging a.s. to X,
    e.g. X_n = X + Z/n for a bounded Z.
    """
    rho_x = rho(X)
    sequence = []
    for n in n_values:
        x_n = perturbation_fn(X, n, rng)
        sequence.append(rho(x_n))
    liminf_estimate = float(np.min(sequence[-max(1, len(sequence) // 3):]))
    return {
        "rho_X": rho_x,
        "sequence": sequence,
        "liminf_estimate": liminf_estimate,
        "holds": bool(liminf_estimate >= rho_x - tol),
    }
