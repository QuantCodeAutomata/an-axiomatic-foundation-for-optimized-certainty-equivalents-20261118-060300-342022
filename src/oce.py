"""Optimized Certainty Equivalent (OCE) and its loss functions.

Implements Eq. (1) of Blier-Wong, Lauzier and Van Oosten (2026):

    rho_ell(X) = inf_{m in R} { m + E[ell(X - m)] },

where ell is a convex loss function with ell(0) = 0 and ell(x) >= x. Section 2
of the paper (Examples 1-3) gives the loss functions implemented below.

Custom implementation: the OCE minimization itself is novel to this paper's
framework; scipy.optimize.minimize_scalar (bounded Brent search) is used as
the numerical workhorse since no library implements the OCE infimum directly.
"""

from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

LossFn = Callable[[np.ndarray], np.ndarray]


def loss_expectation(x: np.ndarray) -> np.ndarray:
    """Risk-neutral loss ell(x) = x, generating rho_ell = E[.] (paper Sec. 2)."""
    return x


def loss_quadratic(x: np.ndarray, kappa: float) -> np.ndarray:
    """Mean-Variance loss ell(x) = x + kappa * x^2, kappa >= 0 (paper Example 1)."""
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    return x + kappa * x**2


def loss_expected_shortfall(x: np.ndarray, p: float) -> np.ndarray:
    """Expected Shortfall loss ell(x) = x_+ / (1 - p), p in (0,1) (paper Example 2)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return np.maximum(x, 0.0) / (1.0 - p)


def loss_entropic(x: np.ndarray, gamma: float) -> np.ndarray:
    """Entropic loss ell(x) = (exp(gamma*x) - 1) / gamma, gamma > 0 (paper Example 3, Eq. 3)."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return (np.exp(gamma * x) - 1.0) / gamma


def loss_chi(x: np.ndarray) -> np.ndarray:
    """Nondecreasing, nonconvex loss ell(x) = x_+ + 1{x>0} generating chi in Proposition 3."""
    return np.maximum(x, 0.0) + (x > 0).astype(float)


def oce(
    samples: np.ndarray,
    loss_fn: LossFn,
    m_bounds: tuple[float, float] | None = None,
) -> float:
    """Optimized Certainty Equivalent rho_ell(X) = inf_m {m + E[ell(X-m)]} (Eq. 1).

    Parameters
    ----------
    samples: an empirical sample (or exact support with equal weights folded
        in by the caller) representing the loss X.
    loss_fn: a convex loss function ell as in `loss_*` above, applied elementwise.
    m_bounds: search bounds for m; defaults to a wide margin around the sample
        range, which is enough since the optimal m lies near the support of X
        for every loss function used in this paper.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        raise ValueError("samples must be non-empty")
    if m_bounds is None:
        spread = float(np.max(np.abs(samples))) + 1.0
        m_bounds = (float(samples.min()) - 20.0 * spread, float(samples.max()) + 20.0 * spread)

    def objective(m: float) -> float:
        return m + float(np.mean(loss_fn(samples - m)))

    result = minimize_scalar(objective, bounds=m_bounds, method="bounded",
                              options={"xatol": 1e-10})
    return float(result.fun)
