"""Experiment 7: Fairness-premium analysis for Expected Shortfall and their
average (Example 4, Figure 2).

Reproduces the paper's Example 4: X = 0 (a.s.), Y ~ Uniform{0,1,2}. For each
risk measure rho in {ES_{1/2}, ES_{3/4}, xi = (1/2)ES_{1/2}+(1/2)ES_{3/4}} the
fairness premium

    pi_rho(a) = 2*rho(W_1(a)) - rho(X) - rho(Y),   W_1(a) = B(X+a)+(1-B)(Y-a),

is computed exactly (no Monte Carlo: X and Y have finite support, so the law
of the fair-coin mixture W_1(a) is exact, Sec. 3.2) over a grid of transfers
a, reproducing the curves and table of Figure 2 (paper Sec. 3.2).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.risk_measures import exact_es_discrete, mix_discrete_fair_coin

X_VALUES = np.array([0.0])  # X = 0 a.s.
X_PROBS = np.array([1.0])


def es_fairness_premium(p: float, a: float) -> dict:
    """pi_{ES_p}(a) = 2*ES_p(W_1(a)) - ES_p(X) - ES_p(Y) (paper Eq. 2, Example 4)."""
    es_x = exact_es_discrete(X_VALUES, X_PROBS, p)
    es_y = exact_es_discrete(config.EXP7_Y_VALUES, config.EXP7_Y_PROBS, p)
    w_values, w_probs = mix_discrete_fair_coin(X_VALUES, X_PROBS,
                                                config.EXP7_Y_VALUES, config.EXP7_Y_PROBS, a)
    es_w = exact_es_discrete(w_values, w_probs, p)
    premium = 2.0 * es_w - es_x - es_y
    return {"es_X": es_x, "es_Y": es_y, "es_W": es_w, "premium": premium}


def premium_curve(p: float, a_grid: np.ndarray) -> np.ndarray:
    """pi_{ES_p}(a) over a grid of transfers."""
    return np.array([es_fairness_premium(p, float(a))["premium"] for a in a_grid])


def xi_premium_curve(a_grid: np.ndarray) -> np.ndarray:
    """pi_xi(a) = (1/2)*pi_{ES_{1/2}}(a) + (1/2)*pi_{ES_{3/4}}(a), by linearity of
    xi = (1/2)ES_{1/2} + (1/2)ES_{3/4} in rho (paper Example 4 / Proposition 3)."""
    p1, p2 = config.EXP7_ES_LEVELS
    return 0.5 * premium_curve(p1, a_grid) + 0.5 * premium_curve(p2, a_grid)


def build_table(a_values: list[float]) -> pd.DataFrame:
    """Reproduces the table of Figure 2: premiums of ES_{1/2}, ES_{3/4} and xi
    at the transfers a = 0, 1/2, 1, 3/2 (paper Sec. 3.2)."""
    p1, p2 = config.EXP7_ES_LEVELS
    rows = []
    for a in a_values:
        prem_1 = es_fairness_premium(p1, a)["premium"]
        prem_2 = es_fairness_premium(p2, a)["premium"]
        prem_xi = 0.5 * prem_1 + 0.5 * prem_2
        rows.append({
            "a": a,
            f"pi_ES_{p1}": prem_1,
            f"pi_ES_{p2}": prem_2,
            "pi_xi": prem_xi,
        })
    return pd.DataFrame(rows)


def plot_curves(a_grid: np.ndarray, path: str) -> None:
    """Plots the fairness-premium curves, mirroring the paper's Figure 2."""
    p1, p2 = config.EXP7_ES_LEVELS
    curve_1 = premium_curve(p1, a_grid)
    curve_2 = premium_curve(p2, a_grid)
    curve_xi = xi_premium_curve(a_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(a_grid, curve_1, label=f"ES_{{{p1}}}", color="tab:blue")
    ax.plot(a_grid, curve_2, label=f"ES_{{{p2}}}", color="tab:orange")
    ax.plot(a_grid, curve_xi, label="xi = (1/2)ES_{1/2} + (1/2)ES_{3/4}", color="tab:green")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("transfer a")
    ax.set_ylabel("fairness premium pi_rho(a)")
    ax.set_title("Fairness premium of Expected Shortfall and its average (Example 4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p1, p2 = config.EXP7_ES_LEVELS

    es_y_1 = exact_es_discrete(config.EXP7_Y_VALUES, config.EXP7_Y_PROBS, p1)
    es_y_2 = exact_es_discrete(config.EXP7_Y_VALUES, config.EXP7_Y_PROBS, p2)
    xi_y = 0.5 * es_y_1 + 0.5 * es_y_2

    table = build_table(config.EXP7_TABLE_TRANSFERS)
    table.to_csv("results/exp7_fairness_premium_table.csv", index=False)

    a_grid = config.EXP7_PLOT_TRANSFER_GRID
    curve_1 = premium_curve(p1, a_grid)
    curve_2 = premium_curve(p2, a_grid)
    curve_xi = xi_premium_curve(a_grid)
    curves_df = pd.DataFrame({
        "a": a_grid,
        f"pi_ES_{p1}": curve_1,
        f"pi_ES_{p2}": curve_2,
        "pi_xi": curve_xi,
    })
    curves_df.to_csv("results/exp7_fairness_premium_curves.csv", index=False)
    plot_curves(a_grid, "results/exp7_fairness_premium.png")

    min_1, a_star_1 = float(np.min(curve_1)), float(a_grid[np.argmin(curve_1)])
    min_2, a_star_2 = float(np.min(curve_2)), float(a_grid[np.argmin(curve_2)])
    min_xi, a_star_xi = float(np.min(curve_xi)), float(a_grid[np.argmin(curve_xi)])

    summary = {
        "ES_Y": {str(p1): es_y_1, str(p2): es_y_2},
        "xi_Y": xi_y,
        "min_premium_ES_1_2": min_1,
        "a_star_ES_1_2": a_star_1,
        "min_premium_ES_3_4": min_2,
        "a_star_ES_3_4": a_star_2,
        "min_premium_xi": min_xi,
        "a_star_xi": a_star_xi,
        "ES_1_2_reaches_zero": bool(abs(min_1) < 1e-6),
        "ES_3_4_reaches_zero": bool(abs(min_2) < 1e-6),
        "xi_premium_strictly_positive": bool(min_xi > 1e-6),
    }
    with open("results/exp7_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("exp_7 summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
