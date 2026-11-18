# An axiomatic foundation for optimized certainty equivalents

Replication of selected results from Blier-Wong, Lauzier and Van Oosten,
"An axiomatic foundation for optimized certainty equivalents" (2026).

This repository verifies, by numerical experiment on synthetic random
variables, the axiomatic characterizations the paper gives for the
Optimized Certainty Equivalent (OCE)

    rho_ell(X) = inf_{m in R} { m + E[ell(X - m)] }

and for the specific risk measures it builds from it: the Mean-Variance
criterion, Expected Shortfall, the Entropic Risk Measure, and Value-at-Risk.
No market data is used: the paper works on an abstract atomless probability
space, and every experiment constructs explicit random variables (normal,
uniform, Bernoulli, discrete) to test an axiom or a counterexample.

## Layout

- `src/` -- the OCE and risk-measure implementations and the axiom checks
  shared by every experiment (`oce.py`, `risk_measures.py`, `axioms.py`,
  `config.py`).
- `data/generate_synthetic.py` -- the random-variable generators used by the
  experiments (no market data is used anywhere in this replication).
- `exp/` -- one script per experiment, run as `python -m exp.exp_N`.
- `tests/` -- unit tests of `src/` and of each experiment.
- `results/` -- tables, plots and `RESULTS.md` written by the experiments.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Experiments

```bash
python -m exp.exp_1   # Theorem 1: SRA, Normalization, Fair-Coin Compensation
python -m exp.exp_2   # Proposition 3: independence of SRA, FC, N, M
python -m exp.exp_3   # Theorem 3: Mean-Variance and Equal-Mean Randomization
python -m exp.exp_4   # Proposition 4 / Corollary 1: Expected Shortfall, PH and Prudence
python -m exp.exp_5   # Proposition 5: Entropic Risk Measure and Fair-Coin Independence
python -m exp.exp_6   # Proposition 2: VaR violates Fair-Coin Compensation
python -m exp.exp_7   # Example 4 / Figure 2: fairness premium of ES and its average
```

Or run everything with `./reproduce.sh`.

Every run overwrites the tables and plots in `results/`. See
`results/RESULTS.md` for the comparison against the paper's reported values,
deviations, and what is not reproduced.
