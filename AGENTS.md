# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A numerical replication of selected results from Blier-Wong, Lauzier and
Van Oosten, "An axiomatic foundation for optimized certainty equivalents"
(2026). The paper characterizes the Optimized Certainty Equivalent (OCE)

    rho_ell(X) = inf_{m in R} { m + E[ell(X - m)] }

by the axioms Strong Risk Aversion, Normalization and Fair-Coin Compensation
(Theorem 1), shows the independence of these axioms plus Monotonicity
(Proposition 3), and derives Mean-Variance, Expected Shortfall and the
Entropic Risk Measure from additional axioms (Theorem 3, Proposition 4 /
Corollary 1, Proposition 5), together with the Value-at-Risk counterexample
(Proposition 2) and the fairness-premium illustration of Example 4 / Figure 2.
Every experiment works on synthetic random variables; the paper is purely
theoretical and uses no market data.

## Layout

- `src/` -- shared library code: `oce.py` (OCE minimization and loss
  functions, Eq. 1), `risk_measures.py` (closed forms for Mean-Variance,
  Expected Shortfall, the Entropic Risk Measure, VaR, `chi`, and the mixture
  `xi`), `axioms.py` (Normalization, Strong Risk Aversion, convex order,
  Fair-Coin Compensation/Independence, Monotonicity, Positive Homogeneity,
  Prudence checks), `config.py` (every experiment parameter as a module-level
  variable, grouped by experiment, with a source comment).
- `data/generate_synthetic.py` -- the only place synthetic random variables
  are generated (normal, uniform, Bernoulli, discrete laws, fair coins).
- `exp/exp_1.py` .. `exp/exp_7.py` -- one script per experiment, each calling
  only `src/` and `data/generate_synthetic.py`, run as `python -m exp.exp_N`.
- `tests/` -- pytest tests of `src/` (fixtures and synthetic inputs) and of
  each experiment's distinctive code path.
- `results/` -- every table (`*.csv`), plot (`*.png`) and summary (`*.json`)
  written directly by the experiment scripts, plus `results/RESULTS.md` with
  the comparison against the paper, deviations, and what is not reproduced.

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest
python -m pytest -m "not market_data"   # equivalent here: no test is marked market_data
```

Experiments (paper configuration, no reduction was needed for any
experiment in this replication since none depends on market data or a
download):

```bash
python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
python -m exp.exp_5
python -m exp.exp_6
python -m exp.exp_7
```

Or `./reproduce.sh` to run the tests and every experiment in one command.

## Conventions

- Every experiment parameter (sample sizes, loss-function constants, levels,
  transfer grids, random seed) lives in `src/config.py` as an
  `UPPER_CASE` module-level variable with a one-line comment giving its
  meaning, unit, and source (a paper section/equation, or "not specified in
  the paper"). No parameter is hardcoded inside `exp/` or `src/` scripts.
- All synthetic data generation goes through `data/generate_synthetic.py`;
  no experiment script or `src/` module draws random variables on its own.
- Each `exp/exp_N.py` only calls functions exposed by `src/` and
  `data/generate_synthetic.py`, and writes its outputs directly under
  `results/` (no per-experiment subdirectories).
- Tests are plain pytest functions in `tests/`, run on synthetic fixtures
  (no live network calls anywhere in this repository, since no experiment
  uses market data).
- API keys are not used by this repository; there is no `.env.example`
  because no experiment loads market data.
- Running an experiment overwrites its files in `results/`. After a rerun,
  update the numbers, verdicts and any changed values in
  `results/RESULTS.md` by hand -- they are not regenerated automatically.
