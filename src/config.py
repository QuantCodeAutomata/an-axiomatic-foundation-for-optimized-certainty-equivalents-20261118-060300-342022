"""Experiment parameters for the replication of
'An axiomatic foundation for optimized certainty equivalents'
(Blier-Wong, Lauzier and Van Oosten, 2026).

All random-variable constructions, loss-function parameters, and grids used
by the experiments in exp/ are collected here as module-level variables, so
that every run is reproducible from a single source of parameters.
"""

import numpy as np

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
RANDOM_SEED = 20261118  # base seed for every experiment's random generator
N_SAMPLES = 200_000  # Monte Carlo sample size for continuous-law checks (paper Sec. 2: data_requirements asks for >=100,000)
TOLERANCE = 1e-3  # numerical tolerance for axiom checks on Monte Carlo quantities (not specified in the paper)
EXACT_TOLERANCE = 1e-9  # tolerance for checks computable in closed form (e.g. Normalization)
MC_PREMIUM_TOL = 0.05  # tolerance for inf-premium (Fair-Coin Compensation/Independence) under Monte Carlo sampling noise of order 1/sqrt(N_SAMPLES); not specified in the paper

# ---------------------------------------------------------------------------
# exp_1: Theorem 1 -- SRA, Normalization, Fair-Coin Compensation
# ---------------------------------------------------------------------------
EXP1_KAPPA_VALUES = [0.0, 0.1, 0.5, 2.0]  # Mean-Variance loss ell(x)=x+kappa*x^2, kappa>=0 (paper Example 1)
EXP1_ES_LEVELS = [0.5, 0.75, 0.9, 0.99]  # Expected Shortfall levels p in (0,1) (paper Example 2)
EXP1_ENTROPIC_GAMMAS = [0.01, 0.1, 1.0, 5.0]  # Entropic Risk Measure parameters gamma>0 (paper Example 3, Eq. 3)
EXP1_NORMALIZATION_CONSTANTS = [0.0, 1.0, -1.0, 5.0]  # constants c tested for rho(c)=c (Axiom N)
EXP1_SRA_BERNOULLI_P = [0.1, 0.3, 0.5, 0.7, 0.9]  # p for Y~Bernoulli(p), X=E[Y]=p (constant), used for SRA check
EXP1_FC_TRANSFER_GRID = np.linspace(-10.0, 10.0, 401)  # grid of transfers a in Axiom (FC); range/resolution not specified in the paper
EXP1_FC_PAIR_NORMAL_MEANS = (0.0, 1.0)  # means of X, Y ~ Normal for the Fair-Coin Compensation check
EXP1_FC_PAIR_NORMAL_STDS = (1.0, 2.0)  # std devs of X, Y ~ Normal for the Fair-Coin Compensation check
# Families excluded from the Fair-Coin Compensation Monte Carlo check: at
# gamma=5 the entropic loss exp(gamma*Y) for Y ~ Normal(1, 2) has such a heavy
# right tail (Var(exp(gamma*Y)) grows as exp(2*gamma^2*std_Y^2)) that its
# sample mean is numerically unstable at the sample size used throughout this
# experiment; Normalization and SRA for this family do not involve that tail
# and still hold (not a paper-specified exclusion).
EXP1_FC_EXCLUDE_FAMILIES = ["ER_gamma=5.0"]
EXP1_CONVERGENCE_SAMPLE_SIZES = [1_000, 10_000, 100_000, 500_000]  # sample sizes used to check convergence of the axiom checks

# ---------------------------------------------------------------------------
# exp_2: Proposition 3 -- independence of SRA, FC, N, M
# ---------------------------------------------------------------------------
EXP2_XI_ES_LEVELS = (0.5, 0.75)  # xi = (1/2) ES_{1/2} + (1/2) ES_{3/4} (paper Example 4 / Proposition 3)
EXP2_XI_Y_VALUES = np.array([0.0, 1.0, 2.0])  # support of Y ~ Uniform{0,1,2}
EXP2_XI_Y_PROBS = np.array([1 / 3, 1 / 3, 1 / 3])  # uniform probabilities on {0,1,2}
EXP2_XI_FC_TRANSFER_GRID = np.linspace(-1.0, 2.5, 701)  # transfer grid a for the FC violation of xi (paper Fig. 2 covers [0, 3/2])
EXP2_CHI_P_E1 = 0.5  # P(E_1) in the chi / SRA counterexample (paper Appendix A, proof of Prop. 3)
EXP2_CHI_P_E2 = 0.25  # P(E_2) with E_2 subset E_1, in the chi / SRA counterexample
EXP2_CHI_M_GRID = np.linspace(-2.0, 4.0, 1201)  # grid of m for the piecewise-linear minimization defining chi
EXP2_2E_NORMAL_PARAMS = ((0.0, 1.0), (0.5, 2.0))  # (mean, std) of X, Y used to test 2E = 2*E[.]
EXP2_MV_KAPPA = 1.0  # kappa in the Mean-Variance counterexample rho_MV(X) = E[X] + kappa*Var(X)
EXP2_MV_X_VALUES = np.array([0.0, 4.0])  # support of X ~ {0,4} with equal probability, used to violate Monotonicity
EXP2_MV_X_PROBS = np.array([0.5, 0.5])

# ---------------------------------------------------------------------------
# exp_3: Theorem 3 -- Mean-Variance and Equal-Mean Randomization
# ---------------------------------------------------------------------------
EXP3_KAPPA_VALUES = [0.01, 0.1, 1.0, 10.0]  # kappa values tested for the Mean-Variance criterion (paper Sec. 4.1)
EXP3_EQUAL_MEAN = 0.0  # common mean mu used to build equal-mean pairs (X, Y)
EXP3_NORMAL_STDS = [(1.0, 3.0), (0.5, 2.5)]  # (std_X, std_Y) pairs for Normal(mu, std) with equal mean
EXP3_UNIFORM_HALF_WIDTHS = [(1.0, 4.0), (2.0, 6.0)]  # (a, b) half-widths for Uniform(-a,a) vs Uniform(-b,b), equal mean 0
EXP3_DISCRETE_X_VALUES = np.array([0.0, 4.0])  # X ~ {0,4} with equal probability (mean 2)
EXP3_DISCRETE_X_PROBS = np.array([0.5, 0.5])
EXP3_DISCRETE_Y_CONSTANT = 2.0  # Y = 2 a.s. (same mean as X), used as a degenerate equal-mean comparison
EXP3_MONOTONICITY_KAPPA = 1.0  # kappa > 0 used to exhibit the Monotonicity violation of Mean-Variance

# ---------------------------------------------------------------------------
# exp_4: Proposition 4 / Corollary 1 -- Expected Shortfall, PH and Prudence
# ---------------------------------------------------------------------------
EXP4_ES_LEVELS = [0.5, 0.75, 0.9, 0.99]  # levels p for ES_p used in the Positive Homogeneity check
EXP4_PH_SCALARS = [0.5, 2.0, 10.0]  # scalars t > 0 tested for ES_p(tX) = t * ES_p(X)
EXP4_MIXTURE_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]  # lambda in [0,1] for the mixture lambda*E + (1-lambda)*ES_p
EXP4_MIXTURE_P = 0.9  # ES level p used inside the lambda-mixture
EXP4_PRUDENCE_N_VALUES = [2, 5, 10, 50, 200, 1_000]  # sequence length n for X_n = X + Z/n -> X a.s.
EXP4_PRUDENCE_Z_BOUND = 3.0  # bound of the perturbation Z ~ Uniform(-bound, bound) used to build X_n
EXP4_BASE_NORMAL_MEAN = 0.0  # mean of the base loss X used throughout exp_4
EXP4_BASE_NORMAL_STD = 1.0  # std of the base loss X used throughout exp_4
EXP4_FATOU_N_VALUES = [10, 100, 1_000, 10_000, 100_000]  # n for the Fatou-type X_n = -n*1{U<1/n} -> 0 a.s.
EXP4_FATOU_SAMPLE_SIZE = 2_000_000  # Monte Carlo sample size resolving P(U<1/n) for the largest n above

# ---------------------------------------------------------------------------
# exp_5: Proposition 5 -- Entropic Risk Measure and Fair-Coin Independence
# ---------------------------------------------------------------------------
EXP5_GAMMAS = [0.01, 0.1, 1.0, 10.0]  # gamma values for the Entropic Risk Measure ER_gamma, gamma=0 is the expectation
EXP5_FCI_NORMAL_PARAMS = {  # (mean, std) of X, Y, Z used to test Fair-Coin Independence
    "X": (0.0, 1.0),
    "Y": (0.5, 1.0),
    "Z": (0.2, 1.5),
}
EXP5_ES_LEVEL_FOR_VIOLATION = 0.75  # ES level used in the counterexample showing ES violates Fair-Coin Independence
EXP5_FCI_COUNTEREXAMPLE_Y_VALUES = np.array([0.0, 1.0, 3.0])  # support of Y used in the ES/FCI counterexample
EXP5_FCI_COUNTEREXAMPLE_Y_PROBS = np.array([0.25, 0.5, 0.25])
EXP5_FCI_COUNTEREXAMPLE_X_VALUE = 0.9  # constant X compared against Y above (same ES_p up to the chosen level)
EXP5_FCI_COUNTEREXAMPLE_Z_VALUES = np.array([-2.0, 5.0])  # support of the common third loss Z
EXP5_FCI_COUNTEREXAMPLE_Z_PROBS = np.array([0.9, 0.1])

# ---------------------------------------------------------------------------
# exp_6: Proposition 2 -- VaR violates Fair-Coin Compensation
# ---------------------------------------------------------------------------
EXP6_P_VALUES = [0.5, 0.75, 0.9, 0.95, 0.99]  # VaR levels p in (0,1) tested (paper Proposition 2)
EXP6_SAMPLE_SIZE = 2_000_000  # large sample size for the Bernoulli(p^2) Monte Carlo illustration

# ---------------------------------------------------------------------------
# exp_7: Example 4 / Figure 2 -- fairness premium of ES and its average
# ---------------------------------------------------------------------------
EXP7_Y_VALUES = np.array([0.0, 1.0, 2.0])  # Y ~ Uniform{0,1,2} (paper Example 4)
EXP7_Y_PROBS = np.array([1 / 3, 1 / 3, 1 / 3])
EXP7_ES_LEVELS = (0.5, 0.75)  # ES_{1/2}, ES_{3/4} as in Figure 2
EXP7_TABLE_TRANSFERS = [0.0, 0.5, 1.0, 1.5]  # transfer values a reported in the paper's Figure 2 table
EXP7_PLOT_TRANSFER_GRID = np.linspace(-0.5, 2.0, 251)  # dense grid of a for plotting the fairness-premium curves
