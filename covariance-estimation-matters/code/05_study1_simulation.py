# -*- coding: utf-8 -*-
"""Study 1: controlled simulation of reference covariance estimation.

Eleven conditions x four reference sample sizes x 30 seeds = 1,320 fits per
estimator, comparing the sample covariance, Ledoit-Wolf shrinkage and MCD at
scikit-learn defaults against a covariance the design fixes rather than
estimates.

Conditions apply a marginal transform to the six symptom features and, in three
cases, add contamination: gaussian, skew, floor10/25/50/75, gaussian_contam05,
gaussian_contam10, floor50_contam05, and two matched conditions built from the
Study 2 control sample. Reference sizes are 62, 200, 1000 and 5000; the deviant
group is held constant within a seed. All distances are taken about the
reference-sample mean.

The primary outcome is the across-seed median of the within-seed median relative
squared-distance error for the deviant group. Covariance and precision-matrix
errors are secondary, and precision error is summarized by medians because
near-singular fits produce extreme values.

Reading the calibration output: compare each estimator against the ORACLE column
for its condition, not against 1.00 and 5%. Chi-square is the right reference
only where the marginals are normal. Under a .50 floor the oracle itself sits at
0.88 with 11.6% exceedance, so an estimator reporting 0.88 and 11.6% is exactly
right and one reporting 1.00 and 5% would be wrong.

This script reads no participant data. The matched conditions use aggregates
recorded below: floor prevalences observed in the Study 2 control sample, and a
latent correlation matrix derived in 04_invert_r.py.

Post hoc additions: docs/study1_amendment_2026-09-17.md and CHANGELOG.md.

Writes  Study1_runs.csv, Study1_summary.csv, Study1_calibration.csv,
        Fig_study1, FigS_study1_ranking, FigS_study1_error_by_N,
        versions_study1.json

Usage   python code/05_study1_simulation.py [--base PATH] [--quick]
        Full run takes about eight minutes; --quick is a code check, not results.
"""

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import linalg, stats
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# parents[1] when run as a file; the working directory when pasted into an
# interactive session, where __file__ does not exist
DEFAULT_BASE = (Path(__file__).resolve().parents[1]
                if "__file__" in globals() else Path.cwd())
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=DEFAULT_BASE)
ap.add_argument("--quick", action="store_true", help="3 seeds, 2 sizes (code check only)")
args = ap.parse_args()
OUT = Path(args.base) / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Design
# ------------------------------------------------------------------
P, P_SYM = 11, 6

# Chi-square reference for in-sample calibration of the reference group. If the
# reference features were multivariate normal and the covariance were known, the
# squared distances would follow chi-square on P degrees of freedom, and the
# nominal exceedance rates would be 5% and 2.5%. The gaussian condition shows how
# far each estimator sits from that with the covariance ESTIMATED at a given n,
# which is the baseline any real-data departure has to be read against. The floor
# conditions show what happens on top of that when the marginals are not normal.
CHI2_Q50, CHI2_Q95, CHI2_Q975 = stats.chi2.ppf([0.50, 0.95, 0.975], P)
N_VALUES = [62, 200, 1000, 5000]
N_SEEDS = 30
SEED_OFFSET = 1000
N_DEV = 263
TARGET_MC = 1_000_000
TOP_Q = 0.90
SKEW_SIGMA = 0.5
CONTAM_SHIFT = 3.0
DEV_SHIFT = 2.0
if args.quick:
    N_VALUES, N_SEEDS, TARGET_MC = [62, 1000], 3, 200_000

# Floor prevalence observed in the Study 2 healthy-control sample, in the order
# the symptom block is generated. Source: TableS_feature_descriptives_by_group.
# gBDI .387, SHAPS .935, PSC .290, DOCS .452, OCTCDQ incompleteness .226,
# OCTCDQ harm avoidance .306. The five personality features carry no floor, which
# the design already reflects by flooring only the first P_SYM columns.
OBSERVED_PI = [0.387, 0.935, 0.290, 0.452, 0.226, 0.306]

CONDITIONS = {                     # name: (marginal, floor_pi, eps)
                                   # floor_pi is a scalar OR a list of P_SYM
    'gaussian':          ('gaussian', 0.00, 0.00),
    'skew':              ('skew',     0.00, 0.00),
    'floor10':           ('floor',    0.10, 0.00),
    'floor25':           ('floor',    0.25, 0.00),
    'floor50':           ('floor',    0.50, 0.00),
    'floor75':           ('floor',    0.75, 0.00),
    'gaussian_contam05': ('gaussian', 0.00, 0.05),
    'gaussian_contam10': ('gaussian', 0.00, 0.10),
    'floor50_contam05':  ('floor',    0.50, 0.05),
    # Appended last on purpose: COND_ID is positional and seeds every RNG stream,
    # so inserting earlier would change every previously reported number.
    'floor_matched':     ('floor', OBSERVED_PI, 0.00),
    'floor_matched_corr': ('floor', OBSERVED_PI, 0.00),   # + R_MATCHED, see CONDITION_R
}
# Latent correlation implied by the Study 2 control sample, obtained by inverting
# its observed correlations through the censoring map at OBSERVED_PI and taking
# the nearest positive-definite matrix (min eigenvalue of the raw inversion was
# -0.129). Derivation in invert_r.py. Feature order matches OBSERVED_PI.
R_MATCHED = np.array([
    [ 1.0000,  0.3316,  0.6067,  0.5087,  0.3841,  0.4406,  0.4919, -0.2409, -0.2715, -0.2283, -0.1529],
    [ 0.3316,  1.0000,  0.2667, -0.0470, -0.2068,  0.0225,  0.2951, -0.7049, -0.3570, -0.7330, -0.3843],
    [ 0.6067,  0.2667,  1.0000,  0.5855,  0.5179,  0.4844,  0.4270, -0.3080, -0.4182, -0.2656, -0.1905],
    [ 0.5087, -0.0470,  0.5855,  1.0000,  0.7120,  0.6994,  0.4248, -0.2057, -0.1893, -0.2720, -0.0489],
    [ 0.3841, -0.2068,  0.5179,  0.7120,  1.0000,  0.8427,  0.4736, -0.2043, -0.1186, -0.2272, -0.2577],
    [ 0.4406,  0.0225,  0.4844,  0.6994,  0.8427,  1.0000,  0.4778, -0.2457, -0.1575, -0.2453, -0.2072],
    [ 0.4919,  0.2951,  0.4270,  0.4248,  0.4736,  0.4778,  1.0000, -0.3088, -0.3971, -0.5613, -0.4641],
    [-0.2409, -0.7049, -0.3080, -0.2057, -0.2043, -0.2457, -0.3088,  1.0000,  0.4045,  0.5707,  0.5136],
    [-0.2715, -0.3570, -0.4182, -0.1893, -0.1186, -0.1575, -0.3971,  0.4045,  1.0000,  0.3136,  0.2631],
    [-0.2283, -0.7330, -0.2656, -0.2720, -0.2272, -0.2453, -0.5613,  0.5707,  0.3136,  1.0000,  0.5257],
    [-0.1529, -0.3843, -0.1905, -0.0489, -0.2577, -0.2072, -0.4641,  0.5136,  0.2631,  0.5257,  1.0000],
])

# Per-condition latent correlation. Anything absent uses the design default R.
CONDITION_R = {'floor_matched_corr': R_MATCHED}

ESTIMATORS = ['sample', 'ledoit_wolf', 'mcd']
COND_ID = {c: i for i, c in enumerate(CONDITIONS)}   # fixed ids for RNG streams
MARGINAL_ID = {'gaussian': 0, 'skew': 1, 'floor': 2}

R = np.full((P, P), 0.15)
R[:P_SYM, :P_SYM] = 0.50
R[P_SYM:, P_SYM:] = 0.20
np.fill_diagonal(R, 1.0)
assert np.linalg.eigvalsh(R).min() > 0
L = np.linalg.cholesky(R)

(OUT / "versions_study1.json").write_text(json.dumps({
    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__, "n_values": N_VALUES, "n_seeds": N_SEEDS,
    "seed_offset": SEED_OFFSET, "target_mc": TARGET_MC,
    "conditions": {k: [m, (p if np.isscalar(p) else list(p)), e]
                   for k, (m, p, e) in CONDITIONS.items()},
    "observed_pi": OBSERVED_PI,
    "condition_latent_R": {k: v.round(4).tolist() for k, v in CONDITION_R.items()},
    "latent_correlation": R.round(3).tolist()}, indent=2))


def chol_for(cond_name):
    """Cholesky factor of the condition's latent correlation, default design R."""
    Rc = CONDITION_R.get(cond_name)
    return L if Rc is None else np.linalg.cholesky(Rc)


def latent(n, rng, shift=0.0, eps=0.0, chol=None):
    """Latent draw, plus the mask of rows that were contaminated.

    The mask is returned so the reference-group exceedance can be computed with
    contaminated observations excluded. The RNG is consumed in exactly the order
    it was before, so every previously reported number is unaffected.
    """
    Z = rng.standard_normal((n, P)) @ (L if chol is None else chol).T
    bad = np.zeros(n, bool)
    if shift:
        Z[:, :P_SYM] += shift
    if eps > 0:
        bad = rng.random(n) < eps
        Z[bad, :P_SYM] += CONTAM_SHIFT
    return Z, bad


def pi_vector(pi):
    """A per-feature floor proportion, whether pi was given as a scalar or a list."""
    return np.full(P_SYM, float(pi)) if np.isscalar(pi) else np.asarray(pi, float)


def pi_key(pi):
    """Hashable cache key. Scalars keep their original form so cached targets
    and seeds are identical to earlier runs."""
    return float(pi) if np.isscalar(pi) else tuple(float(v) for v in pi)


def pi_seed(pi):
    """Deterministic integer for the target-sample RNG.

    Scalars use the original expression so every previously reported condition
    reproduces exactly. Vectors take a disjoint range."""
    if np.isscalar(pi):
        return int(round(float(pi) * 100))
    return 10_000 + int(round(float(np.sum(pi)) * 1000))


def transform(Z, marginal, pi):
    X = Z.copy()
    floor_flag = np.zeros((len(Z), P_SYM), bool)
    if marginal == 'skew':
        X[:, :P_SYM] = np.exp(SKEW_SIGMA * Z[:, :P_SYM])
    elif marginal == 'floor':
        # Broadcasts over columns, so a per-feature vector needs no other change.
        c = stats.norm.ppf(pi_vector(pi))
        floor_flag = Z[:, :P_SYM] <= c
        X[:, :P_SYM] = np.maximum(Z[:, :P_SYM], c)
    return X, floor_flag


# Population moments and targets for each clean marginal (cached)
_targets = {}


def target(marginal, pi, cond_name=None):
    # The structure is part of the population, so a condition with its own
    # correlation matrix needs its own target and its own RNG stream.
    custom = cond_name in CONDITION_R
    key = (marginal, pi_key(pi), cond_name if custom else None)
    if key not in _targets:
        rng = np.random.default_rng([SEED_OFFSET, 999, MARGINAL_ID[marginal],
                                     pi_seed(pi) + (500_000 if custom else 0)])
        Zt, _ = latent(TARGET_MC, rng, chol=chol_for(cond_name))
        X, _ = transform(Zt, marginal, pi)
        mu, sd = X.mean(0), X.std(0)
        S = (X - mu) / sd
        cov = np.cov(S, rowvar=False, bias=True)
        _targets[key] = dict(mu=mu, sd=sd, cov=cov, prec=linalg.pinvh(cov))
    return _targets[key]


def d2(Z, center, Pm):
    D = Z - center
    return np.einsum('ij,jk,ik->i', D, Pm, D)


def fit(name, X, seed):
    if name == 'sample':
        return EmpiricalCovariance().fit(X), 0
    if name == 'ledoit_wolf':
        return LedoitWolf().fit(X), 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            est = MinCovDet(random_state=seed).fit(X)
        except ValueError:
            est = None
    return est, len(w)


def run_cell(cond_name, seed):
    marginal, pi, eps = CONDITIONS[cond_name]
    ch = chol_for(cond_name)
    T = target(marginal, pi, cond_name)
    rng_dev = np.random.default_rng([seed, 1, COND_ID[cond_name]])
    Zd, _ = latent(N_DEV, rng_dev, shift=DEV_SHIFT, chol=ch)
    Xd, _ = transform(Zd, marginal, pi)
    dev = (Xd - T['mu']) / T['sd']
    d2_bench = d2(dev, 0.0, T['prec'])
    top_bench = d2_bench >= np.quantile(d2_bench, TOP_Q)
    cnorm, pnorm = np.linalg.norm(T['cov']), np.linalg.norm(T['prec'])
    rows = []
    for n in N_VALUES:
        rng = np.random.default_rng([seed, 2, n, COND_ID[cond_name]])
        Zr, contam = latent(n, rng, eps=eps, chol=ch)
        Xr, flag = transform(Zr, marginal, pi)
        X = (Xr - T['mu']) / T['sd']
        center = X.mean(0)
        clean = ~contam
        # The oracle. Reference distances formed with the BENCHMARK covariance,
        # centred the same way the estimators are, so the only difference from an
        # estimator row is the covariance itself. Whatever this produces is the
        # correct reference behaviour for the condition; chi-square is correct
        # only where the marginals are normal, which is the gaussian and
        # contamination conditions and none of the others.
        d_oracle = d2(X, center, T['prec'])
        row = dict(condition=cond_name, marginal=marginal, floor_pi=pi, eps=eps, seed=seed, n=n,
                   pct_contam=100 * contam.mean(),
                   oracle_ref_med_ratio=np.median(d_oracle) / CHI2_Q50,
                   oracle_ref_pct_above_q95=100 * np.mean(d_oracle > CHI2_Q95),
                   oracle_ref_pct_above_q975=100 * np.mean(d_oracle > CHI2_Q975),
                   oracle_ref_pct_above_q95_clean=100 * np.mean(d_oracle[clean] > CHI2_Q95))
        for e in ESTIMATORS:
            est, n_warn = fit(e, X, seed)
            if est is None:
                row[f'{e}_failed'] = True
                continue
            C = est.covariance_
            Pm, rank = linalg.pinvh(C, return_rank=True)
            ev = np.linalg.eigvalsh(C)
            dd = d2(dev, center, Pm)
            top = dd >= np.quantile(dd, TOP_Q)
            # In-sample calibration of the REFERENCE group, the Study 1 analogue of
            # the healthy-control distances in Study 2. Added post hoc; see
            # docs/study1_amendment_2026-09-17.md.
            d_ref = d2(X, center, Pm)
            row.update({
                f'{e}_failed': False,
                f'{e}_relfrob': np.linalg.norm(C - T['cov']) / cnorm,
                f'{e}_prec_relfrob': np.linalg.norm(Pm - T['prec']) / pnorm,
                f'{e}_rank': int(rank), f'{e}_singular': rank < P,
                f'{e}_cond': ev.max() / ev.min() if rank == P else np.inf,
                f'{e}_d2_relerr_med': np.median(np.abs(dd - d2_bench) / d2_bench),
                f'{e}_spearman': stats.spearmanr(dd, d2_bench).statistic,
                f'{e}_top_jaccard': (top & top_bench).sum() / (top | top_bench).sum(),
                f'{e}_dev_med': np.median(dd),
                f'{e}_ref_med_ratio': np.median(d_ref) / CHI2_Q50,
                f'{e}_ref_pct_above_q95': 100 * np.mean(d_ref > CHI2_Q95),
                f'{e}_ref_pct_above_q975': 100 * np.mean(d_ref > CHI2_Q975),
                # Exceedance over the uncontaminated part of the reference group.
                # The full-sample figure charges every estimator for the
                # contamination it was handed, which flatters none of them
                # equally: a robust fit leaves those rows far out by design.
                f'{e}_ref_pct_above_q95_clean': 100 * np.mean(d_ref[clean] > CHI2_Q95),
            })
            row[f'_{e}_cov'] = C
            if e == 'mcd':
                sup, raw = est.support_, est.raw_support_
                any_floor = flag.any(axis=1)
                row.update({
                    'mcd_warnings': n_warn,
                    'mcd_support': sup.mean(),
                    'mcd_own_center_dev_med': np.median(est.mahalanobis(dev)),
                    'mcd_pct_floor_retained': 100 * any_floor[sup].mean(),
                    'mcd_pct_floor_excluded': 100 * any_floor[~sup].mean() if (~sup).any() else np.nan,
                    'mcd_constant_feature_raw': bool((Xr[raw][:, :P_SYM].std(0) == 0).any()),
                    'mcd_constant_feature_reweighted': bool((Xr[sup][:, :P_SYM].std(0) == 0).any()),
                })
        rows.append(row)
    return rows


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
seeds = list(range(SEED_OFFSET, SEED_OFFSET + N_SEEDS))
t0 = time.time()
rows = []
for c in CONDITIONS:
    for s in seeds:
        rows += run_cell(c, s)
    print(f"  {c:<18} done ({time.time() - t0:.0f}s)")
runs = pd.DataFrame(rows)

# Mean-matrix bias per condition x N x estimator
bias_rows = []
for (c, n), g in runs.groupby(['condition', 'n']):
    marginal, pi, _ = CONDITIONS[c]
    T = target(marginal, pi, c)
    rec = dict(condition=c, n=n)
    for e in ESTIMATORS:
        mats = [m for m in g[f'_{e}_cov'] if isinstance(m, np.ndarray)]
        rec[f'{e}_meanmatrix_relbias'] = (np.linalg.norm(np.mean(mats, 0) - T['cov'])
                                          / np.linalg.norm(T['cov'])) if mats else np.nan
    bias_rows.append(rec)
runs = runs.drop(columns=[c for c in runs.columns if c.startswith('_')])
runs.to_csv(OUT / "Study1_runs.csv", index=False)


def mean_finite(x):
    return pd.to_numeric(x, errors='coerce').replace(np.inf, np.nan).mean()


def sd_finite(x):
    return pd.to_numeric(x, errors='coerce').replace(np.inf, np.nan).std()


agg = {}
for e in ESTIMATORS:
    for m in ['relfrob', 'prec_relfrob', 'cond', 'd2_relerr_med', 'spearman', 'top_jaccard',
              'dev_med', 'ref_med_ratio', 'ref_pct_above_q95', 'ref_pct_above_q975',
              'ref_pct_above_q95_clean']:
        agg[f'{e}_{m}_mean'] = (f'{e}_{m}', mean_finite)
        agg[f'{e}_{m}_sd'] = (f'{e}_{m}', sd_finite)
        agg[f'{e}_{m}_median'] = (f'{e}_{m}', lambda x: pd.to_numeric(x, errors='coerce').median())
    agg[f'{e}_pct_singular'] = (f'{e}_singular', lambda x: 100 * x.astype(float).mean())
    agg[f'{e}_pct_failed'] = (f'{e}_failed', lambda x: 100 * x.astype(float).mean())
for m in ['ref_med_ratio', 'ref_pct_above_q95', 'ref_pct_above_q975',
          'ref_pct_above_q95_clean']:
    agg[f'oracle_{m}_mean'] = (f'oracle_{m}', mean_finite)
    agg[f'oracle_{m}_sd'] = (f'oracle_{m}', sd_finite)
    agg[f'oracle_{m}_median'] = (f'oracle_{m}',
                                 lambda x: pd.to_numeric(x, errors='coerce').median())
agg['pct_contam_mean'] = ('pct_contam', mean_finite)
for c in ['mcd_support', 'mcd_own_center_dev_med', 'mcd_pct_floor_retained', 'mcd_pct_floor_excluded']:
    agg[f'{c}_mean'] = (c, mean_finite)
for c in ['mcd_constant_feature_raw', 'mcd_constant_feature_reweighted']:
    agg[f'{c}_pct'] = (c, lambda x: 100 * x.astype(float).mean())
agg['mcd_warning_fits_pct'] = ('mcd_warnings', lambda x: 100 * (x > 0).mean())
summary = (runs.groupby(['condition', 'n']).agg(**agg).reset_index()
           .merge(pd.DataFrame(bias_rows), on=['condition', 'n']))
order = {c: i for i, c in enumerate(CONDITIONS)}
summary = summary.sort_values(['condition', 'n'], key=lambda s: s.map(order) if s.name == 'condition' else s)
summary.round(4).to_csv(OUT / "Study1_summary.csv", index=False)
pd.set_option('display.width', 250)
print(summary[['condition', 'n'] + [f'{e}_d2_relerr_med_median' for e in ESTIMATORS] +
              [f'{e}_relfrob_mean' for e in ESTIMATORS] +
              [f'{e}_prec_relfrob_median' for e in ESTIMATORS] +
              ['mcd_pct_singular', 'mcd_top_jaccard_mean', 'mcd_pct_floor_retained_mean',
               'mcd_pct_floor_excluded_mean']].round(3).to_string(index=False))

# ------------------------------------------------------------------
# Reference-group calibration table
#
# The Study 1 analogue of the chi-square calibration in Study 2. The gaussian
# rows give the baseline: what each estimator does when its assumptions hold and
# the only departure from the chi-square reference comes from estimating the
# covariance at that n. Any real-data result has to be read against those rows,
# not against the nominal 5% and 2.5%, or the estimator's own small-sample
# behavior gets attributed to the data.
#
# Added post hoc, after the Study 2 chi-square table raised the question. See
# docs/study1_amendment_2026-09-17.md.
# ------------------------------------------------------------------
# Report the ratio as a MEDIAN across seeds. A near-singular fit pseudo-inverts to
# enormous distances, and one such seed drags a mean into the thousands while the
# typical seed is nowhere near it. The mean is kept in the CSV for completeness.
cal_cols = ['ref_med_ratio_median', 'ref_med_ratio_mean',
            'ref_pct_above_q95_mean', 'ref_pct_above_q975_mean',
            'ref_pct_above_q95_clean_mean']
# The oracle leads the table, because it is the standard the three estimators
# are being held to. Reading down a row, the question is which estimator lands
# on the oracle's value, not which lands on 1.00.
CAL_SERIES = ['oracle'] + ESTIMATORS
cal = summary[['condition', 'n'] + [f'{e}_{m}' for e in CAL_SERIES for m in cal_cols]].copy()
cal.round(4).to_csv(OUT / "Study1_calibration.csv", index=False)
show = ['condition', 'n'] + [f'{e}_{m}' for e in CAL_SERIES for m in
                             ('ref_med_ratio_median', 'ref_pct_above_q95_mean')]
print("\n--- Reference-group calibration ---")
print("COMPARE EACH ESTIMATOR AGAINST THE ORACLE COLUMN FOR ITS CONDITION,")
print("NOT AGAINST 1.00 AND 5%.")
print(f"The oracle forms the same distances with the BENCHMARK mean and covariance,")
print(f"so it is what an estimator that recovered the target exactly would produce.")
print(f"Chi-square({P}) is the right reference only where the marginals are normal,")
print(f"which is the gaussian and contamination conditions and none of the others:")
print(f"under a .50 floor the oracle itself sits near 0.88 with roughly 12% exceedance.")
print(f"(chi2 median = {CHI2_Q50:.3f}; nominal exceedance 5.0% and 2.5%.)")
print("Ratios are medians across seeds; the CSV also carries the means, which a")
print("single near-singular fit can inflate by orders of magnitude.")
print(cal[show].round(3).to_string(index=False))

gauss = cal[cal.condition == 'gaussian']
if len(gauss):
    print("\nGaussian condition by n (median ratio | % above q95). Here, and only")
    print("here, the oracle should itself sit at 1.00 and 5%, which is the check")
    print("that the oracle is doing what it claims to:")
    for _, r in gauss.iterrows():
        cells = "  ".join(f"{e}: {r[f'{e}_ref_med_ratio_median']:.2f} | "
                          f"{r[f'{e}_ref_pct_above_q95_mean']:.1f}%" for e in CAL_SERIES)
        print(f"  n = {int(r['n']):>5}   {cells}")
    print("\nIf MCD's departure shrinks toward the oracle as n grows, the small-sample")
    print("behavior is consistent with the asymptotic consistency correction. If it")
    print("does not, that explanation is wrong and the paper has to say something else.")

# How far each estimator sits from the oracle, which is the quantity the
# manuscript should report. Positive means the estimator over-flags.
print("\nExceedance MINUS the oracle's exceedance (percentage points).")
print("Zero is correct behaviour for the condition; positive over-flags.")
gap = cal[['condition', 'n']].copy()
for e in ESTIMATORS:
    gap[e] = (cal[f'{e}_ref_pct_above_q95_mean']
              - cal['oracle_ref_pct_above_q95_mean']).round(1)
print(gap.to_string(index=False))

print(f"Total {time.time() - t0:.0f}s")

# ==================================================================
# FIGURES
# ==================================================================
COL = {'mcd': "#D55E00", 'ledoit_wolf': "#0072B2", 'sample': "#009E73"}
LAB = {'mcd': 'MCD', 'ledoit_wolf': 'Ledoit-Wolf', 'sample': 'Sample covariance'}
MRK = {'mcd': 'o', 'ledoit_wolf': 's', 'sample': '^'}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight"})


def save(fig, stem):
    """Write PDF and 600-dpi PNG, reporting any fallback loudly.

    A silent fallback would leave a stale figure under the expected name while
    the new one lands under a timestamp, so the failure is printed.
    """
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        target = OUT / f"{stem}{ext}"
        try:
            fig.savefig(target, **kw)
        except OSError as exc:
            alt = OUT / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
            fig.savefig(alt, **kw)
            print(f"  ! could not write {target.name} ({exc.__class__.__name__}: {exc}).")
            print(f"    Saved as {alt.name}. Any existing {target.name} is STALE.")
    plt.close(fig)


floor_conds = ['gaussian', 'floor10', 'floor25', 'floor50', 'floor75']
pis = [CONDITIONS[c][1] * 100 for c in floor_conds]
n_small, n_large = N_VALUES[0], N_VALUES[-1]
get = lambda c, n, col: summary.loc[(summary.condition == c) & (summary.n == n), col].iloc[0]

# Line style carries reference sample size, colour and marker carry the estimator,
# so the same visual code reads the same way in every panel. The sample covariance
# is drawn widest and first: it and Ledoit-Wolf coincide under several conditions,
# and at equal width the later curve would hide the earlier one entirely.
NSTY = ((n_small, '--'), (n_large, '-'))
ELW = {'sample': 3.0, 'ledoit_wolf': 1.6, 'mcd': 1.6}
EMS = {'sample': 5.5, 'ledoit_wolf': 4.0, 'mcd': 4.0}


def by_floor(ax, col, logy=True):
    """One estimator curve per sample size, across the floor-prevalence series."""
    for e in ESTIMATORS:
        for n, ls in NSTY:
            ax.plot(pis, [get(c, n, col.format(e=e)) for c in floor_conds], ls=ls,
                    marker=MRK[e], ms=EMS[e], lw=ELW[e], color=COL[e])
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Reference observations at floor (%)')


# The oracle is the benchmark covariance's own reference distances. It is what
# each estimator should reproduce, and it is not 1.00 or 5% except where the
# marginals are normal.
ORACLE_COL, ORACLE_MRK, ORACLE_LAB = "#000000", 'D', 'Benchmark covariance'


def oracle_curve(ax, col):
    for n, ls in NSTY:
        ax.plot(pis, [get(c, n, col) for c in floor_conds], ls=ls, marker=ORACLE_MRK,
                ms=3.6, lw=1.4, color=ORACLE_COL, zorder=5)


LEGEND_HANDLES = (
    [Line2D([0], [0], color=COL[e], marker=MRK[e], ms=EMS[e], lw=ELW[e], label=LAB[e])
     for e in ESTIMATORS]
    + [Line2D([0], [0], color=ORACLE_COL, marker=ORACLE_MRK, ms=3.6, lw=1.4,
              label=ORACLE_LAB)]
    + [Line2D([0], [0], color='0.45', ls='--', lw=1.4, label=f'N = {n_small:,}'),
       Line2D([0], [0], color='0.45', ls='-', lw=1.4, label=f'N = {n_large:,}')])

fig, axes = plt.subplots(3, 2, figsize=(9.4, 10.6))

# A. Estimation accuracy. The quantity the estimators are judged on directly.
ax = axes[0, 0]
by_floor(ax, '{e}_relfrob_mean')
ax.set_ylabel('Relative covariance error')
ax.set_title('A.  Covariance estimation error', loc='left')

# B. What that costs the scores. The precision matrix, not the covariance, is
# what the distance calculation uses, so this panel is not a rescaling of A.
ax = axes[0, 1]
by_floor(ax, '{e}_d2_relerr_med_median')
ax.set_ylabel('Median relative error in $D^2$')
ax.set_title('B.  Deviation-score error', loc='left')

# C and D. Calibration of the reference sample's own distances. Read against the
# gaussian condition at the same N rather than against the nominal values, since
# the chi-square reference is asymptotic and the covariance here is estimated.
ax = axes[1, 0]
by_floor(ax, '{e}_ref_med_ratio_median')
oracle_curve(ax, 'oracle_ref_med_ratio_median')
ax.axhline(1.0, color='0.75', lw=0.8, ls=':', zorder=0)
ax.set_ylabel('Median $D^2$ ÷ $\\chi^2_{11}$ median')
ax.set_title('C.  Calibration against the benchmark, central tendency', loc='left')

ax = axes[1, 1]
by_floor(ax, '{e}_ref_pct_above_q95_mean', logy=False)
oracle_curve(ax, 'oracle_ref_pct_above_q95_mean')
ax.axhline(5.0, color='0.75', lw=0.8, ls=':', zorder=0)
ax.set_ylim(-3, 103)
ax.set_ylabel('Reference group beyond the .95 cutoff (%)')
ax.set_title('D.  Calibration against the benchmark, tail', loc='left')
# Both reference lines are identified in the caption rather than on the axes.
# Every in-panel position for the label either sat on a curve or ran off the
# bottom of the axis, and the lines are not ambiguous once the caption names them.

# E. Rank deficiency. MCD only, because no fit from either alternative was
# singular under any condition examined. Distinct dash patterns as well as alpha:
# the curves coincide at low floor prevalence, and alpha alone left the smallest
# sample size almost invisible.
ax = axes[2, 0]
DASHES = [(1, 1), (4, 1.6), (6, 1.6, 1.4, 1.6), (None, None)]
for i, n in enumerate(N_VALUES):
    kw = {} if DASHES[i % 4][0] is None else {'dashes': DASHES[i % 4]}
    ax.plot(pis, [get(c, n, 'mcd_pct_singular') for c in floor_conds], marker='o', ms=4,
            color=COL['mcd'], lw=1.6,
            alpha=0.72 + 0.28 * i / max(len(N_VALUES) - 1, 1),
            label=f'N = {n:,}', **kw)
ax.set_xlabel('Reference observations at floor (%)')
ax.set_ylabel('MCD fits with singular covariance (%)')
ax.set_ylim(-3, 103)
ax.set_title('E.  Rank deficiency (MCD)', loc='left')
ax.legend(fontsize=7.5, loc='upper left')

# F. The conditions robust estimation is built for, where the ordering reverses.
ax = axes[2, 1]
bar_conds = ['gaussian', 'skew', 'floor50', 'gaussian_contam05', 'gaussian_contam10', 'floor50_contam05']
bar_labs = ['Gaussian', 'Skew\n(no ties)', 'Floor\n50%', 'Gauss.\n+5% cont.',
            'Gauss.\n+10% cont.', 'Floor 50%\n+5% cont.']
n_mid = 1000 if 1000 in N_VALUES else N_VALUES[-1]
w = 0.26
xs = np.arange(len(bar_conds))
for i, e in enumerate(ESTIMATORS):
    ax.bar(xs + (i - 1) * w, [get(c, n_mid, f'{e}_d2_relerr_med_median') for c in bar_conds],
           width=w, color=COL[e], label=LAB[e])
ax.set_yscale('log')
ax.set_xticks(xs); ax.set_xticklabels(bar_labs, fontsize=6.8)
ax.set_ylabel('Median relative error in $D^2$')
ax.set_title(f'F.  Asymmetry and contamination (N = {n_mid:,})', loc='left')

# One legend for the whole figure, so no panel has to give up plotting area to it.
fig.legend(handles=LEGEND_HANDLES, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=3, fontsize=8, columnspacing=1.6, handlelength=2.2)
fig.tight_layout(h_pad=2.4, w_pad=2.6, rect=(0, 0, 1, 0.955))
save(fig, "Fig_study1")

# Supplementary: consequences for ranking. Kept separate from the principal
# figure because the manuscript reports the accuracy and calibration results and
# refers to ranking agreement only in passing.
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
ax = axes[0]
by_floor(ax, '{e}_spearman_mean', logy=False)
ax.set_ylim(0, 1.02)
ax.set_ylabel('Rank agreement with benchmark (Spearman $\\rho$)')
ax.set_title('A.  Ordering of the deviant group', loc='left')
ax = axes[1]
by_floor(ax, '{e}_top_jaccard_mean', logy=False)
ax.set_ylim(0, 1.02)
ax.set_ylabel('Top-decile overlap with benchmark (Jaccard)')
ax.set_title('B.  Who is flagged as most deviant', loc='left')
fig.legend(handles=LEGEND_HANDLES, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=5, fontsize=8, columnspacing=1.6, handlelength=2.2)
fig.tight_layout(w_pad=2.6, rect=(0, 0, 1, 0.925))
save(fig, "FigS_study1_ranking")


# Supplementary: error by N, faceted by condition
# Grid sized from the condition count, so adding a condition cannot silently
# push one off the figure.
_ncol = 3
_nrow = int(np.ceil(len(CONDITIONS) / _ncol))
fig, axes = plt.subplots(_nrow, _ncol, figsize=(10, 2.8 * _nrow), sharex=True)
# Shared log axes otherwise pad out to N = 1, leaving most of every facet empty.
x_lo, x_hi = min(N_VALUES) / 1.6, max(N_VALUES) * 1.6
for ax in axes.ravel()[len(CONDITIONS):]:
    ax.set_visible(False)
for ax, c in zip(axes.ravel(), CONDITIONS):
    g = summary[summary.condition == c]
    for e in ESTIMATORS:
        ax.errorbar(g['n'], g[f'{e}_relfrob_mean'], yerr=g[f'{e}_relfrob_sd'], marker=MRK[e],
                    ms=3.5, lw=1.3, capsize=2, color=COL[e], label=LAB[e])
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(c, loc='left', fontsize=9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([f'{n:,}' for n in N_VALUES], fontsize=7)
    ax.minorticks_off()
for ax in axes.ravel()[max(0, len(CONDITIONS) - _ncol):len(CONDITIONS)]:
    ax.set_xlabel('Reference sample size, N')
for ax in axes[:, 0]:
    ax.set_ylabel('Relative covariance error')
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
save(fig, "FigS_study1_error_by_N")
print("Done.")
