# -*- coding: utf-8 -*-
"""Does MCD's error shrink with reference sample size in this distribution?

The sample covariance, Ledoit-Wolf shrinkage and MCD are fitted to synthetic
reference samples drawn from the real control data by resampling with Gaussian
jitter, at eight sizes from 62 to 10,000 over 30 seeds. Each seed carries its own
100,000-observation benchmark and its own deviant group of 263, held constant
across sizes, so any change in the scores comes from the reference-side estimate.

This answers the question for this reference distribution only. General claims
come from the controlled simulation in 05_study1_simulation.py.

Distances are taken about a common center, the reference-sample mean, with MCD's
own-center distances kept for the location-versus-scatter decomposition. A jitter
sensitivity check repeats the run at jitter in {0, .01, .05, .10}.

Reads   data/imputed_clinical_data.csv
Writes  Sim2_runs.csv, Table3_simulation_summary.csv, SimS_jitter_runs.csv,
        TableS_jitter_sensitivity.csv, figures, versions_simulation_v2.json

Usage   python code/06_bootstrap_sample_size.py [--base PATH] [--quick]
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
from sklearn.preprocessing import StandardScaler

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
ap.add_argument("--data", type=Path, default=None,
                help="input file; defaults to data/imputed_clinical_data.csv")
ap.add_argument("--quick", action="store_true", help="3 seeds, small grid (code check only)")
args = ap.parse_args()
OUT = Path(args.base) / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
IN_PATH = args.data or Path(args.base) / "data" / "imputed_clinical_data.csv"

N_VALUES = [62, 100, 200, 500, 1000, 2000, 5000, 10000]
N_SEEDS = 30
SEED_OFFSET = 42
JITTER = 0.05
BENCH_N = 100_000
DEV_N = 263
TOP_Q = 0.90
JITTER_GRID = [0.0, 0.01, 0.05, 0.10]
JITTER_N = [62, 1000, 10000]
if args.quick:
    N_VALUES, N_SEEDS, BENCH_N, JITTER_N = [62, 500, 2000], 3, 20_000, [62, 1000]

mcd_features = [
    'gbdi_total_score', 'shaps_total_score_snaith', 'psc_total_score',
    'docs_total', 'octcdq_incompleteness_score', 'octcdq_harm_score',
    'bfas_domain_neuroticism', 'bfas_domain_extraversion',
    'bfas_domain_conscientiousness', 'bfas_domain_agreeableness',
    'bfas_domain_openness_intellect'
]
SCALE_MIN = {'shaps_total_score_snaith': 0, 'docs_total': 0}
ESTIMATORS = ['sample', 'ledoit_wolf', 'mcd']

(OUT / "versions_simulation_v2.json").write_text(json.dumps({
    "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    "scikit-learn": sklearn.__version__, "seeds": [SEED_OFFSET, SEED_OFFSET + N_SEEDS - 1],
    "jitter": JITTER, "bench_n": BENCH_N}, indent=2))

df = pd.read_csv(IN_PATH)
hc_src = df.loc[df['group'] == 'HC', mcd_features].reset_index(drop=True)
ocd_src = df.loc[df['group'] == 'OCD', mcd_features].dropna().reset_index(drop=True)
scaler = StandardScaler().fit(hc_src)
P = len(mcd_features)


def rng_for(seed, role, n=0):
    """Independent random stream per (seed, role, n)."""
    return np.random.default_rng([seed, role, n])


def bootstrap_jitter(src, n, jitter, rng):
    idx = rng.integers(0, len(src), size=n)
    raw = src.to_numpy(float)[idx]
    noisy = raw + rng.normal(0, 1, raw.shape) * (jitter * src.std(axis=0, ddof=1).to_numpy())
    return noisy, raw


def pinv_rank(cov):
    Pm, rank = linalg.pinvh(cov, return_rank=True)
    return Pm, int(rank)


def d2(Z, center, Pm):
    D = Z - center
    return np.einsum('ij,jk,ik->i', D, Pm, D)


def cond(cov, rank):
    if rank < cov.shape[0]:
        return np.inf
    ev = np.linalg.eigvalsh(cov)
    return ev.max() / ev.min()


def fit_all(X, seed):
    out = {}
    out['sample'] = EmpiricalCovariance().fit(X)
    out['ledoit_wolf'] = LedoitWolf().fit(X)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out['mcd'] = MinCovDet(random_state=seed).fit(X)
    return out, len(w)


def one_setting(seed, n, jitter, bench_cov, bench_mean, dev, d2_bench, top_bench, cov_sums):
    rng = rng_for(seed, 3, n)
    noisy, raw = bootstrap_jitter(hc_src, n, jitter, rng)
    X = scaler.transform(pd.DataFrame(noisy, columns=mcd_features))
    center = X.mean(axis=0)
    fits, n_warn = fit_all(X, seed)
    row = dict(seed=seed, n=n, jitter=jitter, mcd_warnings=n_warn)
    bnorm = np.linalg.norm(bench_cov, 'fro')
    Pbench = linalg.pinvh(bench_cov)
    pnorm = np.linalg.norm(Pbench, 'fro')
    for name, est in fits.items():
        C = est.covariance_
        Pm, rank = pinv_rank(C)
        dd = d2(dev, center, Pm)
        top = dd >= np.quantile(dd, TOP_Q)
        err = np.linalg.norm(C - bench_cov, 'fro')
        row.update({
            f'{name}_frob': err, f'{name}_relfrob': err / bnorm,
            f'{name}_prec_relfrob': np.linalg.norm(Pm - Pbench, 'fro') / pnorm,
            f'{name}_rank': rank, f'{name}_singular': rank < P,
            f'{name}_cond': cond(C, rank),
            f'{name}_dev_med': np.median(dd),
            f'{name}_d2_relerr_med': np.median(np.abs(dd - d2_bench) / d2_bench),
            f'{name}_spearman': stats.spearmanr(dd, d2_bench).statistic,
            f'{name}_top_jaccard': (top & top_bench).sum() / (top | top_bench).sum(),
        })
        if cov_sums is not None:
            cov_sums[(n, name)] = cov_sums.get((n, name), 0) + C
    mcd = fits['mcd']
    dd_own = mcd.mahalanobis(dev)          # robust location + MCD scatter
    row['mcd_own_center_dev_med'] = np.median(dd_own)
    row['mcd_support'] = mcd.support_.mean()
    sup = mcd.support_
    docs = raw[:, mcd_features.index('docs_total')]
    shaps = raw[:, mcd_features.index('shaps_total_score_snaith')]
    incl = docs[sup].mean()
    excl = docs[~sup].mean() if (~sup).any() else np.nan
    row['docs_mean_retained'], row['docs_mean_excluded'] = incl, excl
    row['restrict_ratio'] = excl / incl if incl > 0 else np.nan
    for lab, v in (('shaps', shaps), ('docs', docs)):
        row[f'pct_floor_{lab}_retained'] = 100 * np.mean(v[sup] == 0)
        row[f'pct_floor_{lab}_excluded'] = 100 * np.mean(v[~sup] == 0) if (~sup).any() else np.nan
    return row


def run_seed(seed, n_values, jitter, cov_sums=None):
    b_noisy, _ = bootstrap_jitter(hc_src, BENCH_N, jitter, rng_for(seed, 1))
    B = scaler.transform(pd.DataFrame(b_noisy, columns=mcd_features))
    bench_mean, bench_cov = B.mean(axis=0), np.cov(B, rowvar=False, bias=True)
    Pb, rb = pinv_rank(bench_cov)
    d_noisy, _ = bootstrap_jitter(ocd_src, DEV_N, jitter, rng_for(seed, 2))
    dev = scaler.transform(pd.DataFrame(d_noisy, columns=mcd_features))
    d2_bench = d2(dev, bench_mean, Pb)
    top_bench = d2_bench >= np.quantile(d2_bench, TOP_Q)
    rows = [one_setting(seed, n, jitter, bench_cov, bench_mean, dev, d2_bench, top_bench, cov_sums)
            for n in n_values]
    for r in rows:
        r['bench_cond'] = cond(bench_cov, rb)
        r['bench_rank'] = rb
        r['bench_dev_med'] = np.median(d2_bench)
    return rows, bench_cov


# ------------------------------------------------------------------
# Main sweep
# ------------------------------------------------------------------
seeds = list(range(SEED_OFFSET, SEED_OFFSET + N_SEEDS))
print(f"--- Main sweep: {N_SEEDS} seeds x {len(N_VALUES)} N (jitter = {JITTER}) ---")
t0 = time.time()
rows, cov_sums, bench_sum = [], {}, 0
for s in seeds:
    r, bc = run_seed(s, N_VALUES, JITTER, cov_sums)
    rows += r
    bench_sum = bench_sum + bc
runs = pd.DataFrame(rows)
print(f"done in {time.time() - t0:.0f}s")
runs.to_csv(OUT / "Sim2_runs.csv", index=False)

metrics = ['frob', 'relfrob', 'prec_relfrob', 'cond', 'dev_med', 'd2_relerr_med', 'spearman', 'top_jaccard']
agg = {}
for e in ESTIMATORS:
    for m in metrics:
        col = f'{e}_{m}'
        agg[f'{col}_mean'] = (col, lambda x: x.replace(np.inf, np.nan).mean())
        agg[f'{col}_sd'] = (col, lambda x: x.replace(np.inf, np.nan).std())
        # Quantiles for the log-axis panels; means and SDs remain the table values.
        agg[f'{col}_q25'] = (col, lambda x: x.replace(np.inf, np.nan).quantile(.25))
        agg[f'{col}_med'] = (col, lambda x: x.replace(np.inf, np.nan).median())
        agg[f'{col}_q75'] = (col, lambda x: x.replace(np.inf, np.nan).quantile(.75))
    agg[f'{e}_pct_singular'] = (f'{e}_singular', lambda x: 100 * x.mean())
for c in ['mcd_own_center_dev_med', 'mcd_support', 'restrict_ratio',
          'pct_floor_shaps_retained', 'pct_floor_shaps_excluded',
          'pct_floor_docs_retained', 'pct_floor_docs_excluded', 'bench_dev_med', 'bench_cond']:
    agg[f'{c}_mean'] = (c, 'mean')
    agg[f'{c}_sd'] = (c, 'std')
agg['mcd_warning_fits_pct'] = ('mcd_warnings', lambda x: 100 * (x > 0).mean())
summary = runs.groupby('n').agg(**agg).reset_index()

# Mean-matrix bias: || mean over seeds of S_hat - mean benchmark ||
bench_mean_cov = bench_sum / N_SEEDS
for e in ESTIMATORS:
    summary[f'{e}_meanmatrix_bias'] = [np.linalg.norm(cov_sums[(n, e)] / N_SEEDS - bench_mean_cov, 'fro')
                                       for n in summary['n']]
summary.round(4).to_csv(OUT / "Table3_simulation_summary.csv", index=False)
pd.set_option('display.width', 250)
show = ['n'] + [f'{e}_frob_mean' for e in ESTIMATORS] \
       + [f'{e}_prec_relfrob_mean' for e in ESTIMATORS] \
       + [f'{e}_pct_singular' for e in ESTIMATORS] \
       + [f'{e}_dev_med_mean' for e in ESTIMATORS] \
       + ['mcd_own_center_dev_med_mean', 'bench_dev_med_mean'] \
       + [f'{e}_top_jaccard_mean' for e in ESTIMATORS] \
       + ['restrict_ratio_mean']
print(summary[show].round(3).to_string(index=False))

# ------------------------------------------------------------------
# Jitter sensitivity
# ------------------------------------------------------------------
print(f"\n--- Jitter sensitivity: {JITTER_GRID} x N {JITTER_N} ---")
jrows = []
for j in JITTER_GRID:
    for s in seeds:
        r, _ = run_seed(s, JITTER_N, j)
        jrows += r
jruns = pd.DataFrame(jrows)
jruns.to_csv(OUT / "SimS_jitter_runs.csv", index=False)
jsum = jruns.groupby(['jitter', 'n']).agg(
    **{f'{e}_frob_mean': (f'{e}_frob', 'mean') for e in ESTIMATORS},
    **{f'{e}_pct_singular': (f'{e}_singular', lambda x: 100 * x.mean()) for e in ESTIMATORS},
    mcd_top_jaccard_mean=('mcd_top_jaccard', 'mean'),
    restrict_ratio_mean=('restrict_ratio', 'mean'),
).reset_index()
jsum.round(4).to_csv(OUT / "TableS_jitter_sensitivity.csv", index=False)
print(jsum.round(3).to_string(index=False))

# ==================================================================
# FIGURE
# ==================================================================
COL = {'mcd': "#D55E00", 'ledoit_wolf': "#0072B2", 'sample': "#009E73"}
LAB = {'mcd': 'MCD', 'ledoit_wolf': 'Ledoit-Wolf', 'sample': 'Sample covariance'}
MRK = {'mcd': 'o', 'ledoit_wolf': 's', 'sample': '^'}
C_RATIO = "#7A1FA2"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight"})


def save(fig, stem):
    """Write PDF and 600-dpi PNG, reporting any fallback loudly.

    A silent fallback is dangerous here. If the target is locked by a viewer or a
    sync client, the new figure lands under a timestamped name while a stale file
    keeps the expected name, and the stale one is what gets read later.
    """
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        target = OUT / f"{stem}{ext}"
        try:
            fig.savefig(target, **kw)
        except OSError as exc:
            alt = OUT / f"{stem}_{pd.Timestamp.now():%Y%m%d_%H%M%S}{ext}"
            fig.savefig(alt, **kw)
            print(f"  ! could not write {target.name} ({exc.__class__.__name__}: {exc}).")
            print(f"    Saved as {alt.name}. Any existing {target.name} is STALE.")
            print("    Close programs holding the file, or pass --base outside OneDrive/Dropbox.")
    plt.close(fig)


def band(ax, x, mid, lo, hi, color, label=None, marker="o", ls="-"):
    """Central line with a shaded interval given as explicit lower/upper edges."""
    mid = np.asarray(mid, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    ax.plot(x, mid, marker=marker, ls=ls, ms=4, lw=1.6, color=color, label=label, zorder=3)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)


def band_sd(ax, x, mean, sd, color, label=None, marker="o", ls="-"):
    """Mean +/- 1 SD. For linear axes only; on a log axis use band_iqr."""
    mean = np.asarray(mean, float); sd = np.nan_to_num(np.asarray(sd, float))
    band(ax, x, mean, mean - sd, mean + sd, color, label, marker, ls)


def band_iqr(ax, x, col_stem, frame, color, label=None, marker="o", ls="-"):
    """Median with an interquartile band.

    Used for the log-scaled panels. Condition numbers and median D^2 are strongly
    right-skewed across seeds, so mean - SD can reach zero or below, which a log
    axis cannot render and which draws a misleading clipped wedge.
    """
    band(ax, x, frame[f'{col_stem}_med'], frame[f'{col_stem}_q25'], frame[f'{col_stem}_q75'],
         color, label, marker, ls)


x = summary['n'].to_numpy(float)
fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4))
BENCH_COND = float(summary['bench_cond_mean'].mean())

any_singular = float(summary[[f'{e}_pct_singular' for e in ESTIMATORS]].to_numpy().max()) > 0

ax = axes[0, 0]
for e in ESTIMATORS:
    band_iqr(ax, x, f'{e}_cond', summary, COL[e], LAB[e], MRK[e])
ax.axhline(BENCH_COND, color='k', ls=':', lw=1.3, label=f'benchmark ({BENCH_COND:.1f})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Reference sample size, N')
ax.set_ylabel('Condition number' + (' (invertible fits)' if any_singular else ''))
ax.set_title('A.  Covariance conditioning', loc='left'); ax.legend(fontsize=7.5)

ax = axes[0, 1]
for e in ESTIMATORS:
    band_sd(ax, x, summary[f'{e}_frob_mean'], summary[f'{e}_frob_sd'], COL[e], LAB[e], MRK[e])
ax.axhline(0, color='k', ls=':', lw=1.2)
ax.set_xscale('log')
ax.set_xlabel('Reference sample size, N')
ax.set_ylabel('Frobenius estimation error')
ax.set_title('B.  Estimation error relative to the benchmark', loc='left'); ax.legend(fontsize=7.5)

ax = axes[1, 0]
ax.plot(x, summary['mcd_support_mean'], marker='o', ms=4, lw=1.6, color=COL['mcd'])
ax.set_xscale('log'); ax.set_xlabel('Reference sample size, N')
ax.set_ylabel('MCD realized support fraction')
ax2 = ax.twinx(); ax2.spines['right'].set_visible(True)
band_sd(ax2, x, summary['restrict_ratio_mean'], summary['restrict_ratio_sd'], C_RATIO,
        marker='s', ls='--')
ax2.set_ylabel('Restriction ratio (excluded / retained mean DOCS)')
ax.set_title('C.  MCD support selection', loc='left')
ax.legend(handles=[Line2D([0], [0], color=COL['mcd'], marker='o', ms=4, lw=1.6,
                          label='realized support fraction (left axis)'),
                   Line2D([0], [0], color=C_RATIO, marker='s', ms=4, lw=1.6, ls='--',
                          label='restriction ratio (right axis)')],
          fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.20))

ax = axes[1, 1]
for e in ESTIMATORS:
    band_iqr(ax, x, f'{e}_dev_med', summary, COL[e], LAB[e], MRK[e])
bd = summary['bench_dev_med_mean'].mean()
ax.axhline(bd, color='k', ls=':', lw=1.3, label=f'benchmark distances ({bd:.0f})')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Reference sample size, N')
ax.set_ylabel('Median $D^2$, synthetic deviant group')
ax.set_title('D.  Deviation scores (common center)', loc='left'); ax.legend(fontsize=7.5)
fig.text(0.005, 0.005,
         'Panels A and D: across-seed median with an interquartile band (log axes). '
         'Panels B and C: mean ± 1 SD.',
         fontsize=7, color='0.35', ha='left', va='bottom')
fig.tight_layout(h_pad=2.2, w_pad=2.6, rect=(0, 0.025, 1, 1))
save(fig, "Fig_sim2_estimator_behavior")

sub = runs[runs['n'] == N_VALUES[len(N_VALUES) // 2]].reset_index(drop=True)
k = np.arange(1, len(sub) + 1)
rm = sub['mcd_frob'].expanding().mean()
rse = sub['mcd_frob'].expanding().std() / np.sqrt(k)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
band_sd(ax, k, rm, rse, COL['mcd'])
ax.set_xlabel('Seeds accumulated')
ax.set_ylabel(f'Running mean Frobenius error\n(MCD, N = {int(sub.n.iloc[0]):,})')
ax.set_title('Seed-count stability check', loc='left')
ax.annotate('band is ±1 SE of the running mean', xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=7.5, color='gray')
save(fig, "FigS_seed_convergence")
print("\nDone.")
