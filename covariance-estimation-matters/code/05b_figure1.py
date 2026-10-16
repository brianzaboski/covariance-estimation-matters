"""Figure 1, redrawn from the archived Study 1 runs with across-seed variability.

Reads outputs/Study1_runs.csv (one row per condition x reference size x seed, as
written by 05_study1_simulation.py) and outputs/Study1_summary.csv, and draws
Fig_study1 without re-running the simulation. Panels A-D show the across-seed
median as the line and the interquartile range (25th-75th percentile across the
30 seeds) as a shaded band; Panel F shows medians with interquartile whiskers;
Panel E (percentage of singular MCD fits) is unchanged.

Re-running the simulation is deliberately avoided: rank-deficient MCD fits at the
.75 floor depend on floating-point details of the machine (see README). The script
draws whatever outputs/Study1_runs.csv holds; in the archive, that file is the run
reported in the paper.

Panel C's axis is capped at C_TOP so the curves the text discusses stay legible;
any band that extends past the cap is marked with an arrow giving its upper quartile.

Writes  Fig_study1.pdf, Fig_study1.png (600 dpi), Fig_study1_data.csv (every
        plotted value: statistic 'median' with q25/q75 for Panels A-D and F,
        'percent_singular' for Panel E)

Usage   python code/05b_figure1.py [--base PATH] [--out PATH]
        Run after 05_study1_simulation.py; it overwrites 05's version of Fig_study1.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DEFAULT_BASE = (Path(__file__).resolve().parents[1]
                if "__file__" in globals() else Path.cwd())
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=DEFAULT_BASE)
ap.add_argument("--out", default=None, help="output folder (default: BASE/outputs)")
args = ap.parse_args()
SRC = Path(args.base) / "outputs"
OUT = Path(args.out) if args.out else SRC
OUT.mkdir(parents=True, exist_ok=True)

runs = pd.read_csv(SRC / "Study1_runs.csv")
summary = pd.read_csv(SRC / "Study1_summary.csv")

ESTIMATORS = ['sample', 'ledoit_wolf', 'mcd']
N_VALUES = sorted(runs['n'].unique())
COL = {'mcd': "#D55E00", 'ledoit_wolf': "#0072B2", 'sample': "#009E73"}
LAB = {'mcd': 'MCD', 'ledoit_wolf': 'Ledoit-Wolf', 'sample': 'Sample covariance'}
MRK = {'mcd': 'o', 'ledoit_wolf': 's', 'sample': '^'}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight"})

floor_conds = ['gaussian', 'floor10', 'floor25', 'floor50', 'floor75']
pis = [0, 10, 25, 50, 75]
n_small, n_large = N_VALUES[0], N_VALUES[-1]
NSTY = ((n_small, '--'), (n_large, '-'))
ELW = {'sample': 3.0, 'ledoit_wolf': 1.6, 'mcd': 1.6}
EMS = {'sample': 5.5, 'ledoit_wolf': 4.0, 'mcd': 4.0}
BAND_ALPHA = {n_small: 0.13, n_large: 0.22}
N_SEEDS = int(runs.groupby(['condition', 'n']).size().max())
C_TOP = 3e3
ORACLE_COL, ORACLE_MRK, ORACLE_LAB = "#000000", 'D', 'Benchmark covariance'

records = []


def quartiles(cond, n, col):
    x = pd.to_numeric(runs.loc[(runs.condition == cond) & (runs.n == n), col], errors="coerce")
    bad = int(x.isna().sum() + np.isinf(x).sum())
    if bad:
        print(f"  note: {col} {cond} n={n}: {bad} missing/infinite of {len(x)} excluded")
    x = x[np.isfinite(x)]
    return float(np.median(x)), float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))


def by_floor(ax, panel, col, logy=True, oracle_col=None):
    """Median line and IQR band per estimator and reference size, across floor prevalence."""
    series = [(e, col.format(e=e), COL[e], MRK[e], EMS[e], ELW[e], 2) for e in ESTIMATORS]
    if oracle_col:
        series.append(('benchmark', oracle_col, ORACLE_COL, ORACLE_MRK, 3.6, 1.4, 5))
    for name, c, color, mk, ms, lw, z in series:
        for n, ls in NSTY:
            q = [quartiles(cond, n, c) for cond in floor_conds]
            med, lo, hi = (np.array(v) for v in zip(*q))
            ax.fill_between(pis, lo, hi, color=color, alpha=BAND_ALPHA[n] * (0.8 if name == 'benchmark' else 1),
                            lw=0, zorder=z - 1)
            ax.plot(pis, med, ls=ls, marker=mk, ms=ms, lw=lw, color=color, zorder=z)
            for cond, m, a, b in zip(floor_conds, med, lo, hi):
                records.append(dict(panel=panel, condition=cond, n=n, series=name, statistic='median',
                                    value=m, q25=a, q75=b))
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel('Reference observations at floor (%)')


LEGEND_HANDLES = (
    [Line2D([0], [0], color=COL[e], marker=MRK[e], ms=EMS[e], lw=ELW[e], label=LAB[e]) for e in ESTIMATORS]
    + [Line2D([0], [0], color=ORACLE_COL, marker=ORACLE_MRK, ms=3.6, lw=1.4, label=ORACLE_LAB)]
    + [Line2D([0], [0], color='0.45', ls='--', lw=1.4, label=f'N = {n_small:,}'),
       Line2D([0], [0], color='0.45', ls='-', lw=1.4, label=f'N = {n_large:,}')]
    + [Patch(facecolor='0.55', alpha=0.3, lw=0, label=f'Interquartile range ({N_SEEDS} seeds)')])

fig, axes = plt.subplots(3, 2, figsize=(9.4, 10.6))

ax = axes[0, 0]
by_floor(ax, 'A', '{e}_relfrob')
# The log axis spans less than two decades, so label it explicitly; otherwise only
# one tick (0.1) carries a label and values cannot be read off the panel.
ax.set_ylim(top=1.05)
ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.03, 0.1, 0.3, 1.0]))
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylabel('Relative covariance error')
ax.set_title('A.  Covariance estimation error', loc='left')

ax = axes[0, 1]
by_floor(ax, 'B', '{e}_d2_relerr_med')
ax.set_ylabel('Median relative error in $D^2$')
ax.set_title('B.  Deviation-score error', loc='left')

ax = axes[1, 0]
by_floor(ax, 'C', '{e}_ref_med_ratio', oracle_col='oracle_ref_med_ratio')
ax.axhline(1.0, color='0.75', lw=0.8, ls=':', zorder=0)
# Cap the axis: one MCD band (.75 floor) runs to ~10^6 and would otherwise flatten the
# full-sample and benchmark curves into one line. Mark anything clipped.
ax.set_ylim(top=C_TOP)
for rec in [r for r in records if r['panel'] == 'C' and r['q75'] > C_TOP]:
    x = pis[floor_conds.index(rec['condition'])]
    mant, ex = f"{rec['q75']:.1e}".split('e')
    ax.annotate(f"IQR to {mant} × 10$^{{{int(ex)}}}$\n(N = {rec['n']:,})", xy=(x, C_TOP * 0.92),
                xytext=(x - 17, C_TOP * 0.08), fontsize=7, ha='center', color=COL['mcd'],
                arrowprops=dict(arrowstyle='->', color=COL['mcd'], lw=0.9))
ax.set_ylabel('Median $D^2$ ÷ $\\chi^2_{11}$ median')
ax.set_title('C.  Calibration against the benchmark, central tendency', loc='left')

ax = axes[1, 1]
by_floor(ax, 'D', '{e}_ref_pct_above_q95', logy=False, oracle_col='oracle_ref_pct_above_q95')
ax.axhline(5.0, color='0.75', lw=0.8, ls=':', zorder=0)
ax.set_ylim(-3, 103)
ax.set_ylabel('Reference group beyond the .95 cutoff (%)')
ax.set_title('D.  Calibration against the benchmark, tail', loc='left')

# E. Unchanged from 05: percentage of the 30 MCD fits that were singular.
ax = axes[2, 0]
get = lambda c, n, col: summary.loc[(summary.condition == c) & (summary.n == n), col].iloc[0]
DASHES = [(1, 1), (4, 1.6), (6, 1.6, 1.4, 1.6), (None, None)]
for i, n in enumerate(N_VALUES):
    kw = {} if DASHES[i % 4][0] is None else {'dashes': DASHES[i % 4]}
    vals = [get(c, n, 'mcd_pct_singular') for c in floor_conds]
    ax.plot(pis, vals, marker='o', ms=4, color=COL['mcd'], lw=1.6,
            alpha=0.72 + 0.28 * i / max(len(N_VALUES) - 1, 1), label=f'N = {n:,}', **kw)
    for cond, v in zip(floor_conds, vals):
        records.append(dict(panel='E', condition=cond, n=n, series='mcd', statistic='percent_singular',
                            value=v, q25=np.nan, q75=np.nan))
ax.set_xlabel('Reference observations at floor (%)')
ax.set_ylabel('MCD fits with singular covariance (%)')
ax.set_ylim(-3, 103)
ax.set_title('E.  Rank deficiency (MCD)', loc='left')
ax.legend(fontsize=7.5, loc='upper left')

# F. Medians with interquartile whiskers at N = 1,000.
ax = axes[2, 1]
bar_conds = ['gaussian', 'skew', 'floor50', 'gaussian_contam05', 'gaussian_contam10', 'floor50_contam05']
bar_labs = ['Gaussian', 'Skew\n(no ties)', 'Floor\n50%', 'Gauss.\n+5% cont.', 'Gauss.\n+10% cont.', 'Floor 50%\n+5% cont.']
n_mid = 1000 if 1000 in N_VALUES else N_VALUES[-1]
w = 0.26
xs = np.arange(len(bar_conds))
for i, e in enumerate(ESTIMATORS):
    q = [quartiles(c, n_mid, f'{e}_d2_relerr_med') for c in bar_conds]
    med, lo, hi = (np.array(v) for v in zip(*q))
    ax.bar(xs + (i - 1) * w, med, width=w, color=COL[e], label=LAB[e])
    ax.errorbar(xs + (i - 1) * w, med, yerr=[med - lo, hi - med], fmt='none', ecolor='0.15', elinewidth=0.8,
                capsize=1.8, capthick=0.8, zorder=4)
    for c, m, a, b in zip(bar_conds, med, lo, hi):
        records.append(dict(panel='F', condition=c, n=n_mid, series=e, statistic='median', value=m, q25=a, q75=b))
ax.set_yscale('log')
ax.set_xticks(xs); ax.set_xticklabels(bar_labs, fontsize=6.8)
ax.set_ylabel('Median relative error in $D^2$')
ax.set_title(f'F.  Asymmetry and contamination (N = {n_mid:,})', loc='left')

fig.legend(handles=LEGEND_HANDLES, loc='upper center', bbox_to_anchor=(0.5, 1.0),
           ncol=4, fontsize=8, columnspacing=1.4, handlelength=2.2)
fig.tight_layout(h_pad=2.4, w_pad=2.6, rect=(0, 0, 1, 0.955))
for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
    fig.savefig(OUT / f"Fig_study1{ext}", **kw)
plt.close(fig)
pd.DataFrame(records).to_csv(OUT / "Fig_study1_data.csv", index=False)
print("Saved Fig_study1.pdf, Fig_study1.png, Fig_study1_data.csv to", OUT)
