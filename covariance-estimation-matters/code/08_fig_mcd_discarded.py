# -*- coding: utf-8 -*-
"""Figure 2: what MCD discarded.

Two panels sharing a feature axis, one row per model feature. Panel A gives each
group's mean in standard-deviation units of the full control sample, as a
dumbbell, which is the metric the directional profiling uses and lets eleven
measures on eight scales be read against one another. Panel B gives the
percentage of each group at the scale minimum.

The pairing carries the argument. The excluded controls are more symptomatic and
also differ on personality, where panel B shows there is no floor at all, so the
selection is not an artifact of the floors.

Colours are the Okabe-Ito blue and vermillion used in the other figures, and the
legend names both groups, so identity is never carried by hue alone.

Caption: the six features above the divider have a measurement floor; the five
BFAS domains below it do not, and no participant in any support set scored at
their minimum.

Reads   outputs/TableS_mcd_support_composition.csv (written by 03)
Writes  Fig_mcd_discarded.pdf / .png

Usage   python code/08_fig_mcd_discarded.py [--base PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
ap.add_argument("--stem", default="Fig_mcd_discarded")
args = ap.parse_args()
OUT = Path(args.base) / "outputs"

C_RET, C_EXC = "#0072B2", "#D55E00"      # Okabe-Ito blue, vermillion
C_LINE, C_INK, C_MUTE = "#BBBBBB", "#222222", "#666666"

LABEL = {
    'gbdi_total_score': 'gBDI (depression)',
    'shaps_total_score_snaith': 'SHAPS (anhedonia)',
    'psc_total_score': 'PSC (physical symptoms)',
    'docs_total': 'DOCS (OC severity)',
    'octcdq_incompleteness_score': 'OCTCDQ incompleteness',
    'octcdq_harm_score': 'OCTCDQ harm avoidance',
    'bfas_domain_neuroticism': 'BFAS neuroticism',
    'bfas_domain_extraversion': 'BFAS extraversion',
    'bfas_domain_conscientiousness': 'BFAS conscientiousness',
    'bfas_domain_agreeableness': 'BFAS agreeableness',
    'bfas_domain_openness_intellect': 'BFAS openness/intellect',
}
# Symptom block first, then personality; within each, largest separation on top.
ORDER = ['gbdi_total_score', 'docs_total', 'octcdq_harm_score',
         'octcdq_incompleteness_score', 'psc_total_score',
         'shaps_total_score_snaith',
         'bfas_domain_neuroticism', 'bfas_domain_openness_intellect',
         'bfas_domain_conscientiousness', 'bfas_domain_agreeableness',
         'bfas_domain_extraversion']

comp = pd.read_csv(OUT / "TableS_mcd_support_composition.csv").set_index('set')
need = {'reweighted_support', 'excluded'}
if not need.issubset(comp.index):
    raise SystemExit(f"support composition file is missing {need - set(comp.index)}")
if f'z_{ORDER[0]}' not in comp.columns:
    raise SystemExit("no z_* columns; rerun 03_phase2_analyses.py")

n_ret = int(comp.loc['reweighted_support', 'n'])
n_exc = int(comp.loc['excluded', 'n'])


rows = []
for f in ORDER:
    rows.append(dict(
        feature=f,
        z_ret=comp.loc['reweighted_support', f'z_{f}'],
        z_exc=comp.loc['excluded', f'z_{f}'],
        floor_ret=comp.loc['reweighted_support', f'pct_min_{f}'],
        floor_exc=comp.loc['excluded', f'pct_min_{f}'],
    ))
d = pd.DataFrame(rows)
y = np.arange(len(d))[::-1]        # first feature at the top

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(9.6, 5.0), sharey=True,
    gridspec_kw=dict(width_ratios=[1.25, 1.0], wspace=0.08))


def dumbbell(ax, lo_col, hi_col, xlo, xhi, zero_line=False):
    if zero_line:
        ax.axvline(0, color='#AAAAAA', lw=1, zorder=0)
    for yi, (_, r) in zip(y, d.iterrows()):
        a, b = r[lo_col], r[hi_col]
        ax.plot([a, b], [yi, yi], color=C_LINE, lw=2, zorder=1,
                solid_capstyle='round')
        ax.plot(a, yi, 'o', ms=8, color=C_RET, mec='white', mew=1.6, zorder=3)
        ax.plot(b, yi, 'o', ms=8, color=C_EXC, mec='white', mew=1.6, zorder=3)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.grid(axis='x', color='#E8E8E8', lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(length=0, colors=C_MUTE, labelsize=8)


zs = pd.concat([d.z_ret, d.z_exc])
pad = 0.12 * (zs.max() - zs.min())
dumbbell(axA, 'z_ret', 'z_exc', zs.min() - pad, zs.max() + pad, zero_line=True)
dumbbell(axB, 'floor_ret', 'floor_exc', -4, 104)

axA.set_yticks(y)
axA.set_yticklabels([LABEL[f] for f in d.feature], fontsize=9, color=C_INK)
axA.set_xlabel('Group mean (SD units of the full control sample)', fontsize=9,
               color=C_INK)
axB.set_xlabel('At the scale minimum (%)', fontsize=9, color=C_INK)
axA.set_title('A. Where each group sits, relative to all controls', fontsize=10,
              color=C_INK, loc='left', pad=8)
axB.set_title('B. How much of each group is on the floor', fontsize=10,
              color=C_INK, loc='left', pad=8)

# Divider between the symptom block and the personality block.
split = len(y) - 6 - 0.5
for ax in (axA, axB):
    ax.axhline(split, color='#DDDDDD', lw=1, zorder=0)
# Which side is which is said in the caption; in-plot labels collided with data.

fig.legend(handles=[Line2D([], [], marker='o', ls='', ms=8, color=C_RET,
                           label=f'Retained by MCD (n = {n_ret})'),
                    Line2D([], [], marker='o', ls='', ms=8, color=C_EXC,
                           label=f'Excluded by MCD (n = {n_exc})')],
           loc='lower center', ncol=2, frameon=False, fontsize=9,
           bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.24, right=0.985, top=0.90, bottom=0.17)

for ext in ('pdf', 'png'):
    path = OUT / f"{args.stem}.{ext}"
    try:
        fig.savefig(path, dpi=300)
        print(f"  saved {path.name}")
    except OSError as e:
        alt = OUT / f"{args.stem}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.{ext}"
        fig.savefig(alt, dpi=300)
        print(f"  WARNING: could not write {path.name} ({e}); saved {alt.name}")
plt.close(fig)

print("\nValues plotted:")
print(d.round(1).to_string(index=False))
