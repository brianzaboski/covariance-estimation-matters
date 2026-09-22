# -*- coding: utf-8 -*-
"""Internal consistency of the eleven model features.

Table 1 covers four of the eleven. Four more are scored from item subsets it
does not break out and five are BFAS domains it does not cover, so seven have no
in-sample reliability estimate. This supplies them, with Feldt intervals and the
mean inter-item correlation alongside alpha, since alpha rises with item count
and a 20-item domain is not comparable to a 10-item subscale on alpha alone.

Scoring follows 01_clean_and_score.py so that reliability is estimated on the
variables the model uses: SHAPS dichotomized at the same cut, OCTCDQ on its raw
1-5 metric split odd and even, BFAS reverse-keyed and averaged the same way. The
reverse key and aspect map are checked against the data by
02_descriptives_octcdq_bfas.py. The four features Table 1 already reports are
recomputed here as a check that this script and the descriptives script have not
drifted apart; they should return .93, .79, .88 and .94.

Reads   data/final_df.csv (item level, pre-imputation)
Writes  TableS_feature_reliability.csv
        TableS_feature_reliability_bfas_aspects.csv

Usage   python code/02b_feature_reliability.py [--source final_df.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# parents[1] when run as a file; the working directory when pasted into an
# interactive session, where __file__ does not exist
DEFAULT_BASE = (Path(__file__).resolve().parents[1]
                if "__file__" in globals() else Path.cwd())
COMPLETION_THRESHOLD = 0.80          # same proration rule as the descriptives script

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=DEFAULT_BASE)
ap.add_argument("--source", default="final_df.csv")
args = ap.parse_args()
BASE = Path(args.base)
DATA = BASE / "data"
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / args.source)
N = len(df)
print(f"{args.source}: {N} rows, {df.shape[1]} columns")


# ---------------------------------------------------------------------------
# Helpers.  `numeric`, `prorated_total` and `cronbach_alpha` are the same as in
# Descriptive_and_psychometrics_script.py; do not let the two drift apart.
# ---------------------------------------------------------------------------
def numeric(frame, cols):
    cols = [c for c in cols if c in frame.columns]
    return frame[cols].apply(pd.to_numeric, errors='coerce'), cols


def prorated_total(frame, cols, threshold=COMPLETION_THRESHOLD):
    sub, cols = numeric(frame, cols)
    if not cols:
        return pd.Series(np.nan, index=frame.index)
    n_ans = sub.notna().sum(axis=1)
    total = sub.mean(axis=1, skipna=True) * len(cols)
    total[n_ans < np.ceil(threshold * len(cols))] = np.nan
    return total


def cronbach_alpha(item_frame):
    X = item_frame.dropna()
    k = X.shape[1]
    if k < 2 or len(X) < 3:
        return np.nan
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_var.sum() / total_var)


def alpha_ci(alpha, n, k, conf=0.95):
    """Feldt interval for coefficient alpha.

    1 - alpha is distributed as an F ratio with n-1 and (n-1)(k-1) degrees of
    freedom, so the interval follows from the F quantiles. Returns (nan, nan)
    when alpha is undefined or the sample is too small.
    """
    if not np.isfinite(alpha) or n < 3 or k < 2:
        return np.nan, np.nan
    df1, df2 = n - 1, (n - 1) * (k - 1)
    lo_f = stats.f.ppf(1 - (1 - conf) / 2, df1, df2)
    hi_f = stats.f.ppf((1 - conf) / 2, df1, df2)
    return 1 - (1 - alpha) * lo_f, 1 - (1 - alpha) * hi_f


def mean_inter_item_r(item_frame):
    """Average off-diagonal Pearson correlation among items, on complete rows.

    Reported alongside alpha because alpha rises with the number of items: a
    20-item BFAS domain and a 10-item OCTCDQ subscale are not comparable on
    alpha alone.
    """
    X = item_frame.dropna()
    if X.shape[1] < 2 or len(X) < 3:
        return np.nan
    R = X.corr().values
    off = R[~np.eye(R.shape[0], dtype=bool)]
    return float(np.nanmean(off))


def reliability_row(label, items_frame, total, n_total=N, note=""):
    X = items_frame.dropna()
    k = items_frame.shape[1]
    a = cronbach_alpha(items_frame)
    lo, hi = alpha_ci(a, len(X), k)
    obs = total.dropna()
    return {
        'Feature': label,
        'k items': k,
        'N scored': int(obs.shape[0]),
        'N complete (alpha)': int(len(X)),
        'Alpha': round(a, 3) if np.isfinite(a) else np.nan,
        '95% CI': f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(a) else "-",
        'Mean inter-item r': round(mean_inter_item_r(items_frame), 3),
        'Mdn (IQR)': f"{obs.median():.2f} ({obs.quantile(.75) - obs.quantile(.25):.2f})"
                     if len(obs) else "-",
        'N/A (%)': round(100 * (1 - obs.shape[0] / n_total), 2),
        'Note': note,
    }


# ---------------------------------------------------------------------------
# 1. Features already covered by Table 1, recomputed as a check
# ---------------------------------------------------------------------------
rows = []

# gBDI: BDI-II without the three anhedonia items, which form aBDI
for c in ['bdi16', 'bdi18']:                     # a/b sub-option items
    if c in df.columns:
        df[c] = df[c].astype(str).str.extract(r'^(\d+)', expand=False).astype(float)
bdi_gen = [f'bdi{i}' for i in range(1, 22) if i not in (4, 12, 21)]
g_items, _ = numeric(df, bdi_gen)
rows.append(reliability_row('gBDI', g_items, prorated_total(df, bdi_gen),
                            note='matches Table 1'))

# SHAPS, dichotomized as the model scores it
shaps_items = [f'shaps{i}' for i in range(1, 15)]
shaps_raw, sc = numeric(df, shaps_items)
shaps_dich = (shaps_raw <= 2).astype(float)
shaps_dich[shaps_raw.isna()] = np.nan
shaps_dich = pd.DataFrame(shaps_dich.values, columns=sc, index=df.index)
rows.append(reliability_row('SHAPS (dichotomized)', shaps_dich,
                            prorated_total(shaps_dich, sc),
                            note='matches Table 1; alpha on the 0/1 items'))

psc_items_l = [f'psc_{i}' for i in range(1, 24)]
p_items, _ = numeric(df, psc_items_l)
rows.append(reliability_row('PSC Total', p_items, prorated_total(df, psc_items_l),
                            note='matches Table 1'))

docs_items_l = [f'docs_category{c}_{i}' for c in range(1, 5) for i in range(1, 6)]
d_items, _ = numeric(df, docs_items_l)
rows.append(reliability_row('DOCS Total', d_items, prorated_total(df, docs_items_l),
                            note='matches Table 1'))

# ---------------------------------------------------------------------------
# 2. OCTCDQ subscales.  Table 1 reports the 20-item total only.
#    Odd items = harm avoidance, even items = incompleteness.
# ---------------------------------------------------------------------------
oct_all = [f'octcdqr{i}' for i in range(1, 21)]
oct_num, oct_cols = numeric(df, oct_all)
# Raw 1-5, so each subscale runs 10-50, matching OCTCDQ_RECODE_TO_0_4 = False
# in 01_clean_and_score.py. Subtracting one per item shifts each median down
# by ten and puts this table at odds with Table 3; alpha is unaffected.
oct_num = pd.DataFrame(oct_num.values, columns=oct_cols, index=df.index)

for label, idx in [('OCTCDQ Harm Avoidance', range(1, 21, 2)),
                   ('OCTCDQ Incompleteness', range(2, 21, 2))]:
    cols = [f'octcdqr{i}' for i in idx if f'octcdqr{i}' in oct_num.columns]
    rows.append(reliability_row(label, oct_num[cols],
                                prorated_total(oct_num, cols),
                                note='new; Table 1 reports the 20-item total'))

# ---------------------------------------------------------------------------
# 3. BFAS domains.  Reverse-key first, then score. Each domain is the mean of
#    two 10-item aspects, so the 20 items are the composite's item set.
# ---------------------------------------------------------------------------
bfas_cols = [f'bfas{i}' for i in range(1, 101) if f'bfas{i}' in df.columns]
bf, _ = numeric(df, bfas_cols)
bf = pd.DataFrame(bf.values, columns=bfas_cols, index=df.index)

rev_items = [1, 21, 41, 71, 16, 36, 56, 76, 2, 32, 52, 62, 82,
             17, 37, 67, 77, 87, 97, 13, 23, 33, 53, 83, 93,
             8, 48, 68, 78, 14, 24, 34, 54, 64, 29, 49, 79, 99,
             15, 45, 55, 85, 50, 60, 80, 90]
for i in rev_items:
    c = f'bfas{i}'
    if c in bf.columns:
        bf[c] = 6 - bf[c]                                # reverse on 1-5

aspect_keys = {
    "Withdrawal":      [1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
    "Volatility":      [6, 16, 26, 36, 46, 56, 66, 76, 86, 96],
    "Compassion":      [2, 12, 22, 32, 42, 52, 62, 72, 82, 92],
    "Politeness":      [7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
    "Industriousness": [3, 13, 23, 33, 43, 53, 63, 73, 83, 93],
    "Orderliness":     [8, 18, 28, 38, 48, 58, 68, 78, 88, 98],
    "Enthusiasm":      [4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
    "Assertiveness":   [9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
    "Intellect":       [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
    "Openness":        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}
domains = {
    "BFAS Neuroticism":         ["Withdrawal", "Volatility"],
    "BFAS Extraversion":        ["Enthusiasm", "Assertiveness"],
    "BFAS Conscientiousness":   ["Industriousness", "Orderliness"],
    "BFAS Agreeableness":       ["Compassion", "Politeness"],
    "BFAS Openness/Intellect":  ["Intellect", "Openness"],
}


def cols_for(aspects):
    out = []
    for a in aspects:
        out += [f'bfas{i}' for i in aspect_keys[a] if f'bfas{i}' in bf.columns]
    return out


for label, aspects in domains.items():
    cols = cols_for(aspects)
    # domain score as the model builds it: mean of the two aspect means
    aspect_means = pd.DataFrame(
        {a: bf[[f'bfas{i}' for i in aspect_keys[a] if f'bfas{i}' in bf.columns]].mean(axis=1)
         for a in aspects})
    rows.append(reliability_row(label, bf[cols], aspect_means.mean(axis=1),
                                note='new; not in Table 1'))

table = pd.DataFrame(rows)
table.to_csv(OUTPUTS / "TableS_feature_reliability.csv", index=False)
print("\n--- Internal consistency of the eleven model features ---")
print(table.drop(columns=['Note']).to_string(index=False))
print("\nsaved TableS_feature_reliability.csv")

# Aspect-level alphas, for the supplement. A 20-item domain alpha is inflated
# by length relative to a 10-item subscale; the aspects show what the domains
# are built from.
arows = []
for a, items in aspect_keys.items():
    cols = [f'bfas{i}' for i in items if f'bfas{i}' in bf.columns]
    arows.append(reliability_row(f'BFAS {a}', bf[cols], bf[cols].mean(axis=1),
                                 note='aspect'))
aspects_tbl = pd.DataFrame(arows)
aspects_tbl.to_csv(OUTPUTS / "TableS_feature_reliability_bfas_aspects.csv", index=False)
print("\n--- BFAS aspects ---")
print(aspects_tbl.drop(columns=['Note']).to_string(index=False))
print("\nsaved TableS_feature_reliability_bfas_aspects.csv")

# ---------------------------------------------------------------------------
# Check against Table 1
# ---------------------------------------------------------------------------
print("\n--- Check ---")
print("The four rows marked 'matches Table 1' should reproduce the Table 1 alpha")
print("to two decimals: gBDI .93, SHAPS .79, PSC .88, DOCS .94. A mismatch means")
print("this script and the descriptives script have drifted apart.")
low = table[table['Alpha'] < 0.70]
if len(low):
    print("\nBelow .70:")
    print(low[['Feature', 'k items', 'Alpha', '95% CI']].to_string(index=False))
else:
    print("\nAll eleven features are at or above .70.")
