# -*- coding: utf-8 -*-
"""Table 3 rows for the seven features the main descriptives script does not cover.

The two OCTCDQ subscales and the five BFAS domains, in the same five columns as
the rest of Table 3 and computed the same way. Scoring follows
01_clean_and_score.py so these rows describe the variables the model actually
uses: the OCTCDQ stays on its raw 1-5 metric (subscale range 10-50), and BFAS
aspects are plain row means, with the stricter 80% completion rule reported
beside them for comparison.

Also checks the BFAS key, since a mis-keyed item lowers a domain's alpha and
shows no other symptom. The aspect map is tested for the two structural errors a
hand-entered key usually has, and every item's corrected item-total correlation
within its own aspect is reported; a negative one is what a reversed item looks
like.

Reads   data/final_df.csv (item level, pre-imputation)
Writes  Table3_additional_rows.csv
        BFAS_item_diagnostics.csv

Usage   python code/02_descriptives_octcdq_bfas.py [--source final_df.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# parents[1] when run as a file; the working directory when pasted into an
# interactive session, where __file__ does not exist
DEFAULT_BASE = (Path(__file__).resolve().parents[1]
                if "__file__" in globals() else Path.cwd())
COMPLETION_THRESHOLD = 0.80      # same proration rule as the descriptives script
OCTCDQ_RAW = True                # matches OCTCDQ_RECODE_TO_0_4 = False in the pipeline
BFAS_COMPLETION_RULE = False     # False matches the pipeline; True applies the 80% rule

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
print(f"{args.source}: {N} rows, {df.shape[1]} columns\n")


# ---------------------------------------------------------------------------
# Helpers, matching the main descriptives script so the two cannot drift apart
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


def prorated_mean(frame, cols, threshold=COMPLETION_THRESHOLD):
    """The same completion rule, returning the item mean rather than the sum, since
    BFAS domains are means."""
    sub, cols = numeric(frame, cols)
    if not cols:
        return pd.Series(np.nan, index=frame.index)
    n_ans = sub.notna().sum(axis=1)
    out = sub.mean(axis=1, skipna=True)
    out[n_ans < np.ceil(threshold * len(cols))] = np.nan
    return out


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


def describe_row(label, total, item_frame=None, n_total=None, decimals=2):
    """One Table 3 row: N, Mdn (IQR), Range, Alpha, N/A (%)."""
    n_total = len(total) if n_total is None else n_total
    obs = total.dropna()
    q1, q3 = obs.quantile(0.25), obs.quantile(0.75)
    alpha = cronbach_alpha(item_frame) if item_frame is not None else np.nan
    d = decimals
    return {
        'Measure': label,
        'N': int(obs.shape[0]),
        'Mdn (IQR)': f"{obs.median():.{d}f} ({q3 - q1:.{d}f})" if len(obs) else "-",
        'Range': f"{obs.min():.{d}f} - {obs.max():.{d}f}" if len(obs) else "-",
        'Alpha': f"{alpha:.2f}" if pd.notna(alpha) else "-",
        'N/A (%)': f"{100 * (1 - obs.shape[0] / n_total):.2f}",
    }


def require(cols, what):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"STOP: {len(missing)} of {len(cols)} {what} columns are not in "
            f"{args.source}: {missing[:8]}{' ...' if len(missing) > 8 else ''}")


rows = []

# ---------------------------------------------------------------------------
# 1. OCTCDQ subscales.  Odd items harm avoidance, even items incompleteness,
#    matching 01_clean_and_score.py.
# ---------------------------------------------------------------------------
oct_all = [f'octcdqr{i}' for i in range(1, 21)]
require(oct_all, 'OCTCDQ')
oct_num, oct_cols = numeric(df, oct_all)
oct_num = pd.DataFrame(oct_num.values, columns=oct_cols, index=df.index)
if not OCTCDQ_RAW:
    oct_num = oct_num - 1                       # 1-5 -> 0-4, the model's metric

OCT = {'OCTCDQ Harm Avoidance': [f'octcdqr{i}' for i in range(1, 21, 2)],
       'OCTCDQ Incompleteness': [f'octcdqr{i}' for i in range(2, 21, 2)]}
for label, cols in OCT.items():
    rows.append(describe_row(label, prorated_total(oct_num, cols), oct_num[cols],
                             n_total=N))

# ---------------------------------------------------------------------------
# 2. BFAS.  Reverse-key, then aspects, then domains as the mean of two aspect
#    means, which is how the Method describes the domain scores.
# ---------------------------------------------------------------------------
bfas_all = [f'bfas{i}' for i in range(1, 101)]
require(bfas_all, 'BFAS')
bf, bfas_cols = numeric(df, bfas_all)
bf = pd.DataFrame(bf.values, columns=bfas_cols, index=df.index)

REV = [1, 21, 41, 71, 16, 36, 56, 76, 2, 32, 52, 62, 82,
       17, 37, 67, 77, 87, 97, 13, 23, 33, 53, 83, 93,
       8, 48, 68, 78, 14, 24, 34, 54, 64, 29, 49, 79, 99,
       15, 45, 55, 85, 50, 60, 80, 90]

ASPECTS = {
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
DOMAINS = {
    "BFAS Neuroticism":        ["Withdrawal", "Volatility"],
    "BFAS Extraversion":       ["Enthusiasm", "Assertiveness"],
    "BFAS Conscientiousness":  ["Industriousness", "Orderliness"],
    "BFAS Agreeableness":      ["Compassion", "Politeness"],
    "BFAS Openness/Intellect": ["Intellect", "Openness"],
}

# Check the key before it touches the data.
flat = [i for items in ASPECTS.values() for i in items]
if sorted(flat) != list(range(1, 101)):
    raise SystemExit("STOP: the aspect map does not partition items 1 to 100.")
if any(len(v) != 10 for v in ASPECTS.values()):
    raise SystemExit("STOP: an aspect does not have exactly ten items.")
if len(set(REV)) != len(REV):
    raise SystemExit(f"STOP: the reverse-key list repeats an item: "
                     f"{sorted({i for i in REV if REV.count(i) > 1})}")
if not set(REV) <= set(range(1, 101)):
    raise SystemExit(f"STOP: the reverse-key list is outside items 1 to 100: "
                     f"{sorted(set(REV) - set(range(1, 101)))}")
# The ten aspects repeat in a fixed order, so each aspect's items are congruent
# modulo ten. A hand-typed key usually breaks this first.
for a, items in ASPECTS.items():
    if len({i % 10 for i in items}) != 1:
        raise SystemExit(f"STOP: {a} does not take every tenth item: {items}")
owner = {i: a for a, items in ASPECTS.items() for i in items}
per_aspect = {a: sum(1 for i in REV if owner[i] == a) for a in ASPECTS}
print(f"key check: 10 aspects x 10 items, {len(REV)} of 100 items reverse-keyed")
print("           per aspect: " + ", ".join(f"{a} {n}" for a, n in per_aspect.items()) + "\n")

bf_rev = bf.copy()
for i in REV:
    bf_rev[f'bfas{i}'] = 6 - bf_rev[f'bfas{i}']     # reverse on the 1-5 scale

def aspect_score(cols):
    return (prorated_mean(bf_rev, cols) if BFAS_COMPLETION_RULE
            else bf_rev[cols].mean(axis=1))


for label, aspects in DOMAINS.items():
    aspect_means = pd.DataFrame(
        {a: aspect_score([f'bfas{i}' for i in ASPECTS[a]]) for a in aspects})
    domain = aspect_means.mean(axis=1)
    domain[aspect_means.isna().any(axis=1)] = np.nan     # both aspects required
    items = [f'bfas{i}' for a in aspects for i in ASPECTS[a]]
    rows.append(describe_row(label, domain, bf_rev[items], n_total=N))

table = pd.DataFrame(rows)
table.to_csv(OUTPUTS / "Table3_additional_rows.csv", index=False)
print("--- Rows for Table 3 ---")
print(table.to_string(index=False))
print("\nsaved Table3_additional_rows.csv")

# ---------------------------------------------------------------------------
# 3. Does the reverse key hold up in the data?
#    Within an aspect every item should correlate positively with the other nine.
#    An item keyed the wrong way round shows up negative and nothing else reveals it.
# ---------------------------------------------------------------------------
diag = []
for a, items in ASPECTS.items():
    cols = [f'bfas{i}' for i in items]
    X = bf_rev[cols].dropna()
    for c in cols:
        rest = X[[o for o in cols if o != c]].sum(axis=1)
        diag.append({'Aspect': a, 'Item': c,
                     'Reverse-keyed': int(c in [f'bfas{i}' for i in REV]),
                     'N complete': len(X),
                     'Corrected item-total r': round(float(X[c].corr(rest)), 3)})
diag = pd.DataFrame(diag)
diag.to_csv(OUTPUTS / "BFAS_item_diagnostics.csv", index=False)

bad = diag[diag['Corrected item-total r'] < 0]
weak = diag[(diag['Corrected item-total r'] >= 0) & (diag['Corrected item-total r'] < .15)]
print("\n--- Reverse-key check ---")
if len(bad):
    print(f"{len(bad)} item(s) correlate NEGATIVELY with the rest of their aspect, "
          "which is what a mis-keyed item looks like. Check these against the "
          "published key before reporting the alphas:")
    print(bad.to_string(index=False))
else:
    print("No BFAS item correlates negatively with the rest of its aspect, so the "
          "reverse key is consistent with the data.")
if len(weak):
    print(f"\n{len(weak)} item(s) correlate below .15 with the rest of their aspect. "
          "Weak rather than mis-keyed, but worth a look:")
    print(weak.to_string(index=False))
print("saved BFAS_item_diagnostics.csv")

# ---------------------------------------------------------------------------
# 4. What the completion rule costs, since the model scored BFAS without one.
# ---------------------------------------------------------------------------
other = {}
for label, aspects in DOMAINS.items():
    am = pd.DataFrame(
        {a: (bf_rev[[f'bfas{i}' for i in ASPECTS[a]]].mean(axis=1) if BFAS_COMPLETION_RULE
             else prorated_mean(bf_rev, [f'bfas{i}' for i in ASPECTS[a]]))
         for a in aspects})
    d = am.mean(axis=1)
    d[am.isna().any(axis=1)] = np.nan
    other[label] = int(d.notna().sum())
used = "no completion rule, as the pipeline scores it" if not BFAS_COMPLETION_RULE \
    else "the 80% rule used elsewhere in Table 3"
alt = "the 80% rule used elsewhere in Table 3" if not BFAS_COMPLETION_RULE \
    else "no completion rule, as the pipeline scores it"
print("\n--- Completion rule ---")
print(f"Reported N uses {used}.")
print(f"Beside it, N under {alt}:")
differs = False
for r in rows[2:]:
    lab = r['Measure']
    same = r['N'] == other[lab]
    differs |= not same
    print(f"  {lab:28s} {r['N']:>4}   vs {other[lab]:>4}" + ("" if same else "   <-- differs"))
print("The two agree, so the choice does not matter here." if not differs else
      "They differ, so decide which N Table 3 should report and say so in its note.")

print("\n--- Settings used ---")
print(f"OCTCDQ metric: {'RAW 1-5, subscale range 10-50' if OCTCDQ_RAW else 'SHIFTED 0-4, subscale range 0-40'}"
      f"  ({'matches' if OCTCDQ_RAW else 'does NOT match'} OCTCDQ_RECODE_TO_0_4 = False in the pipeline)")
print("BFAS domains are means of two aspect means on the 1-5 item metric, so "
      "their Mdn, IQR and Range are on a 1-5 scale while every other row in "
      "Table 3 is a scale total. Table 3's note should say so.")
