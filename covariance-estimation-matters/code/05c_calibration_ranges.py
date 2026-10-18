"""Calibration ranges: where the observed control-sample summaries fall within each
Study 1 condition's spread across seeds.

The manuscript (Calibration Against the Study 1 Conditions) and Supplement S1 treat
each simulated condition as a distribution across its 30 seeds rather than as a
single value. They then count how many of six observed quantities fall inside that
distribution's 2.5th to 97.5th percentile range. This script makes those counts
reproducible from the archived files.

The six quantities are two summaries for each of the sample covariance, Ledoit-Wolf
and default MCD, with all distances taken about the HC mean:
- the median-distance ratio: the control sample's median squared Mahalanobis
  distance divided by the chi-square median on 11 df;
- the percentage of controls beyond the chi-square .95 quantile.
Table 4 reports both.

Observed values are the HC-mean rows of TableS_chi2_calibration.csv, written by 03.
Simulated values come from Study1_runs.csv, written by 05, at the reference size of
the control sample (N = 62).

Percentiles use numpy's default rule (linear interpolation between order
statistics), and the range includes its endpoints. The observed values are stored to
four decimals, so a value within 5e-5 of a bound counts as on the bound. The counts
depend on the percentile rule.
- At the uniform .50 floor, the Ledoit-Wolf ratio lies above only one of the 30
  seeds. It falls outside the range if each percentile is rounded to the nearest
  order statistic (numpy's method="nearest"), which gives 5 of 6.
- Under the classical nearest-rank rule (method="inverted_cdf"), the range runs from
  the lowest to the highest seed. The .50-floor count is then unchanged, but the
  matched-correlation count rises from 2 to 3, because one seed's MCD exceedance
  equals the observed 62.9%.
- At the matched floors, the sample-covariance exceedance equals the upper bound.

Where MCD fits are rank-deficient (most .75-floor fits, a third of matched-floor
fits), the MCD ranges can move on another machine (see README). Counts for those
conditions are the least stable.

Reads   outputs/TableS_chi2_calibration.csv, outputs/Study1_runs.csv
Writes  TableS_calibration_ranges.csv (one row per condition x estimator x quantity)

Usage   python code/05c_calibration_ranges.py [--base PATH] [--n 62]
        Run after 03 and 05.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BASE = (Path(__file__).resolve().parents[1]
                if "__file__" in globals() else Path.cwd())
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=DEFAULT_BASE)
ap.add_argument("--n", type=int, default=62,
                help="reference sample size to compare at (the control sample's N)")
args = ap.parse_args()
OUT = Path(args.base) / "outputs"

LO, HI = 2.5, 97.5
TOL = 5e-5          # half a unit in the fourth decimal of the observed values

# estimator label in TableS_chi2_calibration.csv -> column prefix in Study1_runs.csv
ESTIMATORS = {"sample": "sample", "ledoit_wolf": "ledoit_wolf", "mcd_default": "mcd"}
# statistic in TableS_chi2_calibration.csv -> column suffix in Study1_runs.csv
STATS = {"median_ratio": "ref_med_ratio", "pct_hc_above_q95": "ref_pct_above_q95"}

# Counts stated in the manuscript and Supplement S1, printed alongside for comparison.
REPORTED = {"gaussian": 0, "floor10": 0, "floor25": 1, "floor50": 6,
            "floor_matched": 6, "floor_matched_corr": 2}

obs = pd.read_csv(OUT / "TableS_chi2_calibration.csv")
obs = obs[obs["center"] == "HC mean"].set_index("estimator")
missing = set(ESTIMATORS) - set(obs.index)
if missing:
    raise SystemExit(f"TableS_chi2_calibration.csv has no HC-mean row for {sorted(missing)}")

runs = pd.read_csv(OUT / "Study1_runs.csv")
runs = runs[runs["n"] == args.n]
if runs.empty:
    raise SystemExit(f"Study1_runs.csv has no runs at n = {args.n}")

rows = []
for cond, g in runs.groupby("condition", sort=False):
    for est, prefix in ESTIMATORS.items():
        for stat, suffix in STATS.items():
            x = g[f"{prefix}_{suffix}"].dropna().to_numpy()
            lo, hi = np.percentile(x, [LO, HI])
            o = float(obs.loc[est, stat])
            rows.append(dict(
                condition=cond, estimator=est, statistic=stat, n_ref=args.n,
                n_seeds=len(x), observed=o, p2_5=lo, p97_5=hi,
                seed_min=x.min(), seed_max=x.max(),
                inside=bool(lo - TOL <= o <= hi + TOL),
            ))
res = pd.DataFrame(rows)
res.to_csv(OUT / "TableS_calibration_ranges.csv", index=False)

counts = res.groupby("condition", sort=False)["inside"].agg(["sum", "size"])
print(f"Observed quantities inside the {LO:g}th-{HI:g}th percentile range, n = {args.n}")
for cond, r in counts.iterrows():
    note = ""
    if cond in REPORTED:
        note = ("  (as reported)" if r["sum"] == REPORTED[cond]
                else f"  (manuscript reports {REPORTED[cond]})")
    print(f"  {cond:<20} {int(r['sum'])} of {int(r['size'])}{note}")
print("Saved TableS_calibration_ranges.csv to", OUT)
