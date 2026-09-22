# -*- coding: utf-8 -*-
"""Check regenerated outputs against the values the manuscript reports.

Run after a full pass over the real data. Every script can finish without error
and still produce the wrong numbers, so this compares a handful of reported
quantities against what is now in outputs/. Anything missing is skipped rather
than failed, so a partial run gives a partial report.

These are the values in the submitted manuscript. Under scikit-learn other than
1.8.0 the MCD rows will differ; that is the version dependency, not a failure.

Usage   python code/check_reproduction.py [folder]
        defaults to outputs/
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = (Path(__file__).resolve().parents[1]
        if "__file__" in globals() else Path.cwd())


def rows(df, **eq):
    for col, val in eq.items():
        df = df[df[col] == val]
    return df


CHECKS = []


def check(name, filename, fn, expected, tol):
    CHECKS.append((name, filename, fn, expected, tol))


def check_range(name, filename, fn, lo, hi):
    """For quantities that move between machines. The .75-floor singular rate
    turns on whether a near-zero eigenvalue falls either side of a numerical
    cutoff, so it shifts by a seed or two across platforms; the manuscript
    states it in words for that reason."""
    CHECKS.append((name, filename, fn, (lo, hi), None))


check("Study 1: MCD median ratio, .50 floor, n = 5,000", "Study1_summary.csv",
      lambda d: rows(d, condition="floor50", n=5000)["mcd_ref_med_ratio_median"].iloc[0],
      2.33, 0.01)
check("Study 1: benchmark median ratio, .50 floor, n = 5,000", "Study1_summary.csv",
      lambda d: rows(d, condition="floor50", n=5000)["oracle_ref_med_ratio_median"].iloc[0],
      0.88, 0.01)
check_range("Study 1: MCD singular fits, .75 floor, n = 62 (over half)",
            "Study1_summary.csv",
            lambda d: rows(d, condition="floor75", n=62)["mcd_pct_singular"].iloc[0],
            50.0, 65.0)
check_range("Study 1: MCD singular fits, .75 floor, n = 1,000 (over nine in ten)",
            "Study1_summary.csv",
            lambda d: rows(d, condition="floor75", n=1000)["mcd_pct_singular"].iloc[0],
            90.0, 100.0)
# TableS_estimator_comparison_realdata has one row per estimator and centering;
# take the HC-centred MCD row. The TOST table is a name/value pair table.
def mcd_row(d, col):
    r = d[(d["estimator"] == "mcd_default") & (d["center"] == "HC mean")]
    return r[col].iloc[0]


def tost(d, key):
    return float(d.set_index(d.columns[0])["value"].loc[key])


check("Study 2: MCD condition number, real controls",
      "TableS_estimator_comparison_realdata.csv",
      lambda d: mcd_row(d, "condition_number"), 172.2338, 0.01)
check("Study 2: MCD retained support", "TableS_estimator_comparison_realdata.csv",
      lambda d: mcd_row(d, "n_retained"), 38, 0)
check("Study 2: partial correlation", "TableS_partial_r_interval_tost.csv",
      lambda d: tost(d, "partial_r"), 0.0637, 0.001)
check("Study 2: upper 90% limit", "TableS_partial_r_interval_tost.csv",
      lambda d: tost(d, "ci90_high"), 0.1667, 0.001)
check("Study 2: variance at the upper limit (%)", "TableS_partial_r_interval_tost.csv",
      lambda d: 100 * tost(d, "dR2_at_ci90_high"), 1.26, 0.05)
check("Sweep: singular MCD fits out of 300", "TableS_support_fraction_sweep_byseed.csv",
      lambda d: int(d[(d["setting"].str.contains("real")) &
                      (d["estimator"] == "MCD")]["is_singular"].sum()),
      136, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", default=ROOT / "outputs", type=Path)
    args = ap.parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"{args.folder} does not exist. Run from the repository root.")

    passed = failed = skipped = 0
    for name, filename, fn, expected, tol in CHECKS:
        path = args.folder / filename
        if not path.exists():
            print(f"  -  {name}\n         skipped, {filename} not found")
            skipped += 1
            continue
        try:
            got = float(fn(pd.read_csv(path)))
        except Exception as exc:                       # column renamed, row missing
            print(f"  !  {name}\n         could not read it: {exc}")
            failed += 1
            continue
        if tol is None:
            lo, hi = expected
            if lo <= got <= hi:
                print(f"  ok {name}: {got:g}")
                passed += 1
            else:
                print(f"  X  {name}: got {got:g}, expected between {lo:g} and {hi:g}")
                failed += 1
        elif abs(got - expected) <= tol:
            print(f"  ok {name}: {got:g}")
            passed += 1
        else:
            print(f"  X  {name}: got {got:g}, manuscript reports {expected:g}")
            failed += 1

    print(f"\n{passed} matched, {failed} did not, {skipped} skipped.")
    if failed:
        print("A mismatch is worth tracing before publishing. Check the scikit-learn "
              "version first: MinCovDet changed across releases and the MCD rows move "
              "with it.")


if __name__ == "__main__":
    main()
