# -*- coding: utf-8 -*-
"""Scan a folder for files that should not be published.

A second pair of eyes on the manifest. It looks for the three things that make a
file participant-level rather than aggregate:
an identifier column, a demographic column, or a row count matching one of the
sample sizes. None is conclusive on its own, so every hit is something to look
at rather than proof of a problem, and a clean run is not a guarantee either.
Read what it flags; do not just count the flags.

Usage   python code/check_outputs.py [folder] [--quiet]
        defaults to outputs/
"""

import argparse
import csv
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[1]
        if "__file__" in globals() else Path.cwd())

# Column names that identify a person, or that only exist in per-person data.
ID_COLUMNS = {"record_id", "participant_id", "subject_id", "id", "pid", "case_id"}
DEMOGRAPHIC_COLUMNS = {"age", "sex", "gender", "race", "ethnicity", "dob"}

# Row counts that match a group in this study. A file with one of these is not
# necessarily per-person, but it is worth opening.
SAMPLE_SIZES = {62: "controls", 263: "patients", 256: "patients with outcome data",
                318: "participants with outcome data", 325: "full sample"}

# Files known to be participant-shaped and safe: the synthetic data is generated
# from published aggregates and contains no real record.
ALLOWED = {"synthetic_clinical_data.csv"}


def inspect(path):
    """Return (header, row_count) for a CSV, or (None, None) if unreadable."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            return [h.strip().lower() for h in header], sum(1 for _ in reader)
    except (UnicodeDecodeError, csv.Error, StopIteration):
        return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", nargs="?", default=ROOT / "outputs", type=Path)
    ap.add_argument("--quiet", action="store_true", help="only print flagged files")
    args = ap.parse_args()

    folder = args.folder
    if not folder.is_dir():
        raise SystemExit(
            f"{folder} does not exist.\n\n"
            "If you ran this in an IDE console or notebook, __file__ is undefined "
            "there, so the repository root fell back to the working directory. "
            "Either run it from a terminal at the repository root\n\n"
            "    cd path/to/covariance-estimation-matters\n"
            "    python code/check_outputs.py\n\n"
            "or give the folder directly\n\n"
            "    python code/check_outputs.py path/to/covariance-estimation-matters/outputs")

    files = sorted(p for p in folder.rglob("*.csv") if p.is_file())
    if not files:
        raise SystemExit(f"{folder} exists but holds no CSV files. "
                         "Copy the summary outputs into it first.")

    flagged = []
    for path in files:
        header, n_rows = inspect(path)
        if header is None:
            print(f"  ?  {path.name}: could not read as CSV")
            continue

        reasons = []
        ids = ID_COLUMNS.intersection(header)
        demo = DEMOGRAPHIC_COLUMNS.intersection(header)
        if ids:
            reasons.append("identifier column: " + ", ".join(sorted(ids)))
        if demo:
            reasons.append("demographic column: " + ", ".join(sorted(demo)))
        if n_rows in SAMPLE_SIZES:
            reasons.append(f"{n_rows} rows, the size of the {SAMPLE_SIZES[n_rows]}")

        if path.name in ALLOWED:
            if not args.quiet:
                print(f"  ok {path.name}: {n_rows} rows (allowed: synthetic)")
            continue
        if reasons:
            flagged.append((path, reasons))
        elif not args.quiet:
            print(f"  ok {path.name}: {n_rows} rows x {len(header)} columns")

    print()
    if not flagged:
        print(f"Nothing flagged in {len(files)} files. Check the manifest as well: "
              "this looks for shape, not meaning.")
        return
    print(f"{len(flagged)} file(s) to look at before publishing:\n")
    for path, reasons in flagged:
        print(f"  {path.relative_to(folder)}")
        for r in reasons:
            print(f"      - {r}")
    print("\nA file is safe to publish only if every row is an aggregate. One row "
          "per person is not, however few columns it has.")


if __name__ == "__main__":
    main()
