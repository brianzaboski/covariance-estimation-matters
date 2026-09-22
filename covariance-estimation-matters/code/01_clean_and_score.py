# -*- coding: utf-8 -*-
"""Score the questionnaires and build the analytic file.

Scores the eleven model features and the Q-LES-Q-SF outcome from the item-level
export, imputes missing feature values with group-aware chained equations, and
writes one row per participant.

Two scoring choices are configurable below, the OCTCDQ metric and whether to
impute the outcome. Neither changes a standardized result. Other decisions that
are not self-evident are explained where they are made.

Reads   data/final_df.csv
Writes  data/imputed_clinical_data.csv

Usage   python code/01_clean_and_score.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# Paths are relative to the repository root, so the scripts run from anywhere.
# parents[1] when run as a file; the working directory when pasted into an
# interactive session, where __file__ does not exist
ROOT = (Path(__file__).resolve().parents[1]
        if "__file__" in globals() else Path.cwd())
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
IN_PATH  = DATA / "final_df.csv"
OUT_PATH = DATA / "imputed_clinical_data.csv"

COMPLETION_THRESHOLD = 0.80   # min fraction of items answered for a valid total
OCTCDQ_RECODE_TO_0_4 = False  # False: keep as administered (1-5 -> 10-50), fix report.
                              # True:  subtract 1 per item -> 0-40 for norm-comparability.
                              # Does NOT affect any standardized result either way.
IMPUTE_OUTCOME       = False  # False: leave Q-LES-Q-SF missing where absent; the
                              #   regression handles it by listwise deletion (cleaner:
                              #   the DV is not imputed). Report the actual regression n.
                              # True:  impute Q-LES-Q-SF alongside the features to
                              #   preserve n exactly (imputes the outcome under single
                              #   imputation - a minor blemish).
RANDOM_STATE = 42
MAX_ITER = 10

print("--- Starting Data Scoring and Imputation Pipeline (revised) ---")

# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
df = pd.read_csv(IN_PATH)

# --------------------------------------------------------------------------
# Helper: prorated scale scoring
# --------------------------------------------------------------------------
def score_scale(frame, items, new_col, threshold=COMPLETION_THRESHOLD):
    """Prorated total: the mean of answered items times the item count, or NaN if
    fewer than `threshold` of items were answered. Equals a plain sum when the
    scale is complete."""
    cols = [c for c in items if c in frame.columns]
    if not cols:
        frame[new_col] = np.nan
        return frame
    sub = frame[cols].apply(pd.to_numeric, errors="coerce")
    n_items = len(cols)
    n_answered = sub.notna().sum(axis=1)
    total = sub.mean(axis=1, skipna=True) * n_items
    total[n_answered < np.ceil(threshold * n_items)] = np.nan
    frame[new_col] = total
    return frame

# --------------------------------------------------------------------------
# 2. BDI-II -> gBDI (general depression, anhedonia items removed)
# --------------------------------------------------------------------------
print("Scoring gBDI...")
def clean_bdi_string(value):
    if pd.isna(value):
        return np.nan
    match = re.match(r"^(\d+)", str(value).strip())  # '1b' -> 1
    return float(match.group(1)) if match else np.nan

for col in ["bdi16", "bdi18"]:  # the two BDI-II items with a/b sub-options
    if col in df.columns:
        df[col] = df[col].apply(clean_bdi_string)

bdi_all = [f"bdi{i}" for i in range(1, 22)]
bdi_anh = ["bdi4", "bdi12", "bdi21"]          # anhedonia items -> reserved for SHAPS
bdi_gen = [c for c in bdi_all if c not in bdi_anh]
df = score_scale(df, bdi_gen, "gbdi_total_score")

# --------------------------------------------------------------------------
# 3. SHAPS (dichotomous Snaith scoring)  [disagree = anhedonic = 1]
# --------------------------------------------------------------------------
print("Scoring SHAPS...")
shaps_cols = [f"shaps{i}" for i in range(1, 15)]
present = [c for c in shaps_cols if c in df.columns]
raw = df[present].apply(pd.to_numeric, errors="coerce")
# Snaith scoring: disagree or strongly disagree (1 or 2 in REDCap) indicates
# anhedonia and scores 1; agree or strongly agree (3 or 4) scores 0. Check the
# anchor direction against the codebook if the export format changes.
dich = (raw <= 2).astype(float)
dich[raw.isna()] = np.nan
dich.columns = present
df = score_scale(pd.concat([df.drop(columns=present, errors="ignore"), dich], axis=1),
                 present, "shaps_total_score_snaith")

# --------------------------------------------------------------------------
# 4. PSC, QLES, OCTCDQ
# --------------------------------------------------------------------------
print("Scoring PSC, QLES, OCTCDQ...")
df = score_scale(df, [f"psc_{i}" for i in range(1, 24)], "psc_total_score")

# Summary total is items 1-14 (range 14-70). Items 15-16 are excluded by convention.
df = score_scale(df, [f"qles{i}" for i in range(1, 15)], "qles_total_score")

octcdq_items = [f"octcdqr{i}" for i in range(1, 21)]
if OCTCDQ_RECODE_TO_0_4:
    for c in octcdq_items:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") - 1  # 1-5 -> 0-4
df = score_scale(df, [f"octcdqr{i}" for i in range(2, 21, 2)], "octcdq_incompleteness_score")  # even
df = score_scale(df, [f"octcdqr{i}" for i in range(1, 21, 2)], "octcdq_harm_score")            # odd

# --------------------------------------------------------------------------
# 5. BFAS domains
#    The reverse key and the item-to-aspect map are checked against the data by
#    02_descriptives_octcdq_bfas.py. Do not cross-check against REDCap's own
#    bfas_*_w / bfas_*_v columns, which use a different signed metric.
# --------------------------------------------------------------------------
print("Scoring BFAS domains...")
rev_items = [1, 21, 41, 71, 16, 36, 56, 76, 2, 32, 52, 62, 82,
             17, 37, 67, 77, 87, 97, 13, 23, 33, 53, 83, 93,
             8, 48, 68, 78, 14, 24, 34, 54, 64, 29, 49, 79, 99,
             15, 45, 55, 85, 50, 60, 80, 90]
for i in range(1, 101):
    c = f"bfas{i}"
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
for i in rev_items:
    c = f"bfas{i}"
    if c in df.columns:
        df[c] = 6 - df[c]  # reverse on 1-5

aspect_keys = {
    "bfas_withdrawal":      [1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
    "bfas_volatility":      [6, 16, 26, 36, 46, 56, 66, 76, 86, 96],
    "bfas_compassion":      [2, 12, 22, 32, 42, 52, 62, 72, 82, 92],
    "bfas_politeness":      [7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
    "bfas_industriousness": [3, 13, 23, 33, 43, 53, 63, 73, 83, 93],
    "bfas_orderliness":     [8, 18, 28, 38, 48, 58, 68, 78, 88, 98],
    "bfas_enthusiasm":      [4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
    "bfas_assertiveness":   [9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
    "bfas_intellect":       [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
    "bfas_openness":        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}
for aspect, items in aspect_keys.items():
    cols = [f"bfas{i}" for i in items if f"bfas{i}" in df.columns]
    df[aspect] = df[cols].mean(axis=1)

df["bfas_domain_neuroticism"]        = df[["bfas_withdrawal", "bfas_volatility"]].mean(axis=1)
df["bfas_domain_agreeableness"]      = df[["bfas_compassion", "bfas_politeness"]].mean(axis=1)
df["bfas_domain_conscientiousness"]  = df[["bfas_industriousness", "bfas_orderliness"]].mean(axis=1)
df["bfas_domain_extraversion"]       = df[["bfas_enthusiasm", "bfas_assertiveness"]].mean(axis=1)
df["bfas_domain_openness_intellect"] = df[["bfas_intellect", "bfas_openness"]].mean(axis=1)

# DOCS total comes prescored from REDCap
df["docs_total"] = pd.to_numeric(df["docs_total"], errors="coerce")

# --------------------------------------------------------------------------
# 6. Assemble analytic frame
# --------------------------------------------------------------------------
FEATURES = [
    "gbdi_total_score", "shaps_total_score_snaith", "psc_total_score",
    "docs_total", "octcdq_incompleteness_score", "octcdq_harm_score",
    "bfas_domain_neuroticism", "bfas_domain_extraversion",
    "bfas_domain_conscientiousness", "bfas_domain_agreeableness",
    "bfas_domain_openness_intellect",
]  # the 11-dimensional phenotypic feature space
keep = ["record_id", "group", "age", "sex"] + FEATURES + ["qles_total_score"]
da = df[keep].copy()

# --------------------------------------------------------------------------
# 7. Demographics: sex categorical (mode), age continuous (median)
# --------------------------------------------------------------------------
print("Handling demographics (sex -> 3 categories, age continuous)...")
# REDCap codebook: 1=Female, 2=Male, 3=non-binary, 4=prefer-not-to-say,
# collapsed to Female, Male, Other. An earlier version had 1 and 2 the wrong way
# round, which inverted the sex counts and swapped the labels on the regression's
# sex coefficients. 03_phase2_analyses.py checks this against Table 1 and stops.
sex_num = pd.to_numeric(da["sex"], errors="coerce")
da["sex"] = sex_num.map({1: "Female", 2: "Male", 3: "Other", 4: "Other"})
# Missing sex takes the mode; dummy-coded downstream.
da["sex"] = da["sex"].fillna(da["sex"].mode(dropna=True).iloc[0])
da["age"] = pd.to_numeric(da["age"], errors="coerce")
da["age"] = SimpleImputer(strategy="median").fit_transform(da[["age"]]).ravel()

# --------------------------------------------------------------------------
# 8. Group-aware iterative imputation of the 11 features
# --------------------------------------------------------------------------
print("Imputing features (group retained as a predictor)...")
impute_cols = FEATURES + (["qles_total_score"] if IMPUTE_OUTCOME else [])
mat = da[impute_cols].copy()
mat["is_ocd"] = (da["group"] == "OCD").astype(int)  # imputation respects group

imputer = IterativeImputer(estimator=BayesianRidge(),
                           random_state=RANDOM_STATE, max_iter=MAX_ITER)
imputed = imputer.fit_transform(mat)
da[impute_cols] = imputed[:, :len(impute_cols)]  # drop the is_ocd helper column

# --------------------------------------------------------------------------
# 9. Clip imputed features to valid ranges
# --------------------------------------------------------------------------
octcdq_bounds = (0, 40) if OCTCDQ_RECODE_TO_0_4 else (10, 50)
bounds = {
    "gbdi_total_score": (0, 54), "shaps_total_score_snaith": (0, 14),
    "psc_total_score": (23, 92), "docs_total": (0, 80),
    "octcdq_incompleteness_score": octcdq_bounds, "octcdq_harm_score": octcdq_bounds,
    "bfas_domain_neuroticism": (1, 5), "bfas_domain_extraversion": (1, 5),
    "bfas_domain_conscientiousness": (1, 5), "bfas_domain_agreeableness": (1, 5),
    "bfas_domain_openness_intellect": (1, 5),
}
for c, (lo, hi) in bounds.items():
    da[c] = da[c].clip(lo, hi)

# --------------------------------------------------------------------------
# 10. Save
# --------------------------------------------------------------------------
da.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH}")
print(f"Rows: {len(da)} | groups: {da['group'].value_counts().to_dict()}")
print(f"Q-LES-Q-SF missing (un-imputed outcome): {da['qles_total_score'].isna().sum()}")