# -*- coding: utf-8 -*-
"""Generate a synthetic dataset with the structure of the analytic file.

The registry data cannot be released. This builds a stand-in from published
aggregates only: per-group means, standard deviations and floor prevalences from
TableS_feature_descriptives_by_group, and the latent correlation matrix derived
in 04_invert_r.py. No real record is resampled, copied or perturbed, so nothing
here carries participant information beyond the summaries already in the paper.

Each feature is drawn as a correlated normal, censored at the normal quantile of
its group's observed floor prevalence, then mapped onto the scale so that the
censored point lands exactly on the scale minimum and the spread matches the
observed standard deviation. The mean is whatever that mapping implies, and the
verification step below reports how close it lands.

What this reproduces: the floor geometry, the correlation structure, the group
separation, and the column contract the analysis scripts expect. Scripts 03, 06
and 07 run against it end to end.

What it does not reproduce: any reported estimate. MCD will collapse onto a
floored subset and Ledoit-Wolf will not, because that follows from the geometry,
but the particular split, the singular-fit counts and every coefficient will
differ from the paper.

Scripts 01 and 02 need the item-level export and cannot run against this file.

Writes  data/synthetic/synthetic_clinical_data.csv
        data/synthetic/synthetic_data_constants.json

Usage   python data/synthetic/make_synthetic_data.py [--seed 20260920]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# parents[2] when run as a file; the working directory when pasted into an
# interactive session, where __file__ does not exist
ROOT = (Path(__file__).resolve().parents[2]
        if "__file__" in globals() else Path.cwd())
OUT_DIR = ROOT / "data" / "synthetic"

FEATURES = [
    "gbdi_total_score", "shaps_total_score_snaith", "psc_total_score",
    "docs_total", "octcdq_incompleteness_score", "octcdq_harm_score",
    "bfas_domain_neuroticism", "bfas_domain_extraversion",
    "bfas_domain_conscientiousness", "bfas_domain_agreeableness",
    "bfas_domain_openness_intellect",
]

# Per-group targets, from TableS_feature_descriptives_by_group.csv.
# mean, sd, percentage at the scale minimum.
TARGETS = {
    "HC": {
        "gbdi_total_score":              (2.242,  3.012, 38.710),
        "shaps_total_score_snaith":      (0.129,  0.665, 93.548),
        "psc_total_score":              (25.758,  3.108, 29.032),
        "docs_total":                    (2.419,  4.448, 45.161),
        "octcdq_incompleteness_score":  (15.210,  5.505, 22.581),
        "octcdq_harm_score":            (13.710,  4.248, 30.645),
        "bfas_domain_neuroticism":       (2.163,  0.582,  0.000),
        "bfas_domain_extraversion":      (3.731,  0.419,  0.000),
        "bfas_domain_conscientiousness": (3.681,  0.417,  0.000),
        "bfas_domain_agreeableness":     (4.060,  0.414,  0.000),
        "bfas_domain_openness_intellect": (3.844, 0.459,  0.000),
    },
    "OCD": {
        "gbdi_total_score":             (16.574,  9.521,  2.281),
        "shaps_total_score_snaith":      (0.949,  1.759, 61.597),
        "psc_total_score":              (37.115,  7.967,  1.901),
        "docs_total":                   (28.452, 12.259,  0.000),
        "octcdq_incompleteness_score":  (34.400,  7.935,  0.760),
        "octcdq_harm_score":            (34.560,  8.008,  1.141),
        "bfas_domain_neuroticism":       (3.405,  0.590,  0.000),
        "bfas_domain_extraversion":      (3.289,  0.569,  0.000),
        "bfas_domain_conscientiousness": (3.349,  0.549,  0.000),
        "bfas_domain_agreeableness":     (3.996,  0.463,  0.000),
        "bfas_domain_openness_intellect": (3.809, 0.442,  0.000),
    },
}

N = {"HC": 62, "OCD": 263}

# lower bound, upper bound, rounding grain
SCALE = {
    "gbdi_total_score":              (0, 54, 1.0),
    "shaps_total_score_snaith":      (0, 14, 1.0),
    "psc_total_score":              (23, 92, 1.0),
    "docs_total":                    (0, 80, 1.0),
    "octcdq_incompleteness_score":  (10, 50, 1.0),
    "octcdq_harm_score":            (10, 50, 1.0),
    "bfas_domain_neuroticism":       (1,  5, 0.05),
    "bfas_domain_extraversion":      (1,  5, 0.05),
    "bfas_domain_conscientiousness": (1,  5, 0.05),
    "bfas_domain_agreeableness":     (1,  5, 0.05),
    "bfas_domain_openness_intellect": (1, 5, 0.05),
}

# Latent correlation matrix for the control features, from 04_invert_r.py. It is
# the latent structure, not the observed one: observed correlations are computed
# on data that already carry the floors, so censoring them again would attenuate
# everything twice. Applied to both groups, since only the control matrix is
# published; the synthetic clinical group therefore has control-like structure.
LATENT_R = np.array([
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

# Outcome. The fitted Step 3 equation from Table4a_coefficients.csv, applied on
# the natural scale, with residual noise scaled to the reported Step 2 R-squared.
# Nothing here is assumed: the coefficients come from the regression, the
# R-squared from Table 5b. The deviation term is omitted because its coefficient
# is 0.0092 and the score does not exist until the analysis runs.
QLES = dict(
    intercept=59.7596,
    b_age=-0.0194,
    b_sex={"Female": 0.0, "Male": -1.5920, "Other": 2.8208},
    b_gbdi=-0.6124,
    b_docs=-0.0673,
    r2=0.546,
    lo=14, hi=70,
)
N_MISSING_OUTCOME = 7           # 263 patients, 256 with complete outcome data

# Demographics, from the analytic file. Age parameters exclude the participants
# sitting exactly on the imputed median, which would otherwise inflate the count
# and shrink the spread. Sex proportions are post mode-fill, as the analytic file
# stores them, so they already carry the 27 filled values.
AGE = {
    "HC":  dict(mean=33.42, sd=12.62, lo=18, hi=70, n_at_median=3),
    "OCD": dict(mean=30.68, sd=11.33, lo=18, hi=70, n_at_median=46),
}
AGE_MEDIAN = 28                 # the value SimpleImputer wrote
SEX_P = {
    "HC":  {"Female": 0.581, "Male": 0.419, "Other": 0.000},
    "OCD": {"Female": 0.719, "Male": 0.259, "Other": 0.023},
}
N_SEX_MISSING = 27              # sex recorded for 298 of 325, before mode-fill


def draw_group(group, rng):
    """One group: correlated normals, censored at the observed floor prevalence,
    mapped onto each scale."""
    n = N[group]
    chol = np.linalg.cholesky(LATENT_R)
    z = rng.standard_normal((n, len(FEATURES))) @ chol.T

    out = {}
    for j, f in enumerate(FEATURES):
        mean, sd, pct_min = TARGETS[group][f]
        lo, hi, grain = SCALE[f]
        x = mean + sd * z[:, j]
        if pct_min > 0:
            # Put exactly the observed proportion on the minimum by flooring the
            # lowest draws. Anchoring a censored normal at the minimum instead
            # would fix the floor and the spread but throw the mean off badly
            # wherever the real distribution is skewed, which is most of them.
            k = int(round(pct_min / 100.0 * n))
            floored = np.zeros(n, bool)
            if k:
                floored[np.argsort(x)[:k]] = True
                x[floored] = lo
            # Flooring compresses the spread and shifts the mean, so solve the
            # location and scale of what is left above the floor to put both
            # back on target. Two free parameters, two targets, one solution.
            m = int((~floored).sum())
            if m > 2:
                c = (n * mean - k * lo) / m                  # unfloored mean
                M = (k * lo + m * c) / n                     # overall mean
                d2 = ((n - 1) * sd ** 2 - k * (lo - M) ** 2
                      - m * (c - M) ** 2) / (m - 1)
                if d2 > 0:
                    u = x[~floored]
                    u = (u - u.mean()) / u.std(ddof=1)
                    x[~floored] = c + np.sqrt(d2) * u
            x = np.clip(x, lo, hi)
            x = np.round(x / grain) * grain
            # rounding can drop an unfloored value onto the minimum, which would
            # inflate the floor proportion past its target
            strays = (~floored) & (x <= lo)
            x[strays] = lo + grain
        else:
            # never let a clipped value sit exactly on the minimum: these
            # features have no floor and the analysis depends on that
            x = np.clip(x, lo + grain, hi)
            x = np.round(x / grain) * grain
        out[f] = np.round(x, 2)   # keep the grain clean of float dust
    return pd.DataFrame(out)


def add_outcome(df, rng):
    """Q-LES-Q-SF from the fitted regression, with residual noise sized to the
    reported R-squared."""
    yhat = (QLES["intercept"]
            + QLES["b_age"] * df["age"].to_numpy(float)
            + df["sex"].map(QLES["b_sex"]).to_numpy(float)
            + QLES["b_gbdi"] * df["gbdi_total_score"].to_numpy(float)
            + QLES["b_docs"] * df["docs_total"].to_numpy(float))
    # The reported R-squared is from the patient-only regression, so size the
    # residual against the fitted variance in that group, not the pooled one.
    ocd = (df["group"] == "OCD").to_numpy()
    resid_sd = yhat[ocd].std(ddof=1) * np.sqrt((1 - QLES["r2"]) / QLES["r2"])
    noise = rng.normal(0, 1, len(df))
    # residualise the noise against the fitted values so the realised R-squared
    # lands on target rather than a sample away from it
    n_ocd = noise[ocd] - noise[ocd].mean()
    f = yhat[ocd] - yhat[ocd].mean()
    n_ocd = n_ocd - f * (n_ocd @ f) / (f @ f)
    noise[ocd] = n_ocd / n_ocd.std(ddof=1)
    return np.clip(np.round(yhat + resid_sd * noise), QLES["lo"], QLES["hi"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frames = []
    for group in ("HC", "OCD"):
        g = draw_group(group, rng)
        g.insert(0, "group", group)
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)

    df["age"] = np.nan
    df["sex"] = ""
    for group in ("HC", "OCD"):
        idx = df.index[df["group"] == group]
        a = AGE[group]
        age = np.clip(np.round(rng.normal(a["mean"], a["sd"], len(idx))), a["lo"], a["hi"])
        # Reinstate the median-imputation spike the analytic file carries, topping
        # up whatever the draw already put on the median rather than adding to it.
        short = a["n_at_median"] - int((age == AGE_MEDIAN).sum())
        if short > 0:
            pool = np.flatnonzero(age != AGE_MEDIAN)
            age[rng.choice(pool, short, replace=False)] = AGE_MEDIAN
        df.loc[idx, "age"] = age
        # exact counts rather than a multinomial draw, which at n = 62 wanders
        p = SEX_P[group]
        counts = {k: int(round(v * len(idx))) for k, v in p.items()}
        counts[max(counts, key=counts.get)] += len(idx) - sum(counts.values())
        sex = np.repeat(list(counts), list(counts.values()))
        rng.shuffle(sex)
        df.loc[idx, "sex"] = sex

    df["qles_total_score"] = add_outcome(df, rng)

    # a handful of patients without the outcome, as in the real file
    ocd_idx = df.index[df["group"] == "OCD"]
    df.loc[rng.choice(ocd_idx, N_MISSING_OUTCOME, replace=False), "qles_total_score"] = np.nan

    df.insert(0, "record_id", [f"SYN{i:04d}" for i in range(1, len(df) + 1)])
    df = df[["record_id", "group", "age", "sex"] + FEATURES + ["qles_total_score"]]

    path = args.out / "synthetic_clinical_data.csv"
    df.to_csv(path, index=False)

    consts = dict(seed=args.seed, n=N, targets=TARGETS, scale=SCALE, qles=QLES,
                  age=AGE, age_median=AGE_MEDIAN, sex_p=SEX_P,
                  n_missing_outcome=N_MISSING_OUTCOME,
                  n_sex_missing_in_real_data=N_SEX_MISSING,
                  latent_R=LATENT_R.tolist(), features=FEATURES)
    (args.out / "synthetic_data_constants.json").write_text(json.dumps(consts, indent=2))

    # ---- verification ----------------------------------------------------
    print(f"wrote {path}  ({len(df)} rows, {df.shape[1]} columns)\n")
    print(f"{'group':5} {'feature':31} {'mean':>16} {'sd':>14} {'% at min':>16}")
    for group in ("HC", "OCD"):
        sub = df[df["group"] == group]
        for f in FEATURES:
            t_mean, t_sd, t_pct = TARGETS[group][f]
            lo = SCALE[f][0]
            print(f"{group:5} {f:31} {sub[f].mean():7.2f} (t {t_mean:6.2f}) "
                  f"{sub[f].std(ddof=1):6.2f} (t {t_sd:5.2f}) "
                  f"{100 * (sub[f] == lo).mean():6.1f} (t {t_pct:5.1f})")
    ok = all(df[f].between(*SCALE[f][:2]).all() for f in FEATURES)
    print(f"\nall features within scale bounds: {ok}")
    print(f"outcome missing: {int(df['qles_total_score'].isna().sum())} "
          f"(complete-case n = {int(df['qles_total_score'].notna().sum())})")


if __name__ == "__main__":
    main()
