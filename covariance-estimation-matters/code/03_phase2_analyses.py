# -*- coding: utf-8 -*-
"""Study 2 analyses on the real sample.

Estimates the reference covariance four ways, scores squared Mahalanobis
deviation, and runs the hierarchical regression and everything reported around
it: feature descriptives by group, MCD support composition, chi-square
calibration of the control distances, construct-overlap diagnostics, the partial
correlation with its interval and equivalence test, estimator sensitivity, and
the directional profile of the top-deviation decile.

All distances are computed about a common center, the healthy-control mean, so
differences between estimators reflect scatter rather than location; MCD's
own-center distances are kept alongside for the decomposition.

Pin scikit-learn to 1.8.0. MinCovDet differs across releases: at a fixed seed
version 1.4.2 returns a condition number of 155.71 and a reweighted support of
40, against 172.23 and 38 under 1.8.0. Nothing outside MCD is affected.

Reads   data/imputed_clinical_data.csv
Writes  Table4a/4b (regression), Table5 (deviation profile), fifteen TableS_*
        supplementary tables, figures, versions_phase2.json

Usage   python code/03_phase2_analyses.py [--base PATH]
"""

import argparse
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import linalg, stats
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# parents[1] when run as a file; the working directory when pasted into an
# interactive session, where __file__ does not exist
DEFAULT_BASE = (Path(__file__).resolve().parents[1]
                if "__file__" in globals() else Path.cwd())
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=DEFAULT_BASE)
ap.add_argument("--data", type=Path, default=None,
                help="input file; defaults to data/imputed_clinical_data.csv")
args = ap.parse_args()
OUT = Path(args.base) / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
IN_PATH = args.data or Path(args.base) / "data" / "imputed_clinical_data.csv"

EQUIV_BOUND = 0.10
MCD_SEED = 42            # set to the seed used for the originally reported fit
TOP_Q = 0.90
# 'modal'     -> missing sex assigned to the modal category (current manuscript)
# 'own_level' -> missing sex entered as its own category
SEX_MISSING_POLICY = "modal"

# Expected sex counts across the whole analysis file, after missing values are
# resolved. Derived from Table 1: recorded Female 198, Male 94, Other 6, plus the
# 27 unrecorded responses assigned to the modal category (Female). REDCap codes
# sex 1 = Female, 2 = Male, 3 = Non-binary/Other, 4 = Prefer not to say.
#
# This guard exists because Table 1 and the analysis file are built by separate
# scripts from separate codebooks, so a mapping error in either one is invisible
# until the counts are compared. Set to None to skip the check.
EXPECTED_SEX = {'Female': 225, 'Male': 94, 'Other': 6}

mcd_features = [
    'gbdi_total_score', 'shaps_total_score_snaith', 'psc_total_score',
    'docs_total', 'octcdq_incompleteness_score', 'octcdq_harm_score',
    'bfas_domain_neuroticism', 'bfas_domain_extraversion',
    'bfas_domain_conscientiousness', 'bfas_domain_agreeableness',
    'bfas_domain_openness_intellect'
]
# Theoretical scale minima (check against scoring manual / manuscript ranges)
SCALE_MIN = {
    'gbdi_total_score': 0, 'shaps_total_score_snaith': 0, 'psc_total_score': 23,
    'docs_total': 0, 'octcdq_incompleteness_score': 10, 'octcdq_harm_score': 10,
    'bfas_domain_neuroticism': 1, 'bfas_domain_extraversion': 1,
    'bfas_domain_conscientiousness': 1, 'bfas_domain_agreeableness': 1,
    'bfas_domain_openness_intellect': 1,
}
DEV_COL = 'multidimensional_deviance'


def write_csv(frame, name, **kw):
    path = OUT / name
    try:
        frame.to_csv(path, **kw)
    except OSError:
        path = OUT / f"{path.stem}_{pd.Timestamp.now():%Y%m%d_%H%M%S}{path.suffix}"
        frame.to_csv(path, **kw)
    print(f"  saved {path.name}")


# ------------------------------------------------------------------
# 0. Versions
# ------------------------------------------------------------------
versions = {
    "python": platform.python_version(), "numpy": np.__version__,
    "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
    "statsmodels": statsmodels.__version__, "pandas": pd.__version__,
    "mcd_seed": MCD_SEED, "sex_missing_policy": SEX_MISSING_POLICY,
}
(OUT / "versions_phase2.json").write_text(json.dumps(versions, indent=2))
print("Versions:", versions)

# ------------------------------------------------------------------
# 1. Load data; explicit sex handling
# ------------------------------------------------------------------
df = pd.read_csv(IN_PATH)
n_sex_missing = int(df['sex'].isna().sum())
df['sex_raw'] = df['sex']
if n_sex_missing:
    if SEX_MISSING_POLICY == "modal":
        modal = df['sex'].mode().iloc[0]
        df['sex'] = df['sex'].fillna(modal)
        print(f"Sex missing for {n_sex_missing}; assigned modal category '{modal}'.")
    else:
        df['sex'] = df['sex'].fillna('Unrecorded')
        print(f"Sex missing for {n_sex_missing}; entered as 'Unrecorded'.")
else:
    print("Sex has no missing values in the input file. If modal assignment happened "
          "upstream, the own-level sensitivity analysis cannot be run from this file.")

is_hc = df['group'].eq('HC').to_numpy()
is_ocd = df['group'].eq('OCD').to_numpy()
assert df.loc[is_hc, mcd_features].notna().all().all(), "HC features contain missing values"


def check_sex_coding(frame, expected):
    """Compare the analysis file's sex distribution against Table 1.

    A mis-mapped sex variable changes no model fit, because relabelling two
    levels of a factor leaves the design space untouched, so it never shows up
    as an error. It shows up only as coefficients attached to the wrong group
    and as demographics that contradict Table 1.
    """
    by_group = pd.crosstab(frame['sex'], frame['group'], margins=True, margins_name='Total')
    print("\nSex by group (analysis file):")
    print(by_group.to_string())
    if expected is None:
        print("  EXPECTED_SEX is None; sex coding not checked.")
        return
    counts = frame['sex'].value_counts()
    levels = sorted(set(expected) | set(counts.index))
    got = {k: int(counts.get(k, 0)) for k in levels}
    want = {k: int(expected.get(k, 0)) for k in levels}
    if got != want:
        raise SystemExit(
            "\nSex coding disagrees with Table 1.\n"
            f"  expected {want}\n  found    {got}\n"
            "REDCap codes sex 1 = Female, 2 = Male, 3 = Non-binary/Other, "
            "4 = Prefer not to say.\n"
            "Fix the mapping in the script that writes imputed_clinical_data.csv, "
            "not here, then regenerate the file. Adjust EXPECTED_SEX only if "
            "Table 1 itself has changed."
        )
    print(f"  Sex coding matches Table 1: {want}.")


check_sex_coding(df, EXPECTED_SEX)

# ------------------------------------------------------------------
# 2. Feature descriptives by group (floor-effect evidence)
# ------------------------------------------------------------------
print("\n--- Feature descriptives by group ---")
rows = []
for g in ['HC', 'OCD']:
    sub = df.loc[df['group'] == g, mcd_features]
    for f in mcd_features:
        x = sub[f].dropna()
        rows.append(dict(
            group=g, feature=f, n=len(x), mean=x.mean(), sd=x.std(ddof=1),
            median=x.median(), iqr=x.quantile(.75) - x.quantile(.25),
            observed_min=x.min(), scale_min=SCALE_MIN[f],
            pct_at_scale_min=100 * np.isclose(x, SCALE_MIN[f]).mean(),
            n_distinct=x.nunique()))
desc = pd.DataFrame(rows)
write_csv(desc.round(3), "TableS_feature_descriptives_by_group.csv", index=False)
print(desc.loc[desc.group == 'HC', ['feature', 'pct_at_scale_min', 'n_distinct']]
      .round(1).to_string(index=False))

# ------------------------------------------------------------------
# 3. Reference covariance estimators (common center)
# ------------------------------------------------------------------
print("\n--- Reference covariance estimators ---")
hc_raw = df.loc[is_hc, mcd_features]
scaler = StandardScaler().fit(hc_raw)
Z_all = scaler.transform(df[mcd_features])
Z_hc = Z_all[is_hc]
hc_center = Z_hc.mean(axis=0)          # ~0 by construction; computed explicitly


def d2(Z, center, cov):
    """Squared Mahalanobis distance about `center`, using the same pseudo-inverse
    scikit-learn uses internally. Returns distances and the numerical rank."""
    P, rank = linalg.pinvh(cov, return_rank=True)
    D = Z - center
    return np.einsum('ij,jk,ik->i', D, P, D), int(rank)


def cond(cov):
    ev = np.linalg.eigvalsh(cov)
    return np.inf if ev.min() <= ev.max() * cov.shape[0] * np.finfo(float).eps else ev.max() / ev.min()


fits = {}
with warnings.catch_warnings(record=True) as w_mcd:
    warnings.simplefilter("always")
    mcd = MinCovDet(random_state=MCD_SEED).fit(Z_hc)
    mcd1 = MinCovDet(support_fraction=1.0, random_state=MCD_SEED).fit(Z_hc)
mcd_warnings = pd.Series([str(w.message).split(':')[0] for w in w_mcd]).value_counts()
fits['sample'] = EmpiricalCovariance().fit(Z_hc)
fits['ledoit_wolf'] = LedoitWolf().fit(Z_hc)
fits['mcd_default'] = mcd
fits['mcd_sf1'] = mcd1

est_rows, dists = [], {}
dist_records = []                       # (estimator, center, distances, rank) for calibration
for name, est in fits.items():
    dist_c, rank = d2(Z_all, hc_center, est.covariance_)
    dists[name] = dist_c
    dist_records.append((name, 'HC mean', dist_c, rank))
    row = dict(estimator=name, condition_number=cond(est.covariance_), rank=rank,
               n_retained=int(est.support_.sum()) if hasattr(est, 'support_') else int(is_hc.sum()),
               center='HC mean',
               hc_median_d2=np.median(dist_c[is_hc]), ocd_median_d2=np.median(dist_c[is_ocd]))
    row['ocd_hc_ratio'] = row['ocd_median_d2'] / row['hc_median_d2']
    if name == 'ledoit_wolf':
        row['shrinkage'] = est.shrinkage_
    est_rows.append(row)
    if name.startswith('mcd'):
        # Own-center (robust location) distances, as est.mahalanobis() returns
        dist_own = est.mahalanobis(Z_all)
        dist_records.append((name, 'MCD location', dist_own, rank))
        r2 = dict(row, center='MCD location',
                  hc_median_d2=np.median(dist_own[is_hc]),
                  ocd_median_d2=np.median(dist_own[is_ocd]))
        r2['ocd_hc_ratio'] = r2['ocd_median_d2'] / r2['hc_median_d2']
        est_rows.append(r2)

# Sanity check: LW about the HC mean equals sklearn's own LW distances
assert np.allclose(dists['ledoit_wolf'], fits['ledoit_wolf'].mahalanobis(Z_all))
est_tbl = pd.DataFrame(est_rows)
write_csv(est_tbl.round(4), "TableS_estimator_comparison_realdata.csv", index=False)
print(est_tbl.round(2).to_string(index=False))
if len(mcd_warnings):
    print("MCD warnings (count):\n" + mcd_warnings.to_string())
    write_csv(mcd_warnings.rename('count').to_frame(), "TableS_mcd_warnings.csv")

# ------------------------------------------------------------------
# 3a. Chi-square calibration of the reference distribution
#
# If the healthy-control features were multivariate normal and the covariance
# were estimated without error, squared Mahalanobis distances in that group
# would follow chi-square on `rank` degrees of freedom. The comparison is a
# reference point, not a test: the covariance is estimated in the same sample
# the distances are computed in, so even a well-behaved estimator sits slightly
# below the reference. What the table shows is the size and direction of the
# departure, which is what the Method promises and the Results have never
# delivered.
#
# Degrees of freedom come from the numerical rank of the precision matrix, not
# from the feature count, so a rank-deficient fit is compared against the right
# reference rather than an optimistic one.
# ------------------------------------------------------------------
print("\n--- Chi-square calibration of healthy-control distances ---")
cal_rows = []
for name, centre, dist, rank in dist_records:
    hc_d = dist[is_hc]
    ref_med = stats.chi2.ppf(0.50, rank)
    q95, q975 = stats.chi2.ppf(0.95, rank), stats.chi2.ppf(0.975, rank)
    cal_rows.append(dict(
        estimator=name, center=centre, df=rank,
        chi2_median=ref_med, chi2_q95=q95, chi2_q975=q975,
        hc_median_d2=np.median(hc_d),
        median_ratio=np.median(hc_d) / ref_med,
        pct_hc_above_q95=100 * np.mean(hc_d > q95),      # nominal 5%
        pct_hc_above_q975=100 * np.mean(hc_d > q975),    # nominal 2.5%
        ks_stat=float(stats.kstest(hc_d, 'chi2', args=(rank,)).statistic),
    ))
cal = pd.DataFrame(cal_rows)
write_csv(cal.round(4), "TableS_chi2_calibration.csv", index=False)
print(cal.round(3).to_string(index=False))
print("  median_ratio: 1.00 = the reference median. Below 1 is compression, above 1")
print("  is inflation. pct_hc_above_q95 / q975 are nominally 5% and 2.5%.")
print("  ks_stat is descriptive; no p-value is reported, because the covariance is")
print("  estimated in the same sample and the null is therefore not exact.")

# ------------------------------------------------------------------
# 3b. Feature correlation matrix
#
# The conditioning argument rests on the condition number, which tells a reader
# that the matrix is ill-conditioned but not why. The correlations show where it
# comes from, and let a reader distinguish real redundancy between features from
# unreliable measurement.
# ------------------------------------------------------------------
print("\n--- Feature correlations ---")
corr_hc = pd.DataFrame(Z_hc, columns=mcd_features).corr()
corr_all = df.loc[:, mcd_features].corr()
write_csv(corr_hc.round(3), "TableS_feature_correlations_hc.csv")
write_csv(corr_all.round(3), "TableS_feature_correlations_full.csv")

pairs = pd.DataFrame([dict(feature_a=a, feature_b=b, r=corr_hc.loc[a, b])
                      for i, a in enumerate(mcd_features)
                      for b in mcd_features[i + 1:]])
n_undefined = int(pairs.r.isna().sum())
pairs = (pairs.dropna()
         .sort_values('r', key=lambda s: s.abs(), ascending=False)
         .reset_index(drop=True))
assert len(pairs) + n_undefined == len(mcd_features) * (len(mcd_features) - 1) // 2
write_csv(pairs.round(3), "TableS_feature_correlation_pairs_hc.csv", index=False)
if n_undefined:
    print(f"  {n_undefined} pair(s) undefined and dropped: a feature has zero variance "
          "in the healthy-control group, which also makes the reference covariance "
          "singular. Do not report the remaining correlations without saying so.")
print(f"healthy controls, n = {int(is_hc.sum())}: "
      f"|r| ranges {pairs.r.abs().min():.3f} to {pairs.r.abs().max():.3f}, "
      f"median |r| = {pairs.r.abs().median():.3f}")
print("Five largest by magnitude:")
print(pairs.head(5).to_string(index=False))

# MCD location shift (scaled units; negative on symptom features = center pulled to floor)
loc_shift = pd.DataFrame({'feature': mcd_features,
                          'mcd_default_minus_hc_mean': mcd.location_ - hc_center,
                          'mcd_sf1_minus_hc_mean': mcd1.location_ - hc_center})
write_csv(loc_shift.round(4), "TableS_mcd_location_shift.csv", index=False)

# MCD support composition
#
# Reports BOTH the mean and the percentage at the scale minimum for every
# feature. The percentages show how much of each set sits on a floor; the means
# put the retained and excluded sets on the measures' own scales, which is what
# the restriction-of-range claim in the Results needs. Before v4.3 only the
# percentages were written, so the excluded-versus-retained subscale means
# quoted in the draft could not be reproduced from any current script.
hc_df = df.loc[is_hc].reset_index(drop=True)
sets = {'raw_support': mcd.raw_support_, 'reweighted_support': mcd.support_,
        'excluded': ~mcd.support_, 'all_hc': np.ones(len(hc_df), bool)}
comp = []
for label, mask in sets.items():
    sub = hc_df.loc[mask]
    r = dict(set=label, n=int(mask.sum()), mean_docs=sub['docs_total'].mean())
    for f in mcd_features:
        r[f'mean_{f}'] = sub[f].mean() if len(sub) else np.nan
        # Each set's mean in units of the FULL control sample's SD, the same
        # metric the directional profiling uses, so eleven measures on eight
        # different scales can be compared with each other.
        r[f'z_{f}'] = ((sub[f].mean() - hc_df[f].mean()) / hc_df[f].std(ddof=1)
                       if len(sub) else np.nan)
    for f in mcd_features:
        r[f'pct_min_{f}'] = 100 * np.isclose(sub[f], SCALE_MIN[f]).mean() if len(sub) else np.nan
    comp.append(r)
comp = pd.DataFrame(comp)
write_csv(comp.round(2), "TableS_mcd_support_composition.csv", index=False)
print(comp[['set', 'n', 'mean_docs', 'pct_min_shaps_total_score_snaith',
            'pct_min_docs_total']].round(2).to_string(index=False))
print("\nFeature means by support set (the restriction-of-range evidence):")
print(comp[['set', 'n'] + [f'mean_{f}' for f in mcd_features]]
      .rename(columns=lambda c: c.replace('mean_', '').replace('bfas_domain_', 'bfas_')
              .replace('_total_score', '').replace('_snaith', '').replace('octcdq_', 'oct_'))
      .round(2).to_string(index=False))

# Primary deviation score: Ledoit-Wolf, HC-mean center (unchanged from v3)
df[DEV_COL] = dists['ledoit_wolf']
write_csv(df, "final_analyzed_dataset.csv", index=False)

# ------------------------------------------------------------------
# 4. Hierarchical regression, VIF, construct overlap
# ------------------------------------------------------------------
print("\n--- Hierarchical regression ---")
ocd_reg = df.loc[is_ocd].dropna(subset=['age', 'sex', 'qles_total_score', 'docs_total',
                                        'gbdi_total_score', DEV_COL]).copy()
for name in fits:
    ocd_reg[f'd2_{name}'] = dists[name][ocd_reg.index]
print(f"Regression cohort: N = {len(ocd_reg)}")

COVS = 'age + C(sex) + docs_total + gbdi_total_score'
m1 = smf.ols('qles_total_score ~ age + C(sex)', data=ocd_reg).fit()
m2 = smf.ols(f'qles_total_score ~ {COVS}', data=ocd_reg).fit()
m3 = smf.ols(f'qles_total_score ~ {COVS} + {DEV_COL}', data=ocd_reg).fit()


def step_row(label, model, prev=None):
    row = {'step': label, 'R2': model.rsquared, 'adj_R2': model.rsquared_adj,
           'df_model': int(model.df_model), 'n': int(model.nobs)}
    if prev is not None:
        a = sm.stats.anova_lm(prev, model)
        row.update({'dR2': model.rsquared - prev.rsquared, 'F_change': a['F'].iloc[1],
                    'df_num': int(a['df_diff'].iloc[1]), 'df_den': int(model.df_resid),
                    'p_change': a['Pr(>F)'].iloc[1]})
    return row


coef_tbl = pd.DataFrame({'coef': m3.params, 'std_err': m3.bse, 't': m3.tvalues,
                         'p': m3.pvalues, 'ci_low': m3.conf_int()[0],
                         'ci_high': m3.conf_int()[1]})
step_tbl = pd.DataFrame([step_row('1_demographics', m1),
                         step_row('2_severity', m2, m1),
                         step_row('3_deviation', m3, m2)])
write_csv(coef_tbl.round(4), "Table4a_coefficients.csv")
write_csv(step_tbl.round(4), "Table4b_step_statistics.csv", index=False)
with pd.option_context('display.float_format', lambda v: f'{v:.4f}'):
    print(step_tbl.to_string(index=False))

X3 = pd.DataFrame(m3.model.exog, columns=m3.model.exog_names)
vif = pd.Series([variance_inflation_factor(X3.values, i) for i in range(X3.shape[1])],
                index=X3.columns, name='VIF').drop('Intercept')
overlap_r2 = smf.ols(f'{DEV_COL} ~ docs_total + gbdi_total_score', data=ocd_reg).fit().rsquared
zero_r = stats.pearsonr(ocd_reg[DEV_COL], ocd_reg['qles_total_score'])
overlap = pd.concat([vif, pd.Series({'overlap_R2_dev_on_DOCS_gBDI': overlap_r2,
                                     'zero_order_r_dev_QoL': zero_r.statistic,
                                     'zero_order_p': zero_r.pvalue})])
write_csv(overlap.round(4).rename('value').to_frame(), "TableS_construct_overlap.csv")
print(overlap.round(3).to_string())


# ------------------------------------------------------------------
# 5. Partial correlation: interval first, TOST second
# ------------------------------------------------------------------
def partial_stats(data, dev, covs=COVS, bound=EQUIV_BOUND):
    base = smf.ols(f'qles_total_score ~ {covs}', data=data).fit()
    full = smf.ols(f'qles_total_score ~ {covs} + {dev}', data=data).fit()
    rdev = smf.ols(f'{dev} ~ {covs}', data=data).fit()
    r = np.corrcoef(rdev.resid, base.resid)[0, 1]
    k, n = int(rdev.df_model), len(data)
    se = 1 / np.sqrt(n - k - 3)
    fz, zb = np.arctanh(r), np.arctanh(bound)
    ci90 = np.tanh([fz - 1.645 * se, fz + 1.645 * se])
    ci95 = np.tanh([fz - 1.960 * se, fz + 1.960 * se])
    a = sm.stats.anova_lm(base, full)
    one_minus = 1 - base.rsquared
    return dict(
        n=n, k_covariate_df=k, se_fisher=se, partial_r=r,
        ci90_low=ci90[0], ci90_high=ci90[1], ci95_low=ci95[0], ci95_high=ci95[1],
        halfwidth90_at_r0=np.tanh(1.645 * se),
        # Delta-R^2 implied by a partial r: pr^2 * (1 - R^2_reduced)
        dR2_observed=full.rsquared - base.rsquared,
        dR2_at_ci90_high=np.sign(ci90[1]) * ci90[1] ** 2 * one_minus,
        dR2_at_ci90_low=np.sign(ci90[0]) * ci90[0] ** 2 * one_minus,
        F=a['F'].iloc[1], p_F=a['Pr(>F)'].iloc[1],
        tost_z_lower=(fz + zb) / se, tost_p_lower=1 - stats.norm.cdf((fz + zb) / se),
        tost_z_upper=(fz - zb) / se, tost_p_upper=stats.norm.cdf((fz - zb) / se),
        smallest_symmetric_bound_rejected=max(abs(ci90[0]), abs(ci90[1])),
    )


print("\n--- Partial correlation: interval and TOST ---")
ps = partial_stats(ocd_reg, DEV_COL)
ps['equivalence_established'] = max(ps['tost_p_lower'], ps['tost_p_upper']) < .05
write_csv(pd.Series(ps).to_frame('value'), "TableS_partial_r_interval_tost.csv")
print(f"partial r = {ps['partial_r']:.3f}, 90% CI [{ps['ci90_low']:.3f}, {ps['ci90_high']:.3f}]; "
      f"half-width at r=0 = {ps['halfwidth90_at_r0']:.3f}; "
      f"upper limit implies dR2 = {ps['dR2_at_ci90_high']:.4f}")
print(f"TOST z = {ps['tost_z_lower']:.2f} / {ps['tost_z_upper']:.2f}, "
      f"p = {ps['tost_p_lower']:.3f} / {ps['tost_p_upper']:.3f}")

# ------------------------------------------------------------------
# 6. Estimator sensitivity of the substantive result
# ------------------------------------------------------------------
print("\n--- Estimator sensitivity ---")
ocd_reg['log_d2_ledoit_wolf'] = np.log(ocd_reg['d2_ledoit_wolf'])
ocd_all_idx = np.where(is_ocd)[0]
ref = dists['ledoit_wolf'][ocd_all_idx]
ref_top = set(ocd_all_idx[ref >= np.quantile(ref, TOP_Q)])
sens = []
for label, col, src in ([(n, f'd2_{n}', n) for n in fits] +
                        [('ledoit_wolf_logD2', 'log_d2_ledoit_wolf', 'ledoit_wolf')]):
    s = partial_stats(ocd_reg, col)
    v = dists[src][ocd_all_idx]
    top = set(ocd_all_idx[v >= np.quantile(v, TOP_Q)])
    sens.append(dict(analysis=label, dR2=s['dR2_observed'], F=s['F'], p=s['p_F'],
                     partial_r=s['partial_r'], ci90_low=s['ci90_low'], ci90_high=s['ci90_high'],
                     spearman_vs_LW_allOCD=stats.spearmanr(v, ref).statistic,
                     top_decile_jaccard_vs_LW=len(top & ref_top) / len(top | ref_top)))
if n_sex_missing and SEX_MISSING_POLICY == 'modal':
    alt = ocd_reg.copy()
    alt['sex'] = df.loc[alt.index, 'sex_raw'].fillna('Unrecorded')
    s = partial_stats(alt, DEV_COL)
    sens.append(dict(analysis='ledoit_wolf_sex_unrecorded_own_level', dR2=s['dR2_observed'],
                     F=s['F'], p=s['p_F'], partial_r=s['partial_r'],
                     ci90_low=s['ci90_low'], ci90_high=s['ci90_high']))
sens = pd.DataFrame(sens)
write_csv(sens.round(4), "TableS_estimator_sensitivity.csv", index=False)
print(sens.round(3).to_string(index=False))

# ------------------------------------------------------------------
# 6b. Step-3 residual, influence, and robustness diagnostics
# ------------------------------------------------------------------
print("\n--- Step-3 diagnostics and robustness ---")
infl = m3.get_influence()
cooks = infl.cooks_distance[0]
lev = infl.hat_matrix_diag
cook_cut = 4 / len(ocd_reg)

# HC3 standard errors for the same model
m3_hc3 = smf.ols(f'qles_total_score ~ {COVS} + {DEV_COL}', data=ocd_reg).fit(cov_type='HC3')

# Incremental test refitted without the cases Cook's D flags
keep = ocd_reg.loc[cooks <= cook_cut]
k2 = smf.ols(f'qles_total_score ~ {COVS}', data=keep).fit()
k3 = smf.ols(f'qles_total_score ~ {COVS} + {DEV_COL}', data=keep).fit()
a_keep = sm.stats.anova_lm(k2, k3)

diag = pd.DataFrame({
    'max_cooks_d': [cooks.max()],
    'cooks_cutoff_4_over_n': [cook_cut],
    'n_cooks_gt_cutoff': [int((cooks > cook_cut).sum())],
    'max_leverage': [lev.max()],
    'mean_leverage': [lev.mean()],
    'shapiro_resid_p': [stats.shapiro(m3.resid).pvalue],
    'hc3_dev_coef': [m3_hc3.params[DEV_COL]],
    'hc3_dev_se': [m3_hc3.bse[DEV_COL]],
    'hc3_dev_t': [m3_hc3.tvalues[DEV_COL]],
    'hc3_dev_p': [m3_hc3.pvalues[DEV_COL]],
    'trimmed_n': [len(keep)],
    'trimmed_dR2': [k3.rsquared - k2.rsquared],
    'trimmed_F': [a_keep['F'].iloc[1]],
    'trimmed_p': [a_keep['Pr(>F)'].iloc[1]],
})
write_csv(diag.round(4), "TableS_step3_influence.csv", index=False)
print(diag.round(4).T.rename(columns={0: 'value'}).to_string())

# Highest-leverage cases. Small sex categories carry high leverage on their own,
# so the sex column is shown alongside the deviation score.
lev_tbl = (ocd_reg.assign(leverage=lev, cooks_d=cooks)
           .nlargest(5, 'leverage')[['sex', DEV_COL, 'qles_total_score', 'leverage', 'cooks_d']])
write_csv(lev_tbl.round(4), "TableS_step3_top_leverage.csv")
print("\nHighest-leverage cases:")
print(lev_tbl.round(3).to_string())
print(f"\nSex category sizes in the regression cohort:\n"
      f"{ocd_reg['sex'].value_counts().to_string()}")

# ------------------------------------------------------------------
# 7. Directional profiling (all OCD)
# ------------------------------------------------------------------
print("\n--- Directional profiling ---")
ocd_all = df.loc[is_ocd].copy()
Z_ocd = Z_all[is_ocd]
hc_sd = Z_hc.std(axis=0, ddof=0)        # = 1 in scaled units (StandardScaler, ddof=0)
Zp = (Z_ocd - hc_center) / hc_sd
# Previous version divided by sqrt(diag(LW covariance)); identical here.
lw_diag = np.sqrt(np.diag(fits['ledoit_wolf'].covariance_))
assert np.allclose(Zp, (Z_ocd - fits['ledoit_wolf'].location_) / lw_diag)
for i, col in enumerate(mcd_features):
    ocd_all[f'{col}_z'] = Zp[:, i]

cutoff = ocd_all[DEV_COL].quantile(TOP_Q)
severe = ocd_all[ocd_all[DEV_COL] >= cutoff]
zcols = [f'{c}_z' for c in mcd_features]
prof = pd.DataFrame({'mean_z_top_decile': severe[zcols].mean(),
                     'sd_z_top_decile': severe[zcols].std(ddof=1),
                     'mean_z_all_ocd': ocd_all[zcols].mean()})
prof.index = [c[:-2].replace('_total_score', '').replace('_domain', '') for c in prof.index]
prof = prof.sort_values('mean_z_top_decile', ascending=False)
write_csv(prof.round(3), "Table5_deviation_profile.csv")
print(f"90th-percentile cutoff D^2 = {cutoff:.2f}; top-decile n = {len(severe)}")
print(prof.round(2).to_string())

# ==================================================================
# FIGURES (unchanged design; profile figure now shows SD bars)
# ==================================================================
C_HC, C_OCD = "#0072B2", "#D55E00"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight"})


def save(fig, stem):
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        target = OUT / f"{stem}{ext}"
        try:
            fig.savefig(target, **kw)
        except OSError as exc:
            alt = OUT / f"{stem}_{pd.Timestamp.now():%Y%m%d_%H%M%S}{ext}"
            fig.savefig(alt, **kw)
            print(f"  ! could not write {target.name} ({exc}); saved as {alt.name}")
    plt.close(fig)


hc_vals = df.loc[is_hc, DEV_COL].values
ocd_vals = df.loc[is_ocd, DEV_COL].values
log_data = [np.log10(hc_vals), np.log10(ocd_vals)]
ns = np.array([len(hc_vals), len(ocd_vals)], float)
widths = 0.80 * np.sqrt(ns / ns.max())
fig, ax = plt.subplots(figsize=(6.6, 4.8))
parts = ax.violinplot(log_data, widths=widths, showmedians=True, showextrema=False)
for pc, c in zip(parts['bodies'], [C_HC, C_OCD]):
    pc.set_facecolor(c); pc.set_alpha(0.55); pc.set_edgecolor(c); pc.set_linewidth(0.8)
parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.6)
lo = min(v.min() for v in log_data) - 0.12
hi = max(v.max() for v in log_data) + 0.18
decades = np.arange(np.floor(lo), np.ceil(hi) + 1)
minor = np.log10(np.concatenate([[k * 10.0 ** d for k in range(2, 10)] for d in decades]))
ax.yaxis.set_major_locator(FixedLocator(decades[(decades >= lo) & (decades <= hi)]))
ax.yaxis.set_minor_locator(FixedLocator(minor[(minor >= lo) & (minor <= hi)]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{10 ** v:,.0f}"))
ax.set_ylim(lo, hi)
ax.hlines(np.log10(cutoff), 1.58, 2.46, color='gray', ls='--', lw=1.1, zorder=3)
ax.annotate(f"90th pct of OCD\n($D^2$ = {cutoff:,.0f})", xy=(2.46, np.log10(cutoff)),
            xytext=(-4, 7), textcoords='offset points', ha='right', va='bottom',
            fontsize=7.5, color='gray')
ax.set_xticks([1, 2])
ax.set_xticklabels([f"HC\n(n = {len(hc_vals)})", f"OCD\n(n = {len(ocd_vals)})"])
ax.set_xlim(0.45, 2.60)
ax.set_ylabel('Multidimensional deviation\n(squared Mahalanobis distance, log scale)')
ax.legend(handles=[Line2D([0], [0], color='black', lw=1.6, label='group median')],
          fontsize=7.5, loc='upper left')
save(fig, "Fig_deviation_distributions")

DISPLAY = {
    'docs_total': 'OC symptom severity (DOCS)', 'gbdi': 'Depression, general factor (gBDI)',
    'shaps_snaith': 'Anhedonia (SHAPS)', 'psc': 'Somatic symptom burden (RSCL)',
    'octcdq_harm_score': 'Harm avoidance (OCTCDQ)',
    'octcdq_incompleteness_score': 'Incompleteness (OCTCDQ)',
    'bfas_neuroticism': 'Neuroticism', 'bfas_extraversion': 'Extraversion',
    'bfas_conscientiousness': 'Conscientiousness', 'bfas_agreeableness': 'Agreeableness',
    'bfas_openness_intellect': 'Openness/Intellect',
}
p_ = prof.iloc[::-1]
fig, ax = plt.subplots(figsize=(7.4, 4.8))
vals = p_['mean_z_top_decile'].values
ax.barh([DISPLAY.get(c, c) for c in p_.index], vals, xerr=p_['sd_z_top_decile'].values,
        color=[C_OCD if v >= 0 else C_HC for v in vals], alpha=0.9,
        error_kw=dict(lw=0.8, capsize=2, ecolor='0.3'))
ax.scatter(p_['mean_z_all_ocd'].values, range(len(p_)), marker='D', s=14, color='k',
           zorder=4, label='all OCD (mean)')
ax.axvline(0, color='k', lw=0.8)
ax.set_xlabel('Mean z score relative to the HC mean and SD (bars: top decile, whiskers: ±1 SD)')
ax.set_title(f"Directional profile of the top-decile OCD subgroup (n = {len(severe)})", loc='left')
ax.legend(fontsize=7.5, loc='lower right')
ax.xaxis.grid(True, lw=0.4, alpha=0.4); ax.set_axisbelow(True)
save(fig, "FigS_zscore_profile")
print("\nDone.")
