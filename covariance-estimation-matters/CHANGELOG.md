# Changelog

Revisions to the analysis scripts, kept here rather than in the module
docstrings. Entries affecting reported values are marked.

## Archive releases

### v1.1.0 (2026-10-14)

- Added `docs/supplementary_materials.pdf`, the manuscript's Supplements S1–S8.
- Corrected wording that described `docs/` as holding preregistration amendments.
  Study 1's design document was not time-stamped or publicly registered and is not
  part of the archive, so no analysis is described as preregistered. README.md and
  MANIFEST.md now describe `docs/study1_amendment_2026-09-17.md` as an analysis log,
  and a dated note at the top of that file explains this; its original text is
  unchanged.
- Comments in 05_study1_simulation.py point to that file instead of the unarchived
  design document. No code, output, or reported value changed.
- README gives the concept DOI, which resolves to the latest version.

### v1.0.0 (2026-09-21)

- Initial archive.

## 01_clean_and_score.py

- **Affects reported values.** Q-LES-Q-SF is scored over items 1-14 (range
  14-70), the standard summary total. An earlier version summed items 1-16,
  giving a 16-80 range and inflating the primary outcome.
- `sex` is a categorical covariate and is no longer passed through the
  regression imputer; missing values take the mode. `age` is imputed by median
  rather than by BayesianRidge.
- Group membership is retained as a predictor during feature imputation, as the
  Method describes, so imputed values respect group.
- Scale totals are prorated against a completion threshold instead of a raw
  skipna sum, so a partial responder is not given a downward-biased total. For a
  complete scale this is identical to a plain sum.
- Imputed feature values are clipped to each scale's valid range.

## 03_phase2_analyses.py

- Version report written to disk, since scikit-learn's MCD correction has
  changed across releases and the version is part of the method.
- Feature descriptives by group for all eleven features, including percentage at
  the scale minimum and number of distinct values.
- Real-data estimator comparison moved here: sample covariance, Ledoit-Wolf, MCD
  at its default support fraction, and MCD at 1.0, all about a common center.
- MCD support composition records raw support, reweighted support and excluded
  sets, with percentage at minimum and mean per feature, and the numerical rank
  of every fit.
- VIFs across the full design, construct-overlap R², and the zero-order
  correlation are computed and saved.
- Partial-correlation reporting leads with the 90% CI, the half-width at r = 0,
  and the ΔR² implied by the limits; TOST is secondary.
- Estimator sensitivity repeats the incremental test, rank agreement and
  top-decile overlap under each estimator, plus log D².
- Profiling z scores are defined as (x − HC mean) / HC SD, with an assertion that
  this matches the earlier Ledoit-Wolf-diagonal version.

### v4.1

- Step-3 robustness checks folded in: HC3 standard errors, the incremental test
  refitted without cases exceeding Cook's D = 4/n, and the highest-leverage cases
  listed with sex category.
- Influence diagnostics print to the console as well as saving.
- Step R² prints to four decimals so table and console cannot disagree by
  rounding.
- Sex coding is checked against Table 1 before anything is computed. Relabelling
  two levels of a factor leaves every fit unchanged, so a mis-mapped sex variable
  is otherwise silent.

### v4.2 to v4.4

- **Affects reported values.** Chi-square calibration of the control distances,
  with degrees of freedom from the numerical rank of each fit rather than the
  feature count.
- Feature correlation matrices for controls and the full sample, with pairs
  ranked by magnitude.
- Support-set composition carries the mean of every feature alongside the
  percentage at minimum. The excluded-versus-retained subscale means quoted in an
  earlier draft could not be reproduced, because only percentages were written.
- Each set's mean is also written in SD units of the full control sample, which
  feeds 08_fig_mcd_discarded.py.

## 05_study1_simulation.py

### v1.2

- **Added post hoc**, after the Study 2 calibration table raised the question.
  Reference-group calibration against chi-square: the median distance as a ratio
  to the chi-square median, and the proportion above the .95 and .975 quantiles,
  per estimator, condition and n. Logged as an amendment; see
  docs/study1_amendment_2026-09-17.md.

### v1.3

- **Added post hoc.** Floor prevalence may be a vector of length P_SYM rather
  than a scalar, and a new condition `floor_matched` uses the prevalences
  observed in the Study 2 control sample. Matching a mean prevalence across a
  heterogeneous vector does not establish that two settings are equivalent, so
  the matched condition tests it directly.
- Existing conditions reproduce exactly: the new one is appended last so no
  condition id shifts, and scalar prevalence keeps its original seeding.

### v1.4

- **Added post hoc.** A condition may carry its own latent correlation matrix,
  and `floor_matched_corr` uses the observed floor vector together with a matrix
  derived from the control sample. The simulated symptom block correlates .50
  where two real features correlate .86, so this asks whether matching the
  structure changes the picture.
- The latent matrix is not the observed one. Observed correlations already carry
  the floors, so they were inverted through the censoring map (04_invert_r.py)
  and projected to the nearest positive-definite matrix.

### v1.6

- **Affects interpretation.** Oracle calibration: reference distances are also
  computed with the benchmark mean and covariance, centred as the estimators are.
  Chi-square is valid only under multivariate normality with population
  parameters, so under floor and skew conditions it was never the right yardstick
  and an estimator closer to 1.00 was not thereby more accurate.
- Exceedance is additionally reported with contaminated observations removed from
  the reference group, since the previous version charged an estimator for the
  contamination it was handed.
- `latent()` returns the contamination mask and consumes the random stream in the
  same order, so previously reported numbers reproduce exactly.

### Figure-only revisions (v1.1, v1.5)

- Fig_study1 became the principal figure in six panels, adding covariance
  estimation error and the two calibration panels; the ranking outcomes moved to
  FigS_study1_ranking. The sample covariance is drawn wider than Ledoit-Wolf
  because the two coincide under several conditions.
- Panel legends and axis limits fixed; save() reports a failed overwrite instead
  of silently writing a timestamped file.

## 06_bootstrap_sample_size.py

- Sample covariance added throughout, separating "use every observation" from
  "shrink".
- All distances about a common center, with MCD's own-center distances kept for
  the decomposition.
- "Bias" renamed "estimation error"; mean-matrix bias reported separately;
  relative Frobenius error added.
- Precision-matrix error added, because Mahalanobis distance depends on the
  inverse and covariance-scale error alone can favour the unshrunk estimator at
  small N. Distance-level outcomes added against benchmark distances.
- Numerical rank recorded for every fit; condition number is infinite when rank
  deficient, with no silent pseudo-inverse.
- Restriction diagnostics use un-jittered source values and add percentage at
  scale minimum in retained versus excluded controls.
- Cohen's d removed: it grows as the seed-to-seed SD shrinks and has no
  score-level meaning.
- Non-overlapping random streams per seed and role.
- Jitter sensitivity added at jitter in {0, .01, .05, .10}.

### v2.1 (figures only)

- Panels A and D plot the across-seed median with an interquartile band, since
  condition numbers and median D² are right-skewed on log axes where mean − SD
  can fall below zero. Tables still report means and SDs.
- The seed-convergence band is ±1 SE, and save() reports a failed overwrite.

## 07_support_fraction_sweep.py

### v2

- **Affects Table S3 (synthetic setting).** The benchmark was bootstrapped from
  the synthetic reference sample itself, so the untrimmed and Ledoit-Wolf fits
  matched it almost by construction. It is now always drawn from the real
  control sample, independently of the reference sample.
- Distances are about a common center, with own-center distances kept separately.
- MCD at its default support fraction and at 1.00 are reported as separate rows,
  and the untrimmed comparator is labelled the sample covariance, since it is not
  the support_fraction -> 1 limit of MCD.
- Numerical rank, precision-matrix error, MCD warning counts and floor
  composition added; restriction ratio is NaN rather than a huge number when the
  retained mean DOCS is zero.
- Seed lookup added: the default MCD fit refitted across 500 seeds on the real
  control sample.

### v1

- **Affects reported values.** Distances are computed directly from each
  estimator's covariance via `pinvh`. The previous version built an
  `EmpiricalCovariance`, fitted it, then overwrote `.covariance_`; scikit-learn
  caches `precision_` at fit time, so every deviance column had been computed
  under the plain empirical precision rather than the estimator being tested.
- **Affects reported values.** Location is each estimator's own `location_`
  rather than hard-coded zero, which was correct only for the real control sample
  and discarded MCD's robust location.
- Singularity is detected from the eigenvalue spectrum rather than a fixed
  threshold on the condition number, and raw versus reweighted support sizes are
  recorded so the pattern can be explained.
- The figure's conditioning reference is the benchmark covariance's own condition
  number rather than a rule of thumb near 30, which would have labelled the
  correctly specified matrix ill-conditioned.

## 01_clean_and_score.py, second pass

- **Affects reported values.** Sex codes 1 and 2 were mapped to Male and Female;
  the REDCap codebook has 1 = Female and 2 = Male. The analytic file used for the
  reported analyses has the correct coding, so this affected only files
  regenerated with this script. Uncorrected it inverts the sex counts and swaps
  the labels on the regression's sex coefficients. Caught by the Table 1 check in
  03_phase2_analyses.py.

## 02b_feature_reliability.py

- **Affects reported values.** OCTCDQ items are no longer shifted from 1-5 to
  0-4. The pipeline scores them raw, so each subscale runs 10 to 50, and the
  shift put the two subscale medians ten points below Table 3 for the same
  variables. Alpha, the interval and the IQR are unchanged by a constant shift;
  the medians and ranges are not.
- The note that the BFAS reverse key was unverified is removed: it is checked
  against the data by 02_descriptives_octcdq_bfas.py.
