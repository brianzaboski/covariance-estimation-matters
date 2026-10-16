# Manifest

Every table and figure in the manuscript and supplement, with the script that produced it
and the output file it came from. Paths are relative to `outputs/`.

Rows marked **manual** were assembled by hand from the listed source. Those are the ones
worth re-checking after any re-run, because nothing enforces the correspondence.

## Manuscript

| Exhibit | Script | Output file | Notes |
|---|---|---|---|
| Table 1. Simulation conditions | `05_study1_simulation.py` | — | design table, written from `CONDITIONS` in the script |
| Table 2. Sample statistics | — | — | built separately from the registry demographics export and not reproduced by the archived scripts; `03` checks its sex counts against the analysis file |
| Table 3. Study measure descriptives | `03_phase2_analyses.py`, `02_descriptives_octcdq_bfas.py` | `TableS_feature_descriptives_by_group.csv`, `Table3_additional_rows.csv` | manual; the seven OCTCDQ and BFAS rows come from `02`, the floor-prevalence column from the by-group file. The characterizing-measure rows (BDI-II total, aBDI, Q-LES-Q-SF, DOCS subscales, STAI, OCTCDQ total) were computed from the registry data and are not in `outputs/` |
| Table 4. Calibration against the Study 1 conditions | `05_study1_simulation.py` | `Study1_calibration.csv`, `Study1_summary.csv` | manual; reference sample *n* = 62 |
| Table 5a. Regression coefficients | `03_phase2_analyses.py` | `Table4a_coefficients.csv` | |
| Table 5b. Step statistics | `03_phase2_analyses.py` | `Table4b_step_statistics.csv` | |
| Figure 1. Estimator behavior under known conditions | `05b_figure1.py`, from the outputs of `05_study1_simulation.py` | `Fig_study1.pdf` / `.png`; plotted values in `Fig_study1_data.csv` | across-seed median lines with interquartile bands (30 seeds) |
| Figure 2. What MCD discarded | `08_fig_mcd_discarded.py` | `Fig_mcd_discarded.pdf` / `.png` | drawn from `TableS_mcd_support_composition.csv`, written by `03` |

## Supplement

The supplement itself is `docs/supplementary_materials.pdf`.

| Exhibit | Script | Output file | Notes |
|---|---|---|---|
| S1. Matched-floor conditions | `05_study1_simulation.py`, `04_invert_r.py` | `Study1_summary.csv`, `latent_R.csv` | conditions `floor_matched`, `floor_matched_corr` |
| Table S1. Calibration by condition | `05_study1_simulation.py` | `Study1_summary.csv` | manual; reference sample *n* = 5,000. Across-seed **median** for the ratio, across-seed **mean** for the percentage; contamination rows also use `*_ref_pct_above_q95_clean_mean` |
| S2 / Figure S1. Ordering of the deviant group | `05_study1_simulation.py` | `Study1_summary.csv`, figure PDF/PNG | |
| S3. Calibration narrative | `05_study1_simulation.py` | `Study1_calibration.csv` | |
| S4. Instrument psychometrics | — | — | published sources only; no output file |
| S5 / Tables S2–S3 / Figure S2. Support-fraction sweep | `07_support_fraction_sweep.py` | `TableS_support_fraction_sweep_summary.csv`, `TableS_support_fraction_sweep_byseed.csv`, `FigS_support_fraction_sweep.pdf` | 10 fractions × 30 seeds = 300 fits |
| S5. Seed sensitivity of the reported MCD fit | `07_support_fraction_sweep.py` | `TableS_support_fraction_seed_lookup.csv` | default MCD refitted over 500 seeds |
| Table 3, α column (reliability in this sample; Supplement S4 cites published values) | `02b_feature_reliability.py` | `TableS_feature_reliability.csv`, `TableS_feature_reliability_bfas_aspects.csv` | re-run after the OCTCDQ metric fix; the two subscale medians now agree with Table 3 |
| S6 / Table S4 / Figure S3. Reference sample size | `06_bootstrap_sample_size.py` | `Table3_simulation_summary.csv`, `Sim2_runs.csv` | jitter check in `TableS_jitter_sensitivity.csv` |
| S7 / Figure S4. Deviation score distributions | `03_phase2_analyses.py` | figure PDF/PNG | |
| S8 / Table S5. Directional profiling | `03_phase2_analyses.py` | `Table5_deviation_profile.csv` | top decile, *D*² = 185.65, *n* = 27 |

## Reported in text but not in a table

| Claim | Script | Output file |
|---|---|---|
| Partial correlation, 90% CI, TOST | `03_phase2_analyses.py` | `TableS_partial_r_interval_tost.csv` |
| Construct overlap *R*², VIFs | `03_phase2_analyses.py` | `TableS_construct_overlap.csv` |
| Estimator sensitivity of the incremental test | `03_phase2_analyses.py` | `TableS_estimator_sensitivity.csv` |
| Condition numbers, retained support, median distances | `03_phase2_analyses.py` | `TableS_estimator_comparison_realdata.csv` |
| Chi-square calibration of control distances | `03_phase2_analyses.py` | `TableS_chi2_calibration.csv` |
| MCD location vs. scatter decomposition | `03_phase2_analyses.py` | `TableS_mcd_location_shift.csv` |
| Step 3 influence diagnostics | `03_phase2_analyses.py` | `TableS_step3_influence.csv`, `TableS_step3_top_leverage.csv` |
| BFAS reverse-key check | `02_descriptives_octcdq_bfas.py` | `BFAS_item_diagnostics.csv` |
| Package versions | `03`, `05`, `06`, `07` | `versions_*.json`, `support_fraction_sweep_provenance.json` |

## Not for release

`python code/check_outputs.py` scans `outputs/` for these by shape.

| File | Script | Why |
|---|---|---|
| `final_analyzed_dataset.csv` | `03_phase2_analyses.py` | one row per participant |
| `TableS_step3_top_leverage.csv` | `03_phase2_analyses.py` | five individual cases with sex category; three are in a six-member category |
| `Study1_summary_v14.csv` | `05_study1_simulation.py` (pre-v1.6) | superseded; no oracle or uncontaminated-subset columns, and its values contradict Table S1 |
| `data/final_df.csv`, `data/imputed_clinical_data.csv` | `01` | registry data |

## Analysis log

Study 1's design was drafted before the simulation was run, but that document was not
time-stamped or publicly registered and is not part of this archive, so no analysis is
described as preregistered.

`docs/study1_amendment_2026-09-17.md` records the post hoc addition of the
reference-group calibration outcomes (`ref_med_ratio`, `ref_pct_above_q95`,
`ref_pct_above_q975`), including the prediction made before the full run. The
matched-floor conditions were also added post hoc; see `CHANGELOG.md`
(`05_study1_simulation.py`, v1.3 and v1.4). Study 1's existing outcomes and
random-number streams were unchanged, so all previously reported values reproduce.

## Synthetic data

`data/synthetic/make_synthetic_data.py` builds `synthetic_clinical_data.csv` from the
aggregates in `TableS_feature_descriptives_by_group.csv` and the latent correlation
matrix from `04_invert_r.py`, recording every constant it used in
`synthetic_data_constants.json`. Age and sex parameters come from the analytic file, with the median-imputation spike in
age reinstated. The outcome is generated from the fitted Step 3 equation in
`Table4a_coefficients.csv` with residual noise sized to the Step 2 R² in
`Table4b_step_statistics.csv`, so no part of it is assumed. The synthetic clinical group
carries the control correlation structure, since only the control matrix is published.
