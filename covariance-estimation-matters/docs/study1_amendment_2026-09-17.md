# Amendment log entry — paste into `study1_design_and_predictions.md`

## Amendment 2, 17 September 2026. Reference-group calibration added post hoc.

**Status.** Post hoc. This outcome was added after Study 2 had been run and its
chi-square calibration table inspected. It was not part of the original design
and is not a confirmatory test.

**What was added.** Three quantities, computed per estimator, condition and *n*,
on the reference sample itself rather than on the deviant group:

- `ref_med_ratio` — the median in-sample squared Mahalanobis distance of the
  reference sample, divided by the median of chi-square(11)
- `ref_pct_above_q95` — the percentage of the reference sample exceeding the .95
  quantile of chi-square(11), nominally 5%
- `ref_pct_above_q975` — the same at the .975 quantile, nominally 2.5%

No existing outcome was changed, removed or redefined. Random-number streams,
conditions, sample sizes, seeds and estimator settings are unchanged, so every
previously reported Study 1 result reproduces exactly.

**Why it was added.** Study 2 reports that MCD places 62.9% of the healthy-control
sample above a chi-square cutoff meant to flag 5%. Read against the nominal rate,
that number attributes the whole departure to the data. It cannot, because
chi-square is an asymptotic reference and the covariance is estimated at *n* = 62.
Some part of the departure belongs to the estimator's own finite-sample behavior
and some to the properties of the sample. Distinguishing them requires knowing
what each estimator does when its assumptions hold at that *n*, which is what the
`gaussian` condition already generates. The amendment computes the quantity rather
than adding a design.

**Prediction, recorded before the full run.** Under `gaussian`, MCD's median ratio
and exceedance rates should approach nominal as *n* increases, because the
consistency correction is asymptotic. If they do not converge, the asymptotic
explanation offered in the Discussion is wrong and must be replaced. Under the
floor conditions, the departure should persist at large *n*, because a floor is a
property of the distribution rather than of the sample size.

**Result at the 3-seed pilot, before the full 30-seed run.** Both predictions held.
Under `gaussian`, MCD ran at a median ratio of 1.24 with 30.1% exceedance at
*n* = 62, and 1.02 with 5.4% at *n* = 1000. Under `floor50`, the ratio was 2.45 at
*n* = 62 and 2.77 at *n* = 1000, with exceedance above 70% at both. The sample
covariance and Ledoit-Wolf both reached nominal by *n* = 1000 under `gaussian`,
which is a check that the calculation is right rather than a finding.

**How it is to be reported.** As an exploratory addition, labelled as such, with
the `gaussian` rows presented as the baseline against which the Study 2 table is
read. The Study 2 chi-square table gains a column referencing these values. No
inferential claim rests on it.
