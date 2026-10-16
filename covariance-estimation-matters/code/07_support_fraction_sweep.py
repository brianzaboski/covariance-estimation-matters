# -*- coding: utf-8 -*-
"""Support-fraction sweep for MCD on the healthy-control sample.

Is MCD's behaviour on a screened reference sample an artefact of scikit-learn's
default support fraction? MCD is refitted from 0.50 to 0.95 in steps of 0.05 over
30 seeds, recording raw and reweighted support size, restriction of range,
conditioning, numerical rank, Frobenius and precision-matrix error against a
per-seed benchmark, floor composition in the retained and excluded sets, and
whether any feature is exactly constant within the retained subset.

Three estimators are fitted alongside the grid and are not points on it: MCD at
its default support fraction, MCD at 1.00, and the plain sample covariance. The
sample covariance is not the support_fraction -> 1 limit of MCD, because MCD
still reweights; at 1.00 it excludes 5 of 62 controls at that step. Ledoit-Wolf
is fitted as a further reference. The same grid runs on synthetic control
samples of N = 2,000, benchmarked against the real control sample rather than
against themselves.

Distances are taken about a common center, the reference-sample mean, with
own-center distances kept separately. They are computed from each estimator's
own covariance with a pseudo-inverse, so near-singular MCD fits return finite
values, and singularity is judged from the eigenvalue spectrum against machine
precision rather than a fixed cutoff.

The seed lookup refits the default MCD across 500 seeds on the real control
sample, which shows how far the reported fit depends on the seed. At n = 62 and
p = 11 the default support size is int(0.60 * 62) = 37, so the default and the
0.60 setting give identical fits.

Reads   data/imputed_clinical_data.csv
Writes  TableS_support_fraction_sweep_byseed.csv,
        TableS_support_fraction_sweep_summary.csv,
        TableS_support_fraction_seed_lookup.csv,
        FigS_support_fraction_sweep.pdf/.png,
        support_fraction_sweep_provenance.json

Usage   python code/07_support_fraction_sweep.py [--n-seeds 5] [--skip-large]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import platform
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import pinvh
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # safe on headless machines; comment out to use plt.show()
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# parents[1] when run as a file; the working directory when pasted into an
# interactive session, where __file__ does not exist
BASE = (Path(__file__).resolve().parents[1]
        if "__file__" in globals() else Path.cwd())
DEFAULT_IN = BASE / "data" / "imputed_clinical_data.csv"
DEFAULT_OUT = BASE / "outputs"

SEED0 = 42                 # seeds are SEED0, SEED0+1, ..., SEED0+N_SEEDS-1
N_SEEDS = 30               # matches the main simulation
SUPPORT_FRACTIONS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
BENCH_N = 100_000          # benchmark draw size, matches the main simulation
LARGE_N = 2_000            # corroboration at a size no clinical study would reach
JITTER = 0.05              # Gaussian jitter as a fraction of each variable's SD

# Originally reported default MCD fit on the real HC sample (manuscript, Phase 2)
TARGET_COND = 172.0
TARGET_RETAINED = 38
COND_TOL = 1.0             # |condition number - TARGET_COND| <= COND_TOL counts as a match
LOOKUP_SEEDS = range(0, 500)   # seeds searched for the reported fit (default MCD, real HC)

SCALE_MIN = {"shaps_total_score_snaith": 0, "docs_total": 0}

MCD_FEATURES = [
    "gbdi_total_score",
    "shaps_total_score_snaith",
    "psc_total_score",
    "docs_total",
    "octcdq_incompleteness_score",
    "octcdq_harm_score",
    "bfas_domain_neuroticism",
    "bfas_domain_extraversion",
    "bfas_domain_conscientiousness",
    "bfas_domain_agreeableness",
    "bfas_domain_openness_intellect",
]
DOCS_COL = "docs_total"

# Colorblind-safe (Okabe-Ito)
C_MCD = "#D55E00"      # vermillion
C_LW = "#0072B2"       # blue
C_FULL = "#009E73"     # bluish green (sample covariance)
C_BENCH = "#000000"    # black
C_ACCENT = "#7A1FA2"   # purple, panel A only


# ----------------------------------------------------------------------------
# Numerical helpers
# ----------------------------------------------------------------------------

def spectrum(cov: np.ndarray) -> tuple[float, float, float, bool]:
    """Return (lambda_min, lambda_max, condition_number, is_singular).

    Uses the symmetric eigenvalue routine rather than `np.linalg.cond`, so that
    singularity is judged against machine precision scaled to the matrix rather
    than against an arbitrary cutoff.
    """
    eig = np.linalg.eigvalsh((cov + cov.T) / 2.0)
    lo, hi = float(eig.min()), float(eig.max())
    tol = max(hi, 0.0) * cov.shape[0] * np.finfo(float).eps
    singular = (lo <= tol)
    cond = np.inf if singular else hi / lo
    return lo, hi, cond, bool(singular)


def precision_and_rank(cov: np.ndarray) -> tuple[np.ndarray, int]:
    """Pseudo-inverse (as scikit-learn uses internally) and its numerical rank."""
    prec, rank = pinvh(cov, return_rank=True)
    return prec, int(rank)


def mahalanobis_sq(X: np.ndarray, loc: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance about `loc` with precision matrix `prec`.

    When the covariance is singular, the pseudo-inverse silently drops the
    degenerate direction, so distances are computed on fewer dimensions. The
    rank is recorded alongside every fit so this is never hidden.
    """
    d = X - loc
    return np.einsum("ij,jk,ik->i", d, prec, d)


def rng_for(seed: int, role: int) -> np.random.Generator:
    """Independent random stream per (seed, role)."""
    return np.random.default_rng([seed, role])


def bootstrap_jitter(src: pd.DataFrame, n: int, rng: np.random.Generator,
                     frac: float = JITTER) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample `src` with replacement and add Gaussian jitter.

    Same generator as simulation_v2.py. Returns (jittered, un-jittered) so floor
    composition can be computed on the original values.
    """
    idx = rng.integers(0, len(src), size=n)
    raw = src.iloc[idx].reset_index(drop=True)
    sd = src.std(axis=0, ddof=1).to_numpy()
    noisy = raw + rng.normal(0, 1, raw.shape) * (frac * sd)
    return noisy, raw


def summarize_fit(
    cov: np.ndarray,
    own_loc: np.ndarray,
    support_mask: np.ndarray,
    raw_vals: pd.DataFrame,
    hc_scaled: np.ndarray,
    dev_scaled: np.ndarray,
    bench_cov: np.ndarray,
    bench_prec: np.ndarray,
    bench_cond: float,
    *,
    estimator: str,
    support_fraction: float | None,
    setting: str,
    n: int,
    seed: int,
    raw_support_size: int | None = None,
    raw_cond: float | None = None,
    n_warnings: int = 0,
) -> dict:
    lo, hi, cond, singular = spectrum(cov)
    prec, rank = precision_and_rank(cov)
    frob = float(np.linalg.norm(cov - bench_cov, "fro"))
    prec_rel = float(np.linalg.norm(prec - bench_prec, "fro") / np.linalg.norm(bench_prec, "fro"))

    docs_raw = raw_vals[DOCS_COL].to_numpy()
    row_floor = {}
    if (~support_mask).any():
        excl = float(docs_raw[~support_mask].mean())
        incl = float(docs_raw[support_mask].mean())
        ratio = excl / incl if incl > 0 else np.nan
        n_excl = int((~support_mask).sum())
    else:
        excl = incl = ratio = np.nan
        n_excl = 0
    for col, mn in SCALE_MIN.items():
        at_min = np.isclose(raw_vals[col].to_numpy(), mn)
        tag = col.split("_")[0]
        row_floor[f"pct_min_{tag}_retained"] = 100 * at_min[support_mask].mean()
        row_floor[f"pct_min_{tag}_excluded"] = 100 * at_min[~support_mask].mean() if n_excl else np.nan

    # Degenerate-column diagnostic (see v1 notes). Computed on the scaled
    # reference values within the selected support; untrimmed fits act as a control.
    sub = hc_scaled[support_mask]
    ranges = np.ptp(sub, axis=0)
    zero_var_idx = np.flatnonzero(ranges == 0)
    zero_var_features = ";".join(MCD_FEATURES[j] for j in zero_var_idx)
    min_feature_var = float(np.min(np.var(sub, axis=0)))

    # Common center: the reference-sample mean
    center = hc_scaled.mean(axis=0)
    hc_d2 = mahalanobis_sq(hc_scaled, center, prec)
    dev_d2 = mahalanobis_sq(dev_scaled, center, prec)
    hc_med, dev_med = float(np.median(hc_d2)), float(np.median(dev_d2))
    # Own center (what est.mahalanobis() returns); differs only for MCD
    hc_own = float(np.median(mahalanobis_sq(hc_scaled, own_loc, prec)))
    dev_own = float(np.median(mahalanobis_sq(dev_scaled, own_loc, prec)))

    return dict(
        setting=setting,
        estimator=estimator,
        support_fraction=support_fraction,
        seed=seed,
        n=n,
        support_retained=int(support_mask.sum()),
        raw_support_size=raw_support_size,
        n_excluded=n_excl,
        excl_docs=excl,
        incl_docs=incl,
        restrict_ratio=ratio,
        **row_floor,
        lambda_min=lo,
        lambda_max=hi,
        condition_number=cond,
        raw_condition_number=raw_cond,
        is_singular=singular,
        rank=rank,
        n_zero_var_support=int(zero_var_idx.size),
        zero_var_features=zero_var_features,
        min_feature_var_support=min_feature_var,
        frobenius_error=frob,
        precision_rel_error=prec_rel,
        benchmark_condition_number=bench_cond,
        hc_median_d2=hc_med,
        dev_median_d2=dev_med,
        ratio_dev_hc=dev_med / hc_med if hc_med > 0 else np.nan,
        hc_median_d2_own_center=hc_own,
        dev_median_d2_own_center=dev_own,
        location_shift_norm=float(np.linalg.norm(own_loc - center)),
        mcd_warnings=n_warnings,
    )


# ----------------------------------------------------------------------------
# One seed of one setting
# ----------------------------------------------------------------------------

def fit_mcd(X: np.ndarray, sf: float | None, seed: int):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = MinCovDet(support_fraction=sf, random_state=seed).fit(X)
    return m, len(w)


def run_setting_seed(
    setting: str,
    ref_noisy: pd.DataFrame,
    ref_raw: pd.DataFrame,
    bench_src: pd.DataFrame,
    dev_raw: pd.DataFrame,
    scaler: StandardScaler,
    seed: int,
) -> list[dict]:
    """One seed of one setting.

    ref_noisy : reference sample as analysed (real HC, or jittered synthetic HC)
    ref_raw   : same rows without jitter (for floor composition)
    bench_src : ALWAYS the real HC sample; the benchmark is drawn from it
                independently of the reference sample.
    """
    hc_scaled = scaler.transform(ref_noisy[MCD_FEATURES])
    dev_scaled = scaler.transform(dev_raw[MCD_FEATURES])
    n = len(ref_noisy)

    bench_noisy, _ = bootstrap_jitter(bench_src[MCD_FEATURES], BENCH_N, rng_for(seed, 1))
    bench = scaler.transform(bench_noisy)
    bench_cov = np.cov(bench, rowvar=False, bias=True)
    bench_prec, _ = precision_and_rank(bench_cov)
    _, _, bench_cond, _ = spectrum(bench_cov)

    rows: list[dict] = []
    common = dict(
        raw_vals=ref_raw,
        hc_scaled=hc_scaled,
        dev_scaled=dev_scaled,
        bench_cov=bench_cov,
        bench_prec=bench_prec,
        bench_cond=bench_cond,
        setting=setting,
        n=n,
        seed=seed,
    )

    # --- MCD: grid, scikit-learn default, and support_fraction = 1.0 ----------
    for label, sf in ([("MCD", f) for f in SUPPORT_FRACTIONS]
                      + [("MCD (default)", None), ("MCD (sf = 1.0)", 1.0)]):
        try:
            m, n_warn = fit_mcd(hc_scaled, sf, seed)
        except Exception as exc:  # e.g. support set too small for n_features
            rows.append(dict(setting=setting, estimator=label,
                             support_fraction=np.nan if sf is None else sf,
                             seed=seed, n=n, fit_failed=True, fit_error=str(exc)))
            continue
        _, _, raw_cond, _ = spectrum(m.raw_covariance_)
        rows.append(
            summarize_fit(
                m.covariance_, m.location_, m.support_,
                estimator=label,
                support_fraction=np.nan if sf is None else sf,
                raw_support_size=int(m.raw_support_.sum()),
                raw_cond=raw_cond, n_warnings=n_warn,
                **common,
            )
        )

    # --- Plain sample covariance (no subset selection, no reweighting) --------
    full = EmpiricalCovariance().fit(hc_scaled)
    rows.append(
        summarize_fit(
            full.covariance_, full.location_, np.ones(n, bool),
            estimator="sample covariance", support_fraction=np.nan, **common,
        )
    )

    # --- Ledoit-Wolf reference ------------------------------------------------
    lw = LedoitWolf().fit(hc_scaled)
    r = summarize_fit(
        lw.covariance_, lw.location_, np.ones(n, bool),
        estimator="Ledoit-Wolf", support_fraction=np.nan, **common,
    )
    r["shrinkage"] = float(lw.shrinkage_)
    rows.append(r)

    return rows


# ----------------------------------------------------------------------------
# Summarisation
# ----------------------------------------------------------------------------

def summarize_across_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-seed table to the across-seed summary you would publish."""
    num_cols = [
        "support_retained", "raw_support_size", "n_excluded",
        "excl_docs", "incl_docs", "restrict_ratio",
        "pct_min_shaps_retained", "pct_min_shaps_excluded",
        "pct_min_docs_retained", "pct_min_docs_excluded",
        "condition_number", "raw_condition_number", "rank",
        "frobenius_error", "precision_rel_error", "benchmark_condition_number",
        "hc_median_d2", "dev_median_d2", "ratio_dev_hc",
        "hc_median_d2_own_center", "dev_median_d2_own_center", "location_shift_norm",
        "n_zero_var_support", "min_feature_var_support",
    ]
    d = df.copy()
    if "fit_failed" in d.columns:
        d = d[d["fit_failed"] != True]  # noqa: E712
    # Condition number is infinite when singular; summarise finite fits only,
    # and report singularity separately as a rate.
    d_fin = d.copy()
    d_fin.loc[~np.isfinite(d_fin["condition_number"]), "condition_number"] = np.nan

    key = ["setting", "estimator", "support_fraction"]
    g = d_fin.groupby(key, dropna=False)
    num_cols = [c for c in num_cols if c in d_fin.columns]
    out = g[num_cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    out["n_singular"] = d.groupby(key, dropna=False)["is_singular"].sum()
    out["pct_singular"] = 100.0 * out["n_singular"] / out["n_seeds"]
    out["pct_rank_deficient"] = 100.0 * d.groupby(key, dropna=False)["rank"].apply(
        lambda v: (v < len(MCD_FEATURES)).mean()
    )
    out["pct_fits_with_warnings"] = 100.0 * d.groupby(key, dropna=False)["mcd_warnings"].apply(
        lambda v: (v > 0).mean()
    )
    out["pct_zero_var"] = 100.0 * d.groupby(key, dropna=False)["n_zero_var_support"].apply(
        lambda v: (v > 0).mean()
    )
    return out.reset_index()


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------

def save_figure(fig, path: Path, **kwargs) -> Path | None:
    """Write `fig` to `path`, tolerating locked files and sync-layer interference.

    Writes to a temporary file in the same directory and then atomically replaces
    the target. On Windows an open viewer or a OneDrive-backed folder can make a
    direct overwrite fail with EACCES or EINVAL; in that case the figure is saved
    under a timestamped name instead so the run is never lost.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=path.suffix, dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        fig.savefig(tmp_path, **kwargs)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"  ! could not render {path.name}: {exc}")
        return None
    try:
        os.replace(tmp_path, path)
        return path
    except OSError as exc:
        alt = path.with_name(f"{path.stem}_{time.strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        try:
            os.replace(tmp_path, alt)
            print(f"  ! could not overwrite {path.name} ({exc.__class__.__name__}: {exc}).")
            print(f"    Saved as {alt.name} instead. Close any program holding the original,")
            print(f"    or pass --out to a folder outside OneDrive/Dropbox.")
            return alt
        except OSError as exc2:
            tmp_path.unlink(missing_ok=True)
            print(f"  ! could not write {path.name} or a fallback name: {exc2}")
            return None


def make_figure(real_summary: pd.DataFrame, real_by_seed: pd.DataFrame, out_stem: Path,
                dpi: int = 600) -> list[Path]:
    """Four-panel figure for the real-HC sweep.

    Panel D deliberately does NOT plot a connected line through conditional
    means. The number of seeds yielding an invertible matrix varies from 0 to 30
    across the support-fraction grid, so a mean over survivors is computed on a
    different denominator at every x value. It is drawn instead as a failure
    count (top) plus the per-seed condition numbers of the fits that succeeded
    (bottom), which keeps the shrinking denominator visible and stops singular
    fits from being assigned an invented magnitude on a quantitative axis.
    """
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    })

    s = real_summary
    mcd = s[s["estimator"] == "MCD"].sort_values("support_fraction")
    lw = s[s["estimator"] == "Ledoit-Wolf"].iloc[0]
    full = s[s["estimator"] == "sample covariance"].iloc[0]
    x = mcd["support_fraction"].to_numpy(dtype=float)

    seeds_df = real_by_seed.copy()
    if "fit_failed" in seeds_df.columns:
        seeds_df = seeds_df[seeds_df["fit_failed"] != True]  # noqa: E712
    mcd_seeds = seeds_df[seeds_df["estimator"] == "MCD"]
    n_seeds_total = int(mcd_seeds["seed"].nunique())

    bench_mean = float(seeds_df["benchmark_condition_number"].mean())
    bench_sd = float(seeds_df["benchmark_condition_number"].std())
    lw_cond = float(lw["condition_number_mean"])
    full_cond = float(full["condition_number_mean"])

    fig = plt.figure(figsize=(9.6, 7.8))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    gsD = gs[1, 1].subgridspec(2, 1, height_ratios=[1, 3.4], hspace=0.10)
    axDt = fig.add_subplot(gsD[0])
    axDb = fig.add_subplot(gsD[1])

    def band(ax, xv, mean, sd, color, label=None, marker="o"):
        mean = np.asarray(mean, float)
        sd = np.nan_to_num(np.asarray(sd, float))
        ax.plot(xv, mean, marker=marker, ms=4, lw=1.6, color=color, label=label, zorder=3)
        ax.fill_between(xv, mean - sd, mean + sd, color=color, alpha=0.18, lw=0, zorder=2)

    # --- A: restriction of range ---------------------------------------------
    band(axA, x, mcd["restrict_ratio_mean"], mcd["restrict_ratio_std"], C_ACCENT)
    axA.axhline(1.0, color="gray", ls=":", lw=1)
    axA.text(0.505, 0.94, "no restriction (ratio = 1)", color="gray", fontsize=7.5, va="top")
    top = float(np.nanmax(mcd["restrict_ratio_mean"] + mcd["restrict_ratio_std"].fillna(0)))
    axA.set_ylim(0, top * 1.30)
    axA.set_xlabel("MCD support fraction")
    axA.set_ylabel("Restriction ratio\n(excluded / included mean DOCS)")
    axA.set_title("A.  Excluded controls score higher\nat every support fraction", loc="left")
    for xi, yi, ni in zip(x, mcd["restrict_ratio_mean"], mcd["n_excluded_mean"]):
        if np.isfinite(yi):
            axA.annotate(f"{ni:.0f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=6.5, color="gray")
    axA.annotate(f"labels = controls excluded\nall {n_seeds_total} seeds contribute",
                 xy=(0.98, 0.97), xycoords="axes fraction", ha="right", va="top",
                 fontsize=7, color="gray")

    # --- B: requested vs realised trimming -----------------------------------
    n_hc = float(full["support_retained_mean"])
    band(axB, x, mcd["raw_support_size_mean"] / n_hc, mcd["raw_support_size_std"] / n_hc,
         C_MCD, label="raw support", marker="s")
    band(axB, x, mcd["support_retained_mean"] / n_hc, mcd["support_retained_std"] / n_hc,
         C_LW, label="after reweighting", marker="o")
    axB.plot(x, x, color="gray", ls="--", lw=1, label="requested fraction")
    axB.set_ylim(0, 1.05)
    axB.set_xlabel("MCD support fraction")
    axB.set_ylabel("Proportion of controls retained")
    axB.set_title("B.  Reweighting adds controls at low fractions\nand removes them at high ones", loc="left")
    axB.legend(fontsize=7.5, loc="lower right")

    # --- C: estimation error against the benchmark ---------------------------
    # The sample covariance is deliberately omitted here: in the real-HC setting
    # the benchmark is generated by resampling these same controls, so the plain
    # sample covariance reproduces it near-definitionally. The informative
    # untrimmed comparison is the independent synthetic setting (Table S2).
    band(axC, x, mcd["frobenius_error_mean"], mcd["frobenius_error_std"],
         C_MCD, label="MCD")
    axC.axhline(float(lw["frobenius_error_mean"]), color=C_LW, ls="--", lw=1.5,
                label=f"Ledoit-Wolf ({lw['frobenius_error_mean']:.2f})")
    axC.axhline(0.0, color=C_BENCH, ls=":", lw=1, label="benchmark (0)")
    _top = float(np.nanmax(mcd["frobenius_error_mean"])) * 1.18
    axC.set_ylim(-0.04 * _top, _top)
    axC.set_xlabel("MCD support fraction")
    axC.set_ylabel("Frobenius estimation error")
    axC.set_title("C.  No support fraction recovers\nthe benchmark covariance", loc="left")
    axC.legend(fontsize=7.5, frameon=True, framealpha=0.92, edgecolor="none")

    # --- D (top): how often no invertible matrix is produced -------------------
    n_sing = mcd["n_singular"].to_numpy(dtype=float)
    axDt.bar(x, n_sing, width=0.032, color=C_MCD, alpha=0.85,
             hatch="///", edgecolor="white", linewidth=0.6)
    axDt.set_ylim(0, n_seeds_total * 1.28)
    axDt.set_ylabel(f"no inverse\n(of {n_seeds_total})", fontsize=7.5)
    axDt.set_xlim(0.475, 0.975)
    axDt.tick_params(labelbottom=False, labelsize=7.5)
    axDt.set_yticks([0, n_seeds_total])
    for xi, ni in zip(x, n_sing):
        if ni > 0:
            axDt.annotate(f"{ni:.0f}", (xi, ni), textcoords="offset points",
                          xytext=(0, 2), ha="center", fontsize=6.5, color=C_MCD)
    axDt.set_title("D.  Singular fits and conditioning\nacross support fractions", loc="left")

    # --- D (bottom): per-seed condition numbers of the fits that succeeded -----
    rng = np.random.RandomState(0)
    fin = mcd_seeds[np.isfinite(mcd_seeds["condition_number"])]
    axDb.axhspan(max(bench_mean - bench_sd, 1e-9), bench_mean + bench_sd,
                 color=C_BENCH, alpha=0.10, lw=0, zorder=1)
    axDb.axhline(bench_mean, color=C_BENCH, ls=":", lw=1.2,
                 label=f"benchmark ({bench_mean:.1f})", zorder=3)
    axDb.axhline(lw_cond, color=C_LW, ls="--", lw=1.5,
                 label=f"Ledoit-Wolf ({lw_cond:.1f})", zorder=2)
    axDb.axhline(full_cond, color=C_FULL, ls="-.", lw=1.5,
                 label=f"sample covariance ({full_cond:.1f})", zorder=2)
    # benchmark and the untrimmed sample sit close together by construction;
    # draw the benchmark last so it stays visible where they nearly coincide.
    if len(fin):
        jx = fin["support_fraction"].to_numpy(float) + rng.uniform(-0.011, 0.011, len(fin))
        axDb.scatter(jx, fin["condition_number"], s=11, color=C_MCD, alpha=0.55,
                     edgecolor="none", zorder=4, label="MCD, one point per seed")
    axDb.set_yscale("log")
    if len(fin):
        lo_lim = min(lw_cond, bench_mean, float(fin["condition_number"].min())) * 0.45
        hi_lim = float(fin["condition_number"].max()) * 2.4
    else:
        lo_lim, hi_lim = min(lw_cond, bench_mean) * 0.45, max(lw_cond, bench_mean) * 10
    axDb.set_ylim(lo_lim, hi_lim)
    axDb.set_xlim(0.475, 0.975)
    axDb.set_xlabel("MCD support fraction")
    axDb.set_ylabel("Condition number (log scale)")
    axDb.legend(fontsize=7.5, loc="upper left", ncol=2, columnspacing=1.0,
                frameon=True, framealpha=0.92, edgecolor="none")
    # denominator for each column, so no point is read as a like-for-like mean
    n_fin = fin.groupby("support_fraction").size() if len(fin) else pd.Series(dtype=int)
    for xi in x:
        axDb.annotate(f"{int(n_fin.get(xi, 0))}", xy=(xi, 0.035), xycoords=("data", "axes fraction"),
                      ha="center", va="bottom", fontsize=6.5, color="gray")
    axDb.annotate("n fits with an invertible matrix", xy=(0.5, 0.115), xycoords="axes fraction",
                  ha="center", va="bottom", fontsize=6.5, color="gray")
    # headroom so the legend clears the point cloud and the count row clears the axis
    axDb.set_ylim(lo_lim * 0.75, hi_lim * 1.6)

    written = [
        save_figure(fig, out_stem.with_suffix(".pdf")),
        save_figure(fig, out_stem.with_suffix(".png"), dpi=dpi),
    ]
    plt.close(fig)
    return [w for w in written if w is not None]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--dpi", type=int, default=600, help="raster resolution for the PNG")
    ap.add_argument("--skip-large", action="store_true",
                    help="skip the synthetic N=2000 corroboration setting")
    args = ap.parse_args()

    t0 = time.time()
    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    missing = [c for c in MCD_FEATURES + ["group"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}")
    if df[MCD_FEATURES].isna().any().any():
        raise SystemExit("Feature matrix contains NaNs; this script expects the imputed file.")

    hc = df.loc[df["group"] == "HC", MCD_FEATURES].reset_index(drop=True)
    ocd = df.loc[df["group"] == "OCD", MCD_FEATURES].reset_index(drop=True)
    print(f"Loaded HC n={len(hc)}, OCD n={len(ocd)}, p={len(MCD_FEATURES)}")

    scaler = StandardScaler().fit(hc)
    seeds = [SEED0 + i for i in range(args.n_seeds)]

    all_rows: list[dict] = []
    for i, seed in enumerate(seeds, 1):
        all_rows += run_setting_seed(f"real HC (N={len(hc)})", hc, hc, hc, ocd, scaler, seed)
        if not args.skip_large:
            big_noisy, big_raw = bootstrap_jitter(hc, LARGE_N, rng_for(seed, 2))
            all_rows += run_setting_seed(f"synthetic HC (N={LARGE_N})", big_noisy, big_raw,
                                         hc, ocd, scaler, seed)
        print(f"  seed {seed} done ({i}/{len(seeds)}, {time.time()-t0:.0f}s elapsed)")

    by_seed = pd.DataFrame(all_rows)
    summary = summarize_across_seeds(by_seed)

    by_seed_path = args.out / "TableS_support_fraction_sweep_byseed.csv"
    summary_path = args.out / "TableS_support_fraction_sweep_summary.csv"
    for obj, dest in ((by_seed, by_seed_path), (summary.round(4), summary_path)):
        try:
            obj.to_csv(dest, index=False)
        except OSError as exc:
            alt = dest.with_name(f"{dest.stem}_{time.strftime('%Y%m%d_%H%M%S')}{dest.suffix}")
            obj.to_csv(alt, index=False)
            print(f"  ! could not write {dest.name} ({exc}); saved as {alt.name}")

    real_label = f"real HC (N={len(hc)})"
    real_summary = summary[summary["setting"] == real_label]
    real_by_seed = by_seed[by_seed["setting"] == real_label]
    fig_stem = args.out / "FigS_support_fraction_sweep"
    figs = make_figure(real_summary, real_by_seed, fig_stem, dpi=args.dpi)

    # --- console report -------------------------------------------------------
    show = [
        "estimator", "support_fraction", "support_retained_mean", "raw_support_size_mean",
        "restrict_ratio_mean", "pct_min_shaps_retained_mean", "pct_min_shaps_excluded_mean",
        "condition_number_mean", "pct_singular", "pct_rank_deficient",
        "frobenius_error_mean", "frobenius_error_std", "precision_rel_error_mean",
        "dev_median_d2_mean", "dev_median_d2_own_center_mean",
    ]
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:9.3f}")
    print("\n=== Real HC sweep, summarised across seeds ===")
    print(real_summary[show].to_string(index=False))
    if not args.skip_large:
        print(f"\n=== Synthetic HC (N={LARGE_N}) corroboration ===")
        print(summary[summary["setting"] != real_label][show].to_string(index=False))

    # --- seed lookup: which seed reproduces the reported default fit? ---------
    # Default MCD on the real HC sample is fast, so search a wide seed range
    # (covers common choices such as 0, 1, 42, 123) rather than only the sweep seeds.
    hc_scaled_real = scaler.transform(hc)
    look_rows = []
    for sd_ in LOOKUP_SEEDS:
        m, _ = fit_mcd(hc_scaled_real, None, sd_)
        _, _, c_, _ = spectrum(m.covariance_)
        prec_, rank_ = precision_and_rank(m.covariance_)
        look_rows.append(dict(
            seed=sd_, support_retained=int(m.support_.sum()), condition_number=c_, rank=rank_,
            hc_median_d2_own_center=float(np.median(m.mahalanobis(hc_scaled_real))),
            dev_median_d2_own_center=float(np.median(m.mahalanobis(scaler.transform(ocd)))),
        ))
    look = pd.DataFrame(look_rows)
    look["matches_reported_fit"] = (
        (look["support_retained"] == TARGET_RETAINED)
        & ((look["condition_number"] - TARGET_COND).abs() <= COND_TOL)
    )
    lookup_path = args.out / "TableS_support_fraction_seed_lookup.csv"
    look.to_csv(lookup_path, index=False)
    n_distinct = look.round({"condition_number": 3})[["support_retained", "condition_number"]].drop_duplicates().shape[0]
    print(f"\n=== Seed lookup: default MCD on real HC, seeds {LOOKUP_SEEDS.start}-{LOOKUP_SEEDS.stop - 1} "
          f"(target: {TARGET_RETAINED} retained, condition number {TARGET_COND:.0f}) ===")
    print(f"{n_distinct} distinct fits across {len(look)} seeds; "
          f"{(look['rank'] < len(MCD_FEATURES)).mean():.0%} rank deficient")
    hits = look.loc[look["matches_reported_fit"], "seed"].tolist()
    if hits:
        print(f"Matching seed(s): {hits[:20]}{' ...' if len(hits) > 20 else ''}")
        print(f"Set MCD_SEED in 03_phase2_analyses.py to {42 if 42 in hits else hits[0]}.")
        print(look[look["matches_reported_fit"]].head(5).to_string(index=False))
        print("Compare the own-center medians with the reported 16.58 / 1,752.30. If they differ,")
        print("the original script used a different centering or preprocessing, not a different seed.")
    else:
        closest = look.assign(gap=(look["condition_number"] - TARGET_COND).abs()).nsmallest(5, "gap")
        print("No seed reproduces the reported fit. Closest fits:")
        print(closest.drop(columns="gap").to_string(index=False))
        print("The original fit was probably unseeded or used different preprocessing;")
        print("report the across-seed distribution of the default fit instead.")
    dflt = by_seed[(by_seed["setting"] == real_label) & (by_seed["estimator"] == "MCD (default)")]
    # Sanity check: at n = 62, p = 11 the default equals support_fraction = 0.60
    g60 = by_seed[(by_seed["setting"] == real_label) & (by_seed["estimator"] == "MCD")
                  & (by_seed["support_fraction"] == 0.60)].set_index("seed")["condition_number"]
    same = np.allclose(dflt.set_index("seed")["condition_number"].reindex(g60.index), g60, equal_nan=True)
    print(f"Default fit identical to support_fraction = 0.60 on real HC: {same}")

    # --- does degeneracy explain singularity? ---------------------------------
    real_seeds = by_seed[(by_seed["setting"] == real_label) & (by_seed["estimator"] == "MCD")]
    if "is_singular" in real_seeds.columns:
        ct = pd.crosstab(
            real_seeds["n_zero_var_support"] > 0,
            real_seeds["is_singular"].astype(bool),
            rownames=["constant feature in support"], colnames=["singular covariance"],
        )
        print("\n=== Degenerate-column check (real HC, MCD fits, all seeds x fractions) ===")
        print(ct.to_string())
        named = real_seeds.loc[real_seeds["n_zero_var_support"] > 0, "zero_var_features"]
        if len(named):
            print("\nFeatures found constant within a support set (count of fits):")
            print(named.str.split(";").explode().value_counts().to_string())
        else:
            print("\nNo feature was ever exactly constant within a support set.")

    prov = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "data_file": args.data.name, "n_hc": int(len(hc)), "n_ocd": int(len(ocd)),
        "seeds": seeds, "support_fractions": SUPPORT_FRACTIONS,
        "benchmark_n": BENCH_N, "large_n": LARGE_N, "jitter_frac": JITTER,
        "benchmark_source": "real HC sample (independent of the reference sample)",
        "distance_center": "reference-sample mean (own-center columns also saved)",
        "target_fit": {"condition_number": TARGET_COND, "retained": TARGET_RETAINED},
        "features": MCD_FEATURES,
    }
    (args.out / "support_fraction_sweep_provenance.json").write_text(json.dumps(prov, indent=2))

    print("\nSaved:")
    for f in [by_seed_path, summary_path, lookup_path, *figs]:
        print(f"  {f}")
    print(f"Total runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
