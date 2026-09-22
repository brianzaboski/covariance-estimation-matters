"""Recover the latent correlation matrix behind the observed control correlations.

Censoring attenuates correlations, so the observed control matrix cannot serve as
the simulation's latent structure: censoring it a second time would attenuate
everything twice. At fixed floor proportions censoring is a monotone map from
latent to observed correlation, so it inverts by a grid search with common random
numbers, pair by pair. The result is projected to the nearest positive-definite
matrix, since pairwise inversion does not guarantee a jointly valid one.

Reads   outputs/TableS_feature_correlations_hc.csv (written by 03)
Writes  outputs/latent_R.csv, outputs/latent_R.npy

The matrix is currently pasted into 05_study1_simulation.py as R_MATCHED. If you
re-run this on different data, update that constant.

Usage   python code/04_invert_r.py
"""
from pathlib import Path

import numpy as np, pandas as pd
from scipy import stats

# parents[1] when run as a file; the working directory when pasted into an
# interactive session, where __file__ does not exist
ROOT = (Path(__file__).resolve().parents[1]
        if "__file__" in globals() else Path.cwd())
OUTPUTS = ROOT / "outputs"
# written by 03_phase2_analyses.py
OBS = pd.read_csv(OUTPUTS / "TableS_feature_correlations_hc.csv", index_col=0)
FEAT = list(OBS.index)
PI = dict(zip(FEAT, [0.387, 0.935, 0.290, 0.452, 0.226, 0.306] + [0.0] * 5))
P = len(FEAT)
CUT = {f: (stats.norm.ppf(PI[f]) if PI[f] > 0 else -np.inf) for f in FEAT}

rng = np.random.default_rng(7)
NMC, GRID = 400_000, np.round(np.arange(-0.97, 0.9751, 0.025), 4)
base = rng.standard_normal((NMC, 2))

def censored_r(rho, ci, cj):
    """Correlation after censoring a bivariate normal at cut-points ci, cj."""
    a = base[:, 0]
    b = rho * base[:, 0] + np.sqrt(max(1 - rho**2, 0)) * base[:, 1]
    return np.corrcoef(np.maximum(a, ci), np.maximum(b, cj))[0, 1]

L = np.eye(P)
rows = []
for i in range(P):
    for j in range(i + 1, P):
        fi, fj, target = FEAT[i], FEAT[j], OBS.iloc[i, j]
        ci, cj = CUT[fi], CUT[fj]
        if ci == -np.inf and cj == -np.inf:          # neither censored: identity
            lat = target
        else:
            curve = np.array([censored_r(r, ci, cj) for r in GRID])
            o = np.argsort(curve)
            lat = float(np.interp(target, curve[o], GRID[o]))
        L[i, j] = L[j, i] = lat
        rows.append(dict(a=fi[:22], b=fj[:22], observed=target, latent=round(lat, 3),
                         attenuation=round(lat - target, 3)))

ev = np.linalg.eigvalsh(L)
print(f"inverted matrix: min eigenvalue {ev.min():.4f}  ({'PSD' if ev.min() > 0 else 'NOT PSD'})")
if ev.min() <= 1e-6:                                  # nearest PSD, then renormalise
    w, V = np.linalg.eigh(L)
    L = V @ np.diag(np.clip(w, 1e-4, None)) @ V.T
    d = np.sqrt(np.diag(L)); L = L / np.outer(d, d)
    print(f"  projected to nearest PSD; min eigenvalue now {np.linalg.eigvalsh(L).min():.4f}")

t = pd.DataFrame(rows).reindex(pd.DataFrame(rows).attenuation.abs().sort_values(ascending=False).index)
print("\nLargest attenuations (latent minus observed):")
print(t.head(8).to_string(index=False))
print(f"\nmean |attenuation| = {t.attenuation.abs().mean():.3f}")
np.save(OUTPUTS / "latent_R.npy", L)
pd.DataFrame(L, index=FEAT, columns=FEAT).round(4).to_csv(OUTPUTS / "latent_R.csv")
print("\nsaved latent_R.npy / latent_R.csv")
