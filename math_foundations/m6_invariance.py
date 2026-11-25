"""Part 1, chapter 2 (mathematical foundations), section M6. Group actions and invariance.

1. A permutation test is exact when, under the null hypothesis, the distribution of the data is invariant under the
   group of transformations used (Hemerik and Goeman 2018).  For a pair of independent autocorrelated series the full
   symmetric group (any permutation of one series) is not such a group — it destroys the autocorrelation — while the
   cyclic group of shifts keeps it (up to the wrap-around).  Simulated rejection rates at the 5 % level for two
   independent AR(1) series of 60 months, and the mutual-information screen of Lab 10 (forecasting stage) with both
   tests.
2. Self-attention without positions is permutation-equivariant: Att(PX) = P Att(X).  Sinusoidal position codes with
   frequencies k/n are characters of the cyclic group Z_n: a shift by s acts on them as a rotation, PE(t + s) =
   R(s) PE(t).
3. Month-of-year means are the projection of a monthly series onto the 12-dimensional space of 12-periodic
   functions, spanned by the characters of Z_12 (Fourier terms, Hyndman and Athanasopoulos); Parseval splits the
   seasonal variance by harmonic.  Belarusian month-on-month inflation (Lab 10).
Outputs: paper/generated/math6_numbers.tex.  Run (from code/):  python -m math_foundations.m6_invariance
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from dmfin.reporting import save_numbers


def ar1(n: int, rho: float, rng) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.normal() / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal()
    return x


def pvalues(x: np.ndarray, y: np.ndarray, rng, n_perm: int = 199, min_shift: int = 6) -> tuple[float, float]:
    """p-values of |corr(x, y)|: plain permutations of y and cyclic shifts of y (shifts min_shift .. n - min_shift)."""
    stat = abs(np.corrcoef(x, y)[0, 1])
    perm = np.array([abs(np.corrcoef(x, rng.permutation(y))[0, 1]) for _ in range(n_perm)])
    shifts = np.array([abs(np.corrcoef(x, np.roll(y, k))[0, 1]) for k in range(min_shift, len(y) - min_shift + 1)])
    return (1 + (perm >= stat).sum()) / (n_perm + 1), (1 + (shifts >= stat).sum()) / (len(shifts) + 1)


def rejection_rates(rho: float, n: int = 60, sims: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rej = np.array([pvalues(ar1(n, rho, rng), ar1(n, rho, rng), rng) for _ in range(sims)]) <= 0.05
    return tuple(rej.mean(0))


def attention(X: np.ndarray, Wq, Wk, Wv) -> np.ndarray:
    S = (X @ Wq) @ (X @ Wk).T / np.sqrt(Wq.shape[1])
    A = np.exp(S - S.max(1, keepdims=True))
    return (A / A.sum(1, keepdims=True)) @ (X @ Wv)


def position_codes(n: int, pairs: int) -> np.ndarray:
    t = np.arange(n)[:, None]
    k = np.arange(1, pairs + 1)[None]
    return np.concatenate([np.cos(2 * np.pi * k * t / n), np.sin(2 * np.pi * k * t / n)], 1)


def main():
    num = {}
    # 1a. simulated size of the two tests (independent AR(1) pairs, n = 60)
    for rho in (0.0, 0.9):
        plain, cyclic = rejection_rates(rho)
        tag = "rho0" if rho == 0 else "rho09"
        num[f"size_plain_{tag}"], num[f"size_cyclic_{tag}"] = 100 * plain, 100 * cyclic
    # 1b. the mutual-information screen of Lab 10 with both tests
    from lab10_economies.common import COUNTRIES
    from lab10_economies.forecasting import SHORT, short_panel, usable
    rng = np.random.default_rng(1)
    sig_plain = sig_cyclic = total = 0
    for c in COUNTRIES:
        d = short_panel(c)
        fs = usable(d, SHORT)
        z = pd.concat([d[fs], (d["y"].shift(-3) - d["y"]).rename("target")], axis=1).dropna()
        mi = mutual_info_regression(z[fs], z["target"], n_neighbors=5, random_state=0)
        tgt = z["target"].to_numpy()
        plain = np.array([mutual_info_regression(z[fs], rng.permutation(tgt), n_neighbors=5, random_state=0)
                          for _ in range(99)])
        cyc = np.array([mutual_info_regression(z[fs], np.roll(tgt, k), n_neighbors=5, random_state=0)
                        for k in range(6, len(tgt) - 5)])
        sig_plain += int(((1 + (plain >= mi).sum(0)) / 100 <= 0.05).sum())
        sig_cyclic += int(((cyc >= mi).mean(0) < 0.05).sum())
        total += len(fs)
    num.update({"mi_features": total, "mi_sig_plain": sig_plain, "mi_sig_cyclic": sig_cyclic})
    # 2. permutation equivariance of attention; shifts act on position codes as rotations
    rng = np.random.default_rng(2)
    n, d = 12, 8
    X = rng.normal(size=(n, d))
    W = [rng.normal(size=(d, d)) for _ in range(3)]
    Pm = np.eye(n)[rng.permutation(n)]
    num["att_equiv_err"] = float(np.abs(attention(Pm @ X, *W) - Pm @ attention(X, *W)).max())
    PE = position_codes(n, d // 2)
    num["att_pe_equiv_err"] = float(np.abs(attention(Pm @ X + PE, *W) - Pm @ attention(X + PE, *W)).max())
    s = 5
    k = np.arange(1, d // 2 + 1)
    c_, s_ = np.cos(2 * np.pi * k * s / n), np.sin(2 * np.pi * k * s / n)
    R = np.block([[np.diag(c_), np.diag(s_)], [-np.diag(s_), np.diag(c_)]])          # rotation by the shift s
    num["pe_rotation_err"] = float(np.abs(np.roll(PE, -s, axis=0) - PE @ R).max())
    # 3. seasonality: month-of-year means = projection onto the characters of Z_12
    from lab10_economies.common import panel
    y = panel("BY")["cpi_mom"].dropna()
    t = np.arange(len(y))
    F = np.column_stack([np.ones(len(y))] + [f(2 * np.pi * h * t / 12) for h in range(1, 7)
                                             for f in (np.cos, np.sin)][:11])
    fit = F @ np.linalg.lstsq(F, y.to_numpy(), rcond=None)[0]
    means = y.groupby(y.index.month).transform("mean").to_numpy()
    num["season_proj_err"] = float(np.abs(fit - means).max())
    prof = y.groupby(y.index.month).mean().to_numpy()
    spec = np.abs(np.fft.rfft(prof - prof.mean())) ** 2
    spec[1:6] *= 2                                           # Parseval: harmonics 1-5 appear twice, 6 once
    share = spec / spec.sum()
    num.update({"season_share_h1": 100 * share[1], "season_share_h2": 100 * share[2],
                "season_top_harmonic": int(np.argmax(share)), "season_months": len(y)})
    save_numbers("math6", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
