"""Part 1, chapter 2 (mathematical foundations), section M5. The algebra of fuzzy logic on the guide's data.

1. t-norms: minimum (Goedel), product (Goguen), Lukasiewicz max(0, a + b - 1); their order T_L <= T_P <= T_M and the
   residuated implications I(a, b) = sup{c : T(a, c) <= b}; the residuation law T(a, c) <= b <=> c <= I(a, b) and the
   MV-algebra identity of Lukasiewicz logic, checked on a grid (Stanford Encyclopedia of Philosophy, "Fuzzy logic").
2. Laws of Boolean algebra that fail: for the memberships of Lab 5 (VIX "high", every trading day) the contradiction
   A AND NOT A = min(mu, 1 - mu) is not 0 and the excluded middle A OR NOT A = max(mu, 1 - mu) is not 1; with the
   Lukasiewicz conjunction and disjunction both laws hold, but A AND A = A (idempotence) fails.
3. The choice of t-norm matters: the Mamdani stress index of Lab 5 with AND = minimum, product or Lukasiewicz,
   ranked against high future volatility (AUC on the test period, as in the lab).
4. Monotone quantifiers (Yager 1982, p. 79-80): if Q1 <= Q2 pointwise, a summary with Q1 is never truer than with
   Q2 — checked on all linguistic summaries of Lab 10 ("almost all" <= "most").
Outputs: paper/generated/math5_numbers.tex.  Run (from code/):  python -m math_foundations.m5_fuzzy_algebra
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from dmfin import config as C
from dmfin.reporting import save_numbers

TNORMS = {"minimum": np.minimum, "product": np.multiply, "lukasiewicz": lambda a, b: np.maximum(0.0, a + b - 1)}
IMPL = {"minimum": lambda a, b: np.where(a <= b, 1.0, b),
        "product": lambda a, b: np.where(a <= b, 1.0, b / np.where(a > 0, a, 1)),
        "lukasiewicz": lambda a, b: np.minimum(1.0, 1 - a + b)}


def grid_checks(n: int = 101) -> dict:
    g = np.linspace(0, 1, n)
    a, b, c = np.meshgrid(g, g, g, indexing="ij")
    T = {k: f(a, b) for k, f in TNORMS.items()}
    out = {"order_holds": int(np.all(T["lukasiewicz"] <= T["product"] + 1e-12) and np.all(T["product"] <= T["minimum"] + 1e-12))}
    for k, f in TNORMS.items():
        lhs = f(a, c) <= b + 1e-12                           # T(a, c) <= b
        rhs = c <= IMPL[k](a, b) + 1e-12                     # c <= I(a, b)
        out[f"residuation_{k}"] = int(np.all(lhs == rhs))
        out[f"assoc_{k}"] = float(np.abs(f(f(a, b), c) - f(a, f(b, c))).max())
    oplus = lambda x, y: np.minimum(1.0, x + y)              # MV-algebra: x (+) y, not x = 1 - x
    x, y = a[:, :, 0], b[:, :, 0]
    out["mv_identity_err"] = float(np.abs(oplus(1 - oplus(1 - x, y), y) - oplus(1 - oplus(1 - y, x), x)).max())
    return out


def mamdani_tnorm(fis, data: dict, tnorm) -> np.ndarray:
    """The Mamdani system of Lab 5 with another AND for the antecedents (implication min, aggregation max)."""
    mu = {v: fis.inputs[v](np.asarray(data[v], dtype=float)) for v in fis.inputs}
    agg = np.zeros((len(next(iter(data.values()))), len(fis.universe)))
    for ante, cons in fis.rules:
        s = None
        for v, setname in ante.items():
            s = mu[v][setname] if s is None else tnorm(s, mu[v][setname])
        agg = np.maximum(agg, np.minimum(np.atleast_1d(s)[:, None], fis.out_mf[cons][None]))
    den = agg.sum(1)
    return np.where(den > 0, (agg * fis.universe[None]).sum(1) / np.where(den > 0, den, 1), np.nan)


def main():
    num = grid_checks()
    # 2-3. memberships and the Mamdani stress index of Lab 5
    from dmfin.data import load_market_panel
    from dmfin.features import add_targets, market_features
    from lab5_fuzzy.run import stress_system
    feats = add_targets(market_features(load_market_panel()), horizons=(5, 22))
    d = feats[["vix", "credit_baa_aaa", "ret_22", "y_rv_5", "y_rv_22", "recession"]].dropna()   # the sample of Lab 5
    tr = d.loc[:C.TRAIN_END]
    fis, rules = stress_system(tr)
    mu = fis.inputs["vix"](d["vix"].to_numpy())["high"]
    num.update({"days": len(mu), "partial_share": 100 * float(((mu > 0) & (mu < 1)).mean()),
                "contradiction_min_max": float(np.minimum(mu, 1 - mu).max()),
                "excluded_middle_max_min": float(np.maximum(mu, 1 - mu).min()),
                "contradiction_luk_max": float(np.maximum(0, mu + (1 - mu) - 1).max()),
                "excluded_middle_luk_min": float(np.minimum(1, mu + (1 - mu)).min()),
                "idempotence_luk_maxgap": float(np.abs(np.maximum(0, 2 * mu - 1) - mu).max())})
    te = d.loc[pd.Timestamp(C.VAL_END) + pd.Timedelta(days=1):]
    label = (te["y_rv_5"] > tr["y_rv_5"].quantile(0.8)).astype(int)
    for k, f in TNORMS.items():
        st = mamdani_tnorm(fis, {"vix": te["vix"], "spread": te["credit_baa_aaa"], "mom": te["ret_22"]}, f)
        num[f"auc_{k}"] = roc_auc_score(label, st)
    # 4. monotone quantifiers on the linguistic summaries of Lab 10
    from lab10_economies.paths import RESULTS
    sm = pd.read_csv(RESULTS / "q6_summaries.csv")
    key = ["country", "period", "qualifier W", "summarizer S"]
    piv = sm.pivot_table(index=key, columns="quantifier Q", values="T1 truth", aggfunc="first").fillna(0)
    num.update({"summaries": len(piv), "quant_monotone": int((piv["almost all"] <= piv["most"] + 1e-12).all())})
    save_numbers("math5", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
