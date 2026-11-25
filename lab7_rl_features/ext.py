"""Lab 7 (iteration 2). A wider volatile stream (50 features, 11 groups) and further agents.

* ``new_fields``: 30 features added to the 20 of Lab 7 - the volatility term structure,
  stress and uncertainty indices, other markets, their volatilities and factor returns.
  Late starts are real (VIX 9-day 2011, VIX 3-month 2007, Euro Stoxx and HYG 2007, ...);
  weekly indices enter with a publication lag.  ``extend_documents`` adds them to the
  documents of the Lab 7 store (with outages), so the 20 old features are unchanged.
* ``build_tables``: for every decision point and every subset of groups the RMSE of the
  rolling ridge forecast (supervised criterion), the inner-validation RMSE, the share of the
  variance of all available features the subset fails to reconstruct (unlabeled criterion)
  and a rolling lasso; computed in parallel, for any dictionary of groups.
* Agents: tabular Q-learning and SARSA, a LinUCB contextual bandit and a deep Q-network.
  The reward of a decision is known only after the labels of its 21 days are realized,
  i.e. after the next decision; every agent therefore learns from it one step later.
"""
from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Lasso
from torch import nn

from dmfin.data import PUBLICATION_LAG_DAYS, available_at, download_fred, download_yahoo
from dmfin.data_ext import load_ff_factors, load_multi_asset

from .stream import GROUPS as GROUPS6
from .stream import OUTAGE_P, MarketStore, normalise

GROUPS_NEW = {
    "termvol": ["log_vix9d", "log_vix3m", "vix_slope", "log_vvix", "skew"],
    "stress": ["stlfsi", "nfci", "log_epu_22"],
    "cross": ["ndx_ret_5", "rut_ret_5", "sx5e_ret_5", "n225_ret_5", "gold_ret_22", "tlt_ret_22", "hyg_ret_22",
              "eem_ret_22", "dxy_ret_22"],
    "crossvol": ["ndx_lrv_22", "rut_lrv_22", "gold_lrv_22", "oil_lrv_22", "tlt_lrv_22", "log_ovx", "log_gvz"],
    "factors": ["mkt_22", "smb_22", "hml_22", "rmw_22", "cma_22", "mom_22"],
}
GROUPS11 = {**GROUPS6, **GROUPS_NEW}
STEP, WINDOW, H = 21, 750, 5
LASSO_ALPHAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def features(groups: dict) -> list[str]:
    return [f for g in groups.values() for f in g]


# ------------------------------------------------------------------ data
def _align(s: pd.Series, idx: pd.DatetimeIndex, lag_days: int = 0, tol_days: int = 10) -> pd.Series:
    """Last value dated at least ``lag_days`` calendar days before each day of ``idx`` (at most ``tol_days`` older)."""
    return available_at(s, idx, lag_days, tolerance_days=tol_days)


def new_fields(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """The 30 new features on the S&P 500 trading days ``idx``, each known at the close of its day.

    Weekly stress indices are used 7 days after their reference date and the daily news-based
    uncertainty index 1 day after it (publication delays).  Foreign closes precede the US close.
    """
    d = pd.DataFrame(index=idx)
    log_vix = np.log(_align(download_yahoo("^VIX")["close"], idx))
    d["log_vix9d"] = np.log(_align(download_yahoo("^VIX9D")["close"], idx))
    d["log_vix3m"] = np.log(_align(download_fred("VXVCLS"), idx))
    d["vix_slope"] = d["log_vix3m"] - log_vix
    d["log_vvix"] = np.log(_align(download_yahoo("^VVIX")["close"], idx))
    d["skew"] = _align(download_yahoo("^SKEW")["close"], idx)
    d["stlfsi"] = _align(download_fred("STLFSI4"), idx, lag_days=PUBLICATION_LAG_DAYS["STLFSI4"])
    d["nfci"] = _align(download_fred("NFCI"), idx, lag_days=PUBLICATION_LAG_DAYS["NFCI"])
    d["log_epu_22"] = np.log(_align(download_fred("USEPUINDXD"), idx,
                                    lag_days=PUBLICATION_LAG_DAYS["USEPUINDXD"]).rolling(22).mean())
    ma = load_multi_asset()
    r = {k: np.log(_align(v["adj_close"], idx)).diff() for k, v in ma.items()}
    for k, name in (("^IXIC", "ndx"), ("^RUT", "rut"), ("^STOXX50E", "sx5e"), ("^N225", "n225")):
        d[f"{name}_ret_5"] = r[k].rolling(5).sum()
    for k, name in (("GC=F", "gold"), ("TLT", "tlt"), ("HYG", "hyg"), ("EEM", "eem"), ("DX-Y.NYB", "dxy")):
        d[f"{name}_ret_22"] = r[k].rolling(22).sum()
    for k, name in (("^IXIC", "ndx"), ("^RUT", "rut"), ("GC=F", "gold"), ("CL=F", "oil"), ("TLT", "tlt")):
        d[f"{name}_lrv_22"] = np.log(np.sqrt(252 * (r[k] ** 2).rolling(22).mean()) + 1e-4)
    d["log_ovx"] = np.log(_align(download_fred("OVXCLS"), idx))
    d["log_gvz"] = np.log(_align(download_fred("GVZCLS"), idx))
    ff = load_ff_factors()
    for c, name in (("Mkt-RF", "mkt"), ("SMB", "smb"), ("HML", "hml"), ("RMW", "rmw"), ("CMA", "cma"), ("Mom", "mom")):
        d[f"{name}_22"] = np.log1p(_align(ff[c], idx, tol_days=4)).rolling(22).sum()
    return d[features(GROUPS_NEW)]


def extend_documents(docs: list[dict], new: pd.DataFrame, seed: int) -> list[dict]:
    """Add the new fields to every document; each value is lost with probability OUTAGE_P."""
    rng = np.random.default_rng(seed)
    out = []
    for doc in docs:
        doc = copy.deepcopy(doc)
        t = pd.Timestamp(doc["date"])
        if t in new.index:
            for k, v in new.loc[t].items():
                if pd.notna(v) and rng.uniform() >= OUTAGE_P:
                    doc["fields"][k] = float(v)
        out.append(doc)
    return out


def read_frame(store: MarketStore, cols: list[str]) -> pd.DataFrame:
    """All documents through the ingestion layer, as one frame."""
    recs = {r["date"]: {**normalise(r["fields"]), "y": r.get("label", np.nan)} for r in store.t.all()}
    df = pd.DataFrame.from_dict(recs, orient="index")
    df.index = pd.to_datetime(df.index)
    return df.sort_index().reindex(columns=cols + ["y"])


# ------------------------------------------------------------------ tables
@dataclass
class Tables:
    groups: list[str]
    dates: pd.DatetimeIndex
    rmse: np.ndarray        # (steps, 2^G) test RMSE over the next STEP days of the ridge on each subset
    inner: np.ndarray       # (steps, 2^G) RMSE on the last 20 % of the training window (inner validation)
    recon: np.ndarray       # (steps, 2^G) unexplained share of the variance of all available features
    lasso: np.ndarray       # (steps,) rolling lasso on all available features
    avail: np.ndarray
    vix: np.ndarray
    vix_chg: np.ndarray
    stress: np.ndarray
    nobs: np.ndarray
    ends: np.ndarray | None = None   # date on which the last label used by the decision becomes known

    @property
    def ng(self) -> int:
        return len(self.groups)


def mask_features(mask: int, groups: dict) -> list[str]:
    return [f for i, g in enumerate(groups) if mask >> i & 1 for f in groups[g]]


def _ridge(Xtr, ytr, Xev, alpha=1.0):
    """Ridge regression on standardized inputs with an intercept (as sklearn ``Ridge``)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z, Ze = (Xtr - mu) / sd, (Xev - mu) / sd
    ym = ytr.mean(0)
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (ytr - ym))
    return Ze @ w + ym


_W = {}


def _init(frame, groups, stress):
    _W.update(frame=frame, groups=groups, stress=stress)


def _step(p: int):
    frame, groups = _W["frame"], _W["groups"]
    G, ALL = list(groups), (1 << len(groups)) - 1
    dates = frame.index
    s = dates[p]
    w = frame.iloc[max(0, p - WINDOW - 30):min(len(frame), p + STEP + 1)]
    X = w.drop(columns="y").ffill(limit=5)
    past = X.loc[:s]
    lab = w["y"].loc[:s].iloc[:-H]                       # labels realised by time s
    trn = lab.index[-WINDOW:]
    ev = X.index[X.index > s][:STEP]
    ev = ev[w.loc[ev, "y"].notna()]
    avail = 0
    for i, g in enumerate(G):
        if len(ev) and past.loc[trn, groups[g]].notna().mean().min() >= 0.8 and X.loc[ev, groups[g]].notna().all().all():
            avail |= 1 << i
    r, inner, u = (np.full(ALL + 1, np.nan) for _ in range(3))
    lr = np.nan
    if len(ev) and avail:
        ytr, yev = lab.loc[trn].to_numpy(), w.loc[ev, "y"].to_numpy()
        cols = mask_features(avail, groups)
        Xt = past.loc[trn, cols].fillna(past.loc[trn, cols].mean()).to_numpy()
        Xe = X.loc[ev, cols].to_numpy()
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        Zt, Ze = (Xt - mu) / sd, (Xe - mu) / sd
        cut = int(0.8 * len(trn))
        pos = {c: j for j, c in enumerate(cols)}
        for m in range(1, ALL + 1):
            if m & ~avail:
                continue
            idx = [pos[c] for c in mask_features(m, groups)]
            r[m] = np.sqrt(np.mean((_ridge(Xt[:, idx], ytr, Xe[:, idx]) - yev) ** 2))
            pi = _ridge(Xt[:cut - H, idx], ytr[:cut - H], Xt[cut:, idx])
            inner[m] = np.sqrt(np.mean((pi - ytr[cut:]) ** 2))
            rec = _ridge(Zt[:, idx], Zt, Ze[:, idx])        # no labels: reconstruct all available features
            u[m] = float(np.mean((rec - Ze) ** 2) / np.mean(Ze ** 2))
        best = min(LASSO_ALPHAS, key=lambda a: np.mean(
            (Lasso(alpha=a, max_iter=5000).fit(Zt[:cut - H], ytr[:cut - H]).predict(Zt[cut:]) - ytr[cut:]) ** 2))
        lr = float(np.sqrt(np.mean((Lasso(alpha=best, max_iter=5000).fit(Zt, ytr).predict(Ze) - yev) ** 2)))
    lv = past["log_vix"].dropna()
    vix_now, vix_old = float(np.exp(lv.iloc[-1])), float(np.exp(lv.iloc[-22]))
    st = _W["stress"].loc[:s].dropna()
    end = dates[min(p + STEP + H, len(dates) - 1)]
    return (s, r, inner, u, lr, avail, vix_now, np.log(vix_now / vix_old), float(st.iloc[-1]) if len(st) else 0.0, len(ev),
            end)


def build_tables(frame: pd.DataFrame, groups: dict, first: int, last: int, stress: pd.Series,
                 workers: int = 12) -> Tables:
    """``stress``: a stress index known at each date (a context variable of the agents)."""
    frame = frame[features(groups) + ["y"]]
    pos = list(range(first, last, STEP))
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(frame, groups, stress)) as ex:
        rows = list(ex.map(_step, pos, chunksize=2))
    col = lambda j: np.array([x[j] for x in rows])
    return Tables(list(groups), pd.DatetimeIndex(col(0)), col(1), col(2), col(3), col(4), col(5).astype(int),
                  col(6), col(7), col(8), col(9), pd.DatetimeIndex(col(10)))


# ------------------------------------------------------------------ environment
def regime(vix: float, chg: float) -> int:
    level = 0 if vix < 16 else (1 if vix < 25 else 2)
    return 2 * level + int(chg > 0)


def popcount(m: int) -> int:
    return bin(int(m)).count("1")


class Env:
    """The environment of Lab 7 for any number of groups.

    ``table="rmse"``: the reward compares the forecast errors of the chosen subset and of all
    available groups (supervised); ``table="recon"``: it compares their reconstruction errors
    of the available features (no labels are used).  The evaluation error returned by
    ``step`` is always the forecast RMSE of the chosen subset.
    """
    LAMBDA, PSI, SCALE = 0.05, 0.10, 100.0

    def __init__(self, T: Tables, steps, table: str = "rmse", lam: float | None = None, relative: bool = True):
        self.T, self.steps, self.E, self.ng = T, np.asarray(steps), getattr(T, table), T.ng
        self.lam = self.LAMBDA if lam is None else lam
        self.relative = relative                         # False: the group cost of Lab 7, lambda * |S|
        self.dim = 4 + 2 * self.ng

    def reset(self, mask: int | None = None):
        """Start from ``mask`` (restricted to the available groups) or from all available groups."""
        self.k = 0
        av = int(self.T.avail[self.steps[0]])
        self.mask = (int(mask) & av or av) if mask is not None else av
        return self.state()

    def random_mask(self, rng) -> int:
        """A random non-empty subset of the groups available at the first step (exploring starts)."""
        av = [i for i in range(self.ng) if int(self.T.avail[self.steps[0]]) >> i & 1]
        m = sum(1 << i for i in av if rng.uniform() < 0.5)
        return m or 1 << int(rng.choice(av))

    def _k(self):
        return self.steps[min(self.k, len(self.steps) - 1)]

    def state(self):
        k = self._k()
        return regime(self.T.vix[k], self.T.vix_chg[k]), self.mask

    def context(self) -> np.ndarray:
        """Numeric state for function approximation: market context, current subset, availability."""
        k = self._k()
        av = int(self.T.avail[k])
        return np.array([1.0, np.log(self.T.vix[k]) - 3.0, self.T.vix_chg[k], self.T.stress[k] / 2.0]
                        + [(self.mask >> i) & 1 for i in range(self.ng)] + [(av >> i) & 1 for i in range(self.ng)],
                        np.float32)

    def valid_actions(self) -> list[int]:
        av = int(self.T.avail[self._k()])
        mask = self.mask & av or av
        acts = [self.ng]                                     # the last action keeps the subset
        for i in range(self.ng):
            m = mask ^ (1 << i)
            if m and not (m & ~av):
                acts.append(i)
        return acts

    def step(self, a: int):
        k = self.steps[self.k]
        av = int(self.T.avail[k])
        self.mask &= av
        if not self.mask:
            self.mask = av
        switched = a != self.ng
        if switched:
            m = self.mask ^ (1 << a)
            if m and not (m & ~av):
                self.mask = m
        # errors and group costs are both counted relative to the model with all available groups, so that
        # keeping all groups earns zero; the shift does not depend on the action and leaves the optimal policy unchanged
        r = (self.SCALE * (self.E[k, av] - self.E[k, self.mask])
             - self.lam * (popcount(self.mask) - (popcount(av) if self.relative else 0)) - self.PSI * switched)
        self.k += 1
        return self.state(), float(r), self.k >= len(self.steps), float(self.T.rmse[k, self.mask])


def eps_schedule(ep: int, episodes: int) -> float:
    return max(0.05, 0.5 * (1 - ep / (0.8 * episodes)))


# ------------------------------------------------------------------ agents
class Tabular:
    """Tabular Q-learning (off-policy) or SARSA (on-policy) with epsilon-greedy exploration."""

    def __init__(self, n_actions, alpha=0.1, gamma=0.9, eps=0.1, sarsa=False, seed=0):
        self.na, self.alpha, self.gamma, self.eps, self.sarsa = n_actions, alpha, gamma, eps, sarsa
        self.Q, self.rng = {}, np.random.default_rng(seed)

    def q(self, s):
        if s not in self.Q:
            self.Q[s] = np.zeros(self.na)
        return self.Q[s]

    def act(self, env, s, greedy=False):
        valid = env.valid_actions()
        if not greedy and self.rng.uniform() < self.eps:
            return int(self.rng.choice(valid))
        qs = self.q(s)
        best = max(qs[a] for a in valid)
        return int(self.rng.choice([a for a in valid if qs[a] == best]))

    def _update(self, tr):
        s, a, r, s2, a2, done = tr
        nxt = 0.0 if done else (self.q(s2)[a2] if self.sarsa else self.q(s2).max())
        self.q(s)[a] += self.alpha * (r + self.gamma * nxt - self.q(s)[a])

    def episode(self, env, learn=True, greedy=False, mask=None):
        s = env.reset(mask)
        a = self.act(env, s, greedy)
        errs, masks, rewards, pending, done = [], [], [], None, False
        while not done:
            s2, r, done, e = env.step(a)
            errs.append(e); masks.append(env.mask); rewards.append(r)
            if learn and pending is not None:
                self._update(pending)                        # the reward of the previous step is now known
            a2 = None if done else self.act(env, s2, greedy)
            if learn:
                pending = (s, a, r, s2, a2, done)
            s, a = s2, a2
        if learn and pending is not None:
            self._update(pending)
        return np.array(errs), np.array(masks), np.array(rewards)


class LinUCB:
    """Disjoint LinUCB (Li et al. 2010): a ridge model of the reward of each arm and an optimism bonus.

    A bandit maximizes the immediate reward only (no bootstrapping over future states).
    """

    def __init__(self, n_actions, d, alpha=1.0, seed=0):
        self.Ainv = np.stack([np.eye(d) for _ in range(n_actions)])
        self.b = np.zeros((n_actions, d))
        self.alpha, self.eps = alpha, 0.0

    def _update(self, tr):
        x, a, r = tr
        Ax = self.Ainv[a] @ x                                # Sherman-Morrison update of the inverse
        self.Ainv[a] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[a] += r * x

    def episode(self, env, learn=True, greedy=False, mask=None):
        env.reset(mask)
        errs, masks, rewards, pending, done = [], [], [], None, False
        while not done:
            x = env.context().astype(float)
            valid = env.valid_actions()
            th = np.einsum("aij,aj->ai", self.Ainv[valid], self.b[valid])
            bonus = np.sqrt(np.einsum("i,aij,j->a", x, self.Ainv[valid], x))
            a = valid[int(np.argmax(th @ x + (0.0 if greedy else self.alpha) * bonus))]
            _, r, done, e = env.step(a)
            errs.append(e); masks.append(env.mask); rewards.append(r)
            if learn:
                if pending is not None:
                    self._update(pending)
                pending = (x, a, r)
        if learn and pending is not None:
            self._update(pending)
        return np.array(errs), np.array(masks), np.array(rewards)


class DQN:
    """Deep Q-network: an MLP Q-function, experience replay and a target network (Mnih et al. 2013)."""

    def __init__(self, n_actions, d, gamma=0.9, lr=1e-3, eps=0.1, seed=0, hidden=64, buffer=5000, batch=64, sync=200):
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))
        self.tgt = copy.deepcopy(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.na, self.gamma, self.eps, self.batch, self.sync = n_actions, gamma, eps, batch, sync
        self.rng = np.random.default_rng(seed)
        self.X = np.zeros((buffer, d), np.float32); self.X2 = np.zeros((buffer, d), np.float32)
        self.A = np.zeros(buffer, np.int64); self.R = np.zeros(buffer, np.float32)
        self.D = np.zeros(buffer, np.float32); self.M2 = np.zeros((buffer, n_actions), bool)
        self.size, self.ptr, self.n = 0, 0, 0

    def act(self, env, x, greedy=False):
        valid = env.valid_actions()
        if not greedy and self.rng.uniform() < self.eps:
            return int(self.rng.choice(valid))
        with torch.no_grad():
            q = self.net(torch.as_tensor(x[None]))[0].numpy()
        return int(max(valid, key=lambda a: q[a]))

    def _store(self, tr):
        i = self.ptr
        self.X[i], self.A[i], self.R[i], self.X2[i], self.D[i], self.M2[i] = tr
        self.ptr = (i + 1) % len(self.A); self.size = min(self.size + 1, len(self.A))

    def _learn(self):
        if self.size < self.batch:
            return
        i = self.rng.integers(0, self.size, self.batch)
        with torch.no_grad():
            q2 = self.tgt(torch.as_tensor(self.X2[i])).numpy()
        q2 = np.where(self.M2[i], q2, -np.inf).max(1)
        q2[self.D[i] == 1] = 0.0
        y = torch.as_tensor(self.R[i] + self.gamma * q2, dtype=torch.float32)
        q = self.net(torch.as_tensor(self.X[i]))[torch.arange(self.batch), torch.as_tensor(self.A[i])]
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.n += 1
        if self.n % self.sync == 0:
            self.tgt.load_state_dict(self.net.state_dict())

    def episode(self, env, learn=True, greedy=False, mask=None):
        env.reset(mask)
        x = env.context()
        errs, masks, rewards, pending, done = [], [], [], None, False
        while not done:
            a = self.act(env, x, greedy)
            _, r, done, e = env.step(a)
            errs.append(e); masks.append(env.mask); rewards.append(r)
            x2 = env.context()
            m2 = np.zeros(self.na, bool)
            if not done:
                m2[env.valid_actions()] = True
            if learn:
                if pending is not None:
                    self._store(pending)
                    self._learn()
                pending = (x, a, r, x2, float(done), m2)
            x = x2
        if learn and pending is not None:
            self._store(pending)
            self._learn()
        return np.array(errs), np.array(masks), np.array(rewards)


def pooled(errs, nobs) -> float:
    return float(np.sqrt(np.sum(nobs * np.asarray(errs) ** 2) / np.sum(nobs)))
