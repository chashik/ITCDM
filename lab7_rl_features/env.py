"""Lab 7. Feature-selection environment and Q-learning agent.

Decision points are every STEP trading days.  At decision point s the
agent sees the market regime and the currently active feature groups and
may switch one group on or off.  A ridge model is then fitted on the last
WINDOW labelled days available in the document store (labels of the last
H days are not yet known) and forecasts the next STEP days.  The reward
is the improvement in RMSE over the model that uses all available groups,
minus a cost per active group and a cost per switch.

Because the ridge fits do not depend on the agent, the RMSE of every
subset at every decision point is pre-computed once (``build_tables``);
learning then runs on these tables, which makes thousands of episodes
cheap.  The forecasts themselves are computed exactly as they would be
online - from data read back from the store, without look-ahead.

The reward of a decision depends on the labels of its STEP forecast days,
which are known only H days after the last of them, i.e. after the next
decision; the agent therefore learns from it one decision later
(``run_episode``).  For the same reason a decision point belongs to the
training or validation block only if all labels it uses lie inside the block
(``purged_blocks``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge

from dmfin import config as C

from .stream import GROUPS, MarketStore

G = list(GROUPS)
NG = len(G)
ALL = (1 << NG) - 1
STEP, WINDOW, H = 21, 750, 5
KEEP = NG                              # action index "keep the subset"


def mask_features(mask: int) -> list[str]:
    return [f for i, g in enumerate(G) if mask >> i & 1 for f in GROUPS[g]]


def popcount(m: int) -> int:
    return bin(m).count("1")


@dataclass
class Tables:
    dates: pd.DatetimeIndex          # decision dates
    rmse: np.ndarray                 # (steps, 2^NG) test-window RMSE; nan = unavailable / empty
    inner: np.ndarray                # (steps, 2^NG) inner-validation RMSE inside the window
    avail: np.ndarray                # (steps,) mask of available groups
    lasso: np.ndarray                # (steps,) RMSE of rolling lasso
    vix: np.ndarray                  # VIX level at decision time
    vix_chg: np.ndarray              # 21-day log change of VIX
    nobs: np.ndarray                 # (steps,) evaluation days
    ends: np.ndarray | None = None   # (steps,) date on which the last label used by the decision becomes known


def _fit_predict(Xtr, ytr, Xev, alpha=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = Ridge(alpha=alpha).fit((Xtr - mu) / sd, ytr)
    return m.predict((Xev - mu) / sd)


def build_tables(store: MarketStore, all_dates: pd.DatetimeIndex, first: int, last: int) -> Tables:
    rows = []
    pos = np.arange(first, last, STEP)
    lasso_alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    for p in pos:
        s = all_dates[p]
        lo = all_dates[max(0, p - WINDOW - 30)]
        hi = all_dates[min(len(all_dates) - 1, p + STEP)]
        w = store.window(lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
        X = w.drop(columns="y").ffill(limit=5)
        past = X.loc[:s]
        lab = w["y"].loc[:s].iloc[:-H]                    # labels realised by time s
        trn = lab.index[-WINDOW:]
        ev = X.index[X.index > s][:STEP] if p + STEP < len(all_dates) else X.index[X.index > s]
        ev = ev[w.loc[ev, "y"].notna()]
        avail = 0
        for i, g in enumerate(G):
            if past.loc[trn, GROUPS[g]].notna().mean().min() >= 0.8 and X.loc[ev, GROUPS[g]].notna().all().all() \
                    if len(ev) else False:
                avail |= 1 << i
        r = np.full(ALL + 1, np.nan); inner = np.full(ALL + 1, np.nan)
        ytr, yev = lab.loc[trn].to_numpy(), w.loc[ev, "y"].to_numpy()
        cut = int(0.8 * len(trn))
        for m in range(1, ALL + 1):
            if m & ~avail or len(ev) == 0:
                continue
            cols = mask_features(m)
            Xt = past.loc[trn, cols].fillna(past.loc[trn, cols].mean()).to_numpy()
            Xe = X.loc[ev, cols].to_numpy()
            r[m] = np.sqrt(np.mean((_fit_predict(Xt, ytr, Xe) - yev) ** 2))
            pi = _fit_predict(Xt[:cut - H], ytr[:cut - H], Xt[cut:])
            inner[m] = np.sqrt(np.mean((pi - ytr[cut:]) ** 2))
        # rolling lasso on all available features, alpha by inner validation
        cols = mask_features(avail)
        Xt = past.loc[trn, cols].fillna(past.loc[trn, cols].mean()).to_numpy()
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        Z = (Xt - mu) / sd
        best = min(lasso_alphas, key=lambda a: np.mean(
            (Lasso(alpha=a, max_iter=5000).fit(Z[:cut - H], ytr[:cut - H]).predict(Z[cut:]) - ytr[cut:]) ** 2))
        lm = Lasso(alpha=best, max_iter=5000).fit(Z, ytr)
        lr = np.sqrt(np.mean((lm.predict((X.loc[ev, cols].to_numpy() - mu) / sd) - yev) ** 2)) if len(ev) else np.nan
        vix_now = np.exp(past["log_vix"].dropna().iloc[-1])
        vix_old = np.exp(past["log_vix"].dropna().iloc[-22])
        end = all_dates[min(p + STEP + H, len(all_dates) - 1)]
        rows.append((s, r, inner, avail, lr, vix_now, np.log(vix_now / vix_old), len(ev), end))
    return Tables(pd.DatetimeIndex([x[0] for x in rows]), np.array([x[1] for x in rows]),
                  np.array([x[2] for x in rows]), np.array([x[3] for x in rows]),
                  np.array([x[4] for x in rows]), np.array([x[5] for x in rows]),
                  np.array([x[6] for x in rows]), np.array([x[7] for x in rows]),
                  pd.DatetimeIndex([x[8] for x in rows]))


def purged_blocks(dates: pd.DatetimeIndex, ends) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decision points of the training, validation and test blocks.

    A point belongs to the training or validation block only if the last label it uses is known by the end of the
    block; the points between are purged (as the targets at the boundaries of the blocks, Lab 0).
    """
    dates, ends = pd.DatetimeIndex(dates), pd.DatetimeIndex(ends)
    tre, vae = pd.Timestamp(C.TRAIN_END), pd.Timestamp(C.VAL_END)
    tr = np.where(ends <= tre)[0]
    va = np.where((dates > tre) & (ends <= vae))[0]
    te = np.where(dates > vae)[0]
    return tr, va, te


def regime(vix: float, chg: float) -> int:
    level = 0 if vix < 16 else (1 if vix < 25 else 2)
    return 2 * level + int(chg > 0)


class FeatureSelectionEnv:
    LAMBDA, PSI, SCALE, INVALID = 0.05, 0.10, 100.0, -1.0

    def __init__(self, T: Tables, steps: np.ndarray):
        self.T, self.steps = T, steps

    def reset(self, mask: int | None = None):
        self.k = 0
        k0 = self.steps[0]
        self.mask = (mask if mask is not None else ALL) & self.T.avail[k0] or self.T.avail[k0]
        return self.state()

    def state(self):
        k = self.steps[min(self.k, len(self.steps) - 1)]
        return regime(self.T.vix[k], self.T.vix_chg[k]), self.mask

    def valid_actions(self):
        k = self.steps[self.k]
        acts = [KEEP]
        for i in range(NG):
            m = self.mask ^ (1 << i)
            if m and not (m & ~self.T.avail[k]):
                acts.append(i)
        return acts

    def step(self, a: int):
        k = self.steps[self.k]
        avail = self.T.avail[k]
        self.mask &= avail
        if not self.mask:
            self.mask = avail
        switched = a != KEEP
        if switched:
            m = self.mask ^ (1 << a)
            if m and not (m & ~avail):
                self.mask = m
        e = self.T.rmse[k, self.mask]
        e_all = self.T.rmse[k, avail]
        r = self.SCALE * (e_all - e) - self.LAMBDA * popcount(self.mask) - self.PSI * switched
        self.k += 1
        done = self.k >= len(self.steps)
        return self.state(), float(r), done, float(e)


class QAgent:
    """Tabular Q-learning (Watkins) with epsilon-greedy exploration.

    Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]
    A constant alpha keeps the agent able to track a non-stationary
    environment (recent experience is weighted more).
    """

    def __init__(self, alpha=0.1, gamma=0.5, eps=0.1, seed=0):
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.Q = {}
        self.rng = np.random.default_rng(seed)

    def q(self, s):
        if s not in self.Q:
            self.Q[s] = np.zeros(NG + 1)
        return self.Q[s]

    def act(self, s, valid, greedy=False):
        if not greedy and self.rng.uniform() < self.eps:
            return int(self.rng.choice(valid))
        qs = self.q(s)
        best = max(qs[a] for a in valid)
        return int(self.rng.choice([a for a in valid if qs[a] == best]))

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * self.q(s2).max()
        self.q(s)[a] += self.alpha * (target - self.q(s)[a])


def run_episode(env, agent, learn=True, greedy=False, mask=None):
    """One pass over the decision points of ``env``.

    The reward of decision k is known only after decision k+1 has been made (the labels of its forecast window
    arrive H days after the window ends); the update for decision k is therefore made after decision k+1.
    """
    s = env.reset(mask)
    errs, masks, rewards = [], [], []
    pending, done = None, False
    while not done:
        a = agent.act(s, env.valid_actions(), greedy=greedy)
        s2, r, done, e = env.step(a)
        if learn and pending is not None:
            agent.update(*pending)                      # the reward of the previous decision is now known
        pending = (s, a, r, s2, done)
        errs.append(e); masks.append(env.mask); rewards.append(r)
        s = s2
    if learn and pending is not None:
        agent.update(*pending)
    return np.array(errs), np.array(masks), np.array(rewards)


def pooled(rmse_steps: np.ndarray, nobs: np.ndarray) -> float:
    return float(np.sqrt(np.sum(nobs * rmse_steps ** 2) / np.sum(nobs)))
