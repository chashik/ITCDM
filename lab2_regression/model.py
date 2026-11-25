"""Lab 2. Polynomial regression on multiple features.

Model family:   scaler -> [PCA] -> PolynomialFeatures(d) -> OLS | Ridge
All transformers are fitted inside the training folds only (Pipeline),
which is the practical guarantee against data leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from dmfin.splits import expanding_window_cv

FEATURES = ["log_pk_d", "log_pk_w", "log_pk_m", "log_vix", "ret_22",
            "term_10y3m", "credit_baa_aaa"]
TARGET = "y_logrv_22"
HORIZON = 22


def design(feats: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=feats.index)
    for c in ("pk_d", "pk_w", "pk_m"):
        d[f"log_{c}"] = np.log(feats[c].clip(lower=1e-4))
    d["log_vix"] = np.log(feats["vix"])
    d["ret_22"] = feats["ret_22"]
    d["term_10y3m"] = feats["term_10y3m"]
    d["credit_baa_aaa"] = feats["credit_baa_aaa"]
    d["log_rv_22"] = np.log(feats["rv_22"].clip(lower=1e-4))  # for the naive forecast
    d[TARGET] = feats[TARGET]
    return d.dropna()


class ClipToTrainingRange(BaseEstimator, TransformerMixin):
    """Clip every input to the [min, max] range seen during fitting.

    A polynomial grows without bound outside the training range; clipping
    turns a dangerous extrapolation into a constant one.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = X.min(axis=0), X.max(axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_model(degree: int = 1, alpha: float | None = None, n_pca: int | None = None,
               scale: bool = True, clip: bool = False) -> Pipeline:
    steps = []
    if clip:
        steps.append(("clip", ClipToTrainingRange()))
    if scale:
        steps.append(("scale", StandardScaler()))
    if n_pca:
        steps.append(("pca", PCA(n_components=n_pca)))
    steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    if degree > 1:
        steps.append(("scale2", StandardScaler()))  # equalise the scale of monomials
    steps.append(("reg", LinearRegression() if alpha is None else Ridge(alpha=alpha)))
    return Pipeline(steps)


def n_terms(p: int, degree: int) -> int:
    from math import comb
    return comb(p + degree, degree) - 1


def cv_score(model_factory, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
             gap: int = HORIZON):
    """Expanding-window CV with an embargo; returns per-fold train/val RMSE."""
    tr_err, va_err = [], []
    for tr, va in expanding_window_cv(len(y), n_splits=n_splits, gap=gap):
        m = model_factory().fit(X[tr], y[tr])
        tr_err.append(np.sqrt(mean_squared_error(y[tr], m.predict(X[tr]))))
        va_err.append(np.sqrt(mean_squared_error(y[va], m.predict(X[va]))))
    return np.array(tr_err), np.array(va_err)


def metrics(y, yhat) -> dict:
    return {"RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
            "MAE": float(mean_absolute_error(y, yhat)),
            "R2": float(r2_score(y, yhat))}
