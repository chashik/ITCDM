"""Lab 4 tests: properties of trees, importances, SHAP and the purged many-series split."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor


def test_regression_tree_is_constant_outside_the_training_range():
    x = np.linspace(0, 10, 200)[:, None]
    tree = DecisionTreeRegressor(max_depth=4).fit(x, x[:, 0] ** 2)
    edge = tree.predict([[10.0]])[0]
    assert tree.predict([[20.0]])[0] == edge and tree.predict([[1000.0]])[0] == edge


def test_impurity_importance_favours_a_high_cardinality_noise_feature():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"signal": rng.integers(0, 3, n).astype(float), "noise_id": rng.permutation(n).astype(float)})
    y = (X["signal"] + rng.normal(0, 1.0, n) > 1).astype(int)
    tr, te = slice(0, 2000), slice(2000, None)
    rf = RandomForestClassifier(100, min_samples_leaf=1, random_state=0).fit(X[tr], y[tr])
    imp = dict(zip(X.columns, rf.feature_importances_))
    perm = permutation_importance(rf, X[te], y[te], n_repeats=5, random_state=0, scoring="roc_auc")
    pimp = dict(zip(X.columns, perm.importances_mean))
    assert imp["noise_id"] > 0.2                    # impurity importance rewards many split points
    assert abs(pimp["noise_id"]) < 0.02 < pimp["signal"]


def test_tree_shap_values_add_up_to_the_log_odds():
    shap = pytest.importorskip("shap")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 3)); y = (X[:, 0] - X[:, 1] + rng.normal(size=800) > 0).astype(int)
    gb = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(X, y)
    ex = shap.TreeExplainer(gb)
    sv = ex.shap_values(X[:20])
    p = gb.predict_proba(X[:20])[:, 1]
    assert np.allclose(np.ravel(ex.expected_value)[0] + sv.sum(1), np.log(p / (1 - p)), atol=1e-5)


def test_many_series_split_purges_the_last_h_days_of_each_block():
    from dmfin import config as C
    from dmfin.panel import split_masks
    dates = pd.bdate_range("2014-11-01", "2015-02-27")
    p = pd.concat([pd.DataFrame({"series": s, "date": dates, "y_5": 1.0}) for s in ("a", "b")], ignore_index=True)
    m = split_masks(p, h=5)
    for s in ("a", "b"):
        tr = p[(p["series"] == s) & m["train"]]["date"]
        all_tr = p[(p["series"] == s) & (p["date"] <= pd.Timestamp(C.TRAIN_END))]["date"]
        assert len(all_tr) - len(tr) == 5 and tr.max() < all_tr.max()
