"""Lab 6 tests: sequence windows, recurrent nets, autoencoder, error features."""
import numpy as np
import pandas as pd
import pytest
import torch

from lab6_rnn.nets import LSTMAutoencoder, SeqData, SeqNet, diebold_mariano, gpu_available, predict, recon_error, train
from lab6_rnn.run_adaptive import H, N_LAGS, err_features
from lab6_rnn.run_anomaly import episodes
from lab6_rnn.run_motifs import L, znorm_windows

torch.set_num_threads(2)


@pytest.fixture
def toy_frame():
    idx = pd.bdate_range("2012-01-02", "2020-12-31")
    rng = np.random.default_rng(0)
    x = np.sin(np.arange(len(idx)) / 7) + 0.1 * rng.normal(size=len(idx))
    df = pd.DataFrame({"a": x, "b": rng.normal(size=len(idx))}, index=idx)
    df["y"] = df["a"].shift(-5)          # target = value 5 days ahead
    return df.dropna()


def test_seqdata_blocks_are_ordered_and_scaler_uses_train_only(toy_frame):
    sd = SeqData(toy_frame, "y", lookback=20, horizon=5, inputs=["a", "b"])
    tr, va, te = sd["train"], sd["val"], sd["test"]
    assert tr.dates.max() < va.dates.min() and va.dates.max() < te.dates.min()
    from dmfin.splits import chrono_split
    train_rows = chrono_split(toy_frame.index, horizon=5).train      # purged training rows
    assert sd.mu["a"] == pytest.approx(toy_frame.loc[train_rows, "a"].mean())
    # purging: the last training label must be >= 5 days before the boundary
    assert (toy_frame.index > tr.dates.max()).sum() - (toy_frame.index > pd.Timestamp("2014-12-31")).sum() >= 5
    assert tr.X.shape[1:] == (20, 2)


def test_lstm_learns_a_predictable_sequence(toy_frame):
    sd = SeqData(toy_frame, "y", lookback=20, horizon=5, inputs=["a", "b"])
    m = SeqNet(2, 16, 1, "lstm")
    m, hist = train(m, sd["train"].X, sd["train"].y, sd["val"].X, sd["val"].y, lr=5e-3, max_epochs=40, patience=8)
    p = predict(m, sd["test"].X)
    naive = np.mean((sd["test"].y - sd["test"].y.mean()) ** 2)
    assert np.mean((p - sd["test"].y) ** 2) < 0.2 * naive
    assert hist[:, 1].min() <= hist[0, 1]


@pytest.mark.parametrize("cell", ["rnn", "gru"])
def test_other_cells_forward_shape(cell):
    m = SeqNet(3, 8, 2, cell, dropout=0.1)
    assert m(torch.zeros(5, 10, 3)).shape == (5,)


@pytest.mark.skipif(not gpu_available(), reason="no CUDA/ROCm device or DMFIN_DEVICE=cpu")
def test_gpu_training_matches_cpu_training():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 12, 2)).astype(np.float32); y = X[:, -3:, 0].sum(1).astype(np.float32)
    out = {}
    for dev in ("cpu", "cuda"):
        torch.manual_seed(0)
        m, hist = train(SeqNet(2, 8, 2, "gru"), X[:250], y[:250], X[250:], y[250:],
                        max_epochs=3, seed=0, device=dev)
        out[dev] = (hist, predict(m, X[250:]))
    assert np.allclose(out["cpu"][0], out["cuda"][0], atol=1e-4)
    assert np.allclose(out["cpu"][1], out["cuda"][1], atol=1e-3)


def test_autoencoder_flags_unusual_window():
    rng = np.random.default_rng(1)
    X = (0.3 * rng.normal(size=(600, 15, 2))).astype(np.float32)
    ae = LSTMAutoencoder(2, hidden=16, latent=4)
    ae, _ = train(ae, X[:500], None, X[500:], None, loss="recon", lr=5e-3, max_epochs=30, patience=5)
    odd = X[:1].copy(); odd[0, 7] = 6.0
    assert recon_error(ae, odd)[0] > np.quantile(recon_error(ae, X[500:]), 0.99)


def test_error_features_use_only_realised_residuals():
    n = 200
    e = np.random.default_rng(2).normal(size=n); f = np.zeros(n); V = np.zeros((n, 8))
    inputs = ["ret", "log_pk_d", "log_vix", "d_log_vix", "term_10y3m", "credit_baa_aaa", "volume_z", "d_dgs10"]
    t = 150
    z1 = err_features([t], e, f, V, inputs)
    e2 = e.copy(); e2[t - H + 1:] = 99.0        # residuals not yet realised at time t
    z2 = err_features([t], e2, f, V, inputs)
    assert np.allclose(z1, z2)
    assert z1[0, 0] == pytest.approx(e[t - H]) and z1.shape[1] == N_LAGS + 2 + 6


def test_diebold_mariano_sign():
    rng = np.random.default_rng(3)
    y = rng.normal(size=2000)
    good, bad = y + 0.1 * rng.normal(size=2000), y + 1.0 * rng.normal(size=2000)
    s, p = diebold_mariano(y - good, y - bad, h=1)
    assert s < 0 and p < 1e-6


def test_episodes_and_znorm():
    d = pd.to_datetime(["2020-03-02", "2020-03-03", "2020-03-20", "2020-06-01"])
    assert len(episodes(pd.DatetimeIndex(d), gap_days=10)) == 3
    assert len(episodes(pd.DatetimeIndex(d), gap_days=20)) == 2
    W = znorm_windows(np.arange(100.0), np.array([0, 10]))
    assert W.shape == (2, L) and np.allclose(W.mean(1), 0) and np.allclose(W.std(1), 1, atol=1e-6)
