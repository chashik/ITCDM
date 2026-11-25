"""Lab 6 (iteration 2) tests: windows of the panel, causality of the global networks, the matrix profile."""
import numpy as np
import pandas as pd
import stumpy
import torch

from dmfin.panel import gather_windows, window_index
from lab6_rnn.global_models import TCN, TransformerHead, build, predict_rows


def test_windows_do_not_cross_series():
    p = pd.DataFrame({"series": ["a"] * 5 + ["b"] * 4, "x": np.arange(9.0)})
    rows = window_index(p, np.ones(len(p), bool), lookback=3)
    assert list(rows) == [2, 3, 4, 7, 8]                     # the first two rows of each series have no full window
    w = gather_windows(p["x"].to_numpy(), rows, 3)
    assert (w[3] == [5, 6, 7]).all()                          # a window of series b contains only rows of b


def test_forecasts_do_not_use_later_rows():
    torch.manual_seed(0)
    X = torch.randn(60, 9)
    offs = torch.arange(-19, 1)
    for name in ("lstm", "tcn", "transformer"):
        m = build(name, 9, 20).eval()
        p1 = predict_rows(m, lambda r: X[r[:, None] + offs[None, :]], np.array([30]), "cpu")
        X2 = X.clone(); X2[31:] = torch.randn(29, 9)           # change everything after the forecast date
        p2 = predict_rows(m, lambda r: X2[r[:, None] + offs[None, :]], np.array([30]), "cpu")
        assert np.allclose(p1, p2)


def test_tcn_and_transformer_are_causal():
    torch.manual_seed(1)
    x = torch.randn(2, 20, 9)
    x2 = x.clone(); x2[:, -1] += 5.0                           # change only the last day
    tcn = TCN(9).eval()

    def tcn_states(inp):
        s = tcn.inp(inp.transpose(1, 2))
        for b in tcn.blocks:
            s = s + b(s)
        return s
    with torch.no_grad():
        assert torch.allclose(tcn_states(x)[:, :, :-1], tcn_states(x2)[:, :, :-1])
        tr = TransformerHead(9, lookback=20).eval()
        enc = lambda inp: tr.enc(tr.inp(inp) + tr.pos, mask=tr.mask)
        assert torch.allclose(enc(x)[:, :-1], enc(x2)[:, :-1], atol=1e-6)
    x3 = x.clone().requires_grad_(True)                        # the receptive field of the TCN covers the window
    tcn(x3).sum().backward()
    assert x3.grad[:, 0].abs().sum() > 0


def test_matrix_profile_finds_a_planted_motif():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=600))
    shape = np.sin(np.linspace(0, 3 * np.pi, 30)) * 4
    x[100:130] = x[99] + shape; x[400:430] = x[399] + shape
    mp = stumpy.stump(x, 30)
    i = int(np.argmin(mp[:, 0].astype(float)))
    assert min(abs(i - 100), abs(i - 400)) <= 2 and min(abs(int(mp[i, 1]) - 100), abs(int(mp[i, 1]) - 400)) <= 2
