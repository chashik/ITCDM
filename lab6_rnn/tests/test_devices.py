"""Choice of the device and agreement of GPU and CPU results (the GPU tests are skipped without a GPU)."""
import numpy as np
import pytest
import torch

from dmfin import device as D
from lab6_rnn.global_models import predict_rows, train_global

GPU = D.gpu_available()          # DMFIN_DEVICE=cpu skips the GPU tests


def test_device_rule(monkeypatch):
    assert D.rnn_device(D.GPU_HIDDEN, gpu=True) == "cuda"
    assert D.rnn_device(D.GPU_HIDDEN // 2, gpu=True) == "cpu"
    assert D.rnn_device(4 * D.GPU_HIDDEN, gpu=False) == "cpu"
    monkeypatch.setenv("DMFIN_DEVICE", "cpu")
    assert not D.gpu_available()
    assert D.large_work_device() == "cpu"


def _panel(n_series=6, length=220, seed=3):
    """Several AR(1) series stacked as a panel; target = next value of feature 0."""
    rng = np.random.default_rng(seed)
    X, y, rows = [], [], []
    for s in range(n_series):
        x = np.zeros((length, 3), np.float32)
        for t in range(1, length):
            x[t] = 0.8 * x[t - 1] + rng.normal(scale=0.5, size=3)
        start = s * length
        X.append(x); y.append(np.r_[x[1:, 0], 0.0].astype(np.float32))
        rows.append(np.arange(start + 10, start + length - 1))
    return np.vstack(X), np.concatenate(y), np.concatenate(rows)


@pytest.mark.skipif(not GPU, reason="no CUDA/ROCm device")
def test_global_lstm_trains_identically_on_gpu_and_cpu():
    """Without dropout the same seed gives the same training path on both devices."""
    X, y, rows = _panel()
    tr, va = rows[rows % 220 < 160], rows[rows % 220 >= 160]
    out = {}
    for dev in ("cpu", "cuda"):
        m, hist, win = train_global("lstm", X, y, tr, va, lookback=10, seed=0, device=dev, batch=128, max_epochs=3)
        out[dev] = (hist, predict_rows(m, win, va, dev))
    assert np.allclose(out["cpu"][0], out["cuda"][0], rtol=1e-3, atol=1e-4)
    assert np.allclose(out["cpu"][1], out["cuda"][1], atol=2e-3)


@pytest.mark.skipif(not GPU, reason="no CUDA/ROCm device")
@pytest.mark.parametrize("name", ["tcn", "transformer"])
def test_global_network_with_dropout_on_gpu(name):
    """With dropout the GPU draws other random masks, so the training paths differ; the same weights
    give the same forecasts on both devices, and training on the GPU reaches a similar validation loss."""
    X, y, rows = _panel()
    tr, va = rows[rows % 220 < 160], rows[rows % 220 >= 160]
    m, hist_cpu, win = train_global(name, X, y, tr, va, lookback=10, seed=0, device="cpu", batch=128, max_epochs=3)
    p_cpu = predict_rows(m, win, va, "cpu")
    m.to("cuda")
    Xt = torch.as_tensor(X, device="cuda"); offs = torch.arange(-9, 1, device="cuda")
    p_gpu = predict_rows(m, lambda r: Xt[r[:, None] + offs[None, :]], va, "cuda")
    assert np.allclose(p_cpu, p_gpu, atol=1e-4)
    _, hist_gpu, _ = train_global(name, X, y, tr, va, lookback=10, seed=0, device="cuda", batch=128, max_epochs=3)
    assert abs(hist_gpu[-1, 1] - hist_cpu[-1, 1]) < 0.1 * hist_cpu[-1, 1]


def _chronos2_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache("amazon/chronos-2", "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not (GPU and _chronos2_cached()), reason="no GPU or Chronos-2 not downloaded")
def test_chronos2_forecasts_on_gpu_match_cpu():
    from chronos import BaseChronosPipeline
    rng = np.random.default_rng(0)
    ctx = [torch.tensor(np.cumsum(rng.normal(size=200)).astype(np.float32)) for _ in range(8)]
    med = {}
    for dev in ("cpu", "cuda"):
        pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=dev, torch_dtype=torch.float32)
        q, _ = pipe.predict_quantiles(ctx, prediction_length=5, quantile_levels=[0.5])
        med[dev] = np.stack([t[0, :, 0].float().cpu().numpy() for t in q])
    assert np.allclose(med["cpu"], med["cuda"], rtol=1e-3, atol=1e-3)
