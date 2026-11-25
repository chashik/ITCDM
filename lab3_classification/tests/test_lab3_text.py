"""Lab 3 (iteration 3) tests: text helpers and the training loop of the text classifiers (synthetic data, offline)."""
import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from dmfin.device import gpu_available
from lab3_classification.text import (LSTMClassifier, Vocabulary, macro_f1, normalize, paired_bootstrap,
                                      predict_proba, train, transformer)

A, B, START, END, PAD = 3, 4, 0, 2, 1


def order_task(n=900, length=12, seed=0):
    """Token A before token B -> class 0, after -> class 1; the other tokens are random.  Both classes contain the same
    tokens, so their counts (a bag of words) carry no information; only the order does."""
    rng = np.random.default_rng(seed)
    X, y = [], rng.integers(0, 2, n)
    for c in y:
        body = rng.integers(5, 24, length - 2)
        i, j = sorted(rng.choice(length - 2, 2, replace=False))
        body[i], body[j] = (A, B) if c == 0 else (B, A)
        X.append(np.concatenate([[START], body, [END]]).astype(np.int64))
    return X[:600], y[:600], X[600:], y[600:]


def tiny_transformer(seed=0):
    from transformers import RobertaConfig
    cfg = RobertaConfig(vocab_size=24, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                        max_position_embeddings=20, pad_token_id=PAD, bos_token_id=START, eos_token_id=END,
                        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, num_labels=3)
    return transformer(pretrained=False, seed=seed, config=cfg)


def test_normalize_replaces_links_and_keeps_tickers():
    assert normalize("Stocks fall  https://t.co/AbC12 on $AAPL news\n") == "Stocks fall URL on $AAPL news"


def test_macro_f1_equals_scikit_learn_with_all_classes():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 2, 500)          # class 2 is never predicted
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))


def test_paired_bootstrap_interval():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 400)
    weak = np.where(rng.uniform(size=400) < 0.5, y, rng.integers(0, 3, 400))
    strong = np.where(rng.uniform(size=400) < 0.9, y, rng.integers(0, 3, 400))
    (lo, hi), p = paired_bootstrap(y, weak, np.vstack([strong, strong]), n=300)
    assert 0 < lo < macro_f1(y, strong) - macro_f1(y, weak) < hi and p == 0
    assert np.allclose(paired_bootstrap(y, weak, weak, n=50)[0], 0)
    assert np.allclose(paired_bootstrap(y, weak, strong, n=50)[0], paired_bootstrap(y, weak, strong, n=50)[0])


def test_vocabulary_maps_unseen_tokens_to_one():
    v = Vocabulary([np.array([5, 7, 7]), np.array([9])], size=12)
    assert v.size == 5 and v([np.array([5, 9, 11])])[0].tolist() == [2, 4, 1]


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_networks_learn_word_order_that_a_bag_of_words_cannot(kind):
    torch.set_num_threads(1)
    Xtr, ytr, Xva, yva = order_task()
    counts = lambda X: np.array([np.bincount(x, minlength=24) for x in X])
    bow = LogisticRegression(max_iter=2000).fit(counts(Xtr), ytr).score(counts(Xva), yva)
    model = LSTMClassifier(24, emb=16, hidden=16, dropout=0.0) if kind == "lstm" else tiny_transformer()
    model, hist, _ = train(model, Xtr, ytr, Xva, yva, lr=3e-3 if kind == "lstm" else 1e-3, max_epochs=40, patience=40,
                           batch=32, pad_id=PAD)
    acc = np.mean(predict_proba(model, Xva, "cpu", PAD).argmax(1) == yva)
    assert bow < 0.62 and acc > 0.9


def test_training_keeps_the_best_epoch_and_is_reproducible():
    torch.set_num_threads(1)
    Xtr, ytr, Xva, yva = order_task(seed=2)
    runs = [train(tiny_transformer(seed=5), Xtr, ytr, Xva, yva, lr=3e-3, max_epochs=8, patience=8, seed=5, pad_id=PAD)
            for _ in range(2)]
    p = [predict_proba(m, Xva, "cpu", PAD) for m, _, _ in runs]
    assert macro_f1(yva, p[0].argmax(1)) == pytest.approx(runs[0][1][:, 3].max())
    assert np.allclose(p[0], p[1]) and np.allclose(runs[0][1], runs[1][1])


@pytest.mark.skipif(not gpu_available(), reason="no GPU")
def test_transformer_gives_the_same_probabilities_on_gpu_and_cpu():
    _, _, Xva, _ = order_task(seed=3)
    m = tiny_transformer(seed=1)
    cpu = predict_proba(m.to("cpu"), Xva, "cpu", PAD)
    gpu = predict_proba(m.to("cuda"), Xva, "cuda", PAD)
    assert np.allclose(cpu, gpu, atol=1e-4)
