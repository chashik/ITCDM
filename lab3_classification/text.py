"""Lab 3 (iteration 3). Classifiers of short financial texts: bag of words, recurrent network, transformer.

* ``normalize`` replaces links by the token URL;
* bag-of-words classifiers (scikit-learn): multinomial naive Bayes on word counts and logistic regression on
  TF-IDF weights of words and character n-grams;
* ``LSTMClassifier``: embedding of the tokens, bidirectional LSTM and max-pooling, trained from random weights;
* ``transformer``: DistilRoBERTa, a six-layer transformer pretrained on English web texts, with its pretrained
  weights or with random weights (the same architecture, so that the comparison isolates pretraining);
* ``train``: one training loop for both networks (AdamW, linear warm-up and decay, the weights of the epoch with the
  best validation macro-F1 are kept); ``embed``: mean of the last hidden states of the pretrained encoder, used as
  fixed features.

All networks read the byte-pair tokens of the RoBERTa tokenizer; the LSTM keeps only the tokens seen in training.
"""
from __future__ import annotations

import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, make_pipeline
from torch import nn

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # no tokenizer threads next to the GPU work

LM = "distilbert/distilroberta-base"
LM_REV = "fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b"         # revision of 2024-02-19, fixed for reproducibility
MAX_LEN = 64                                                 # tokens; longer tweets (7 of 9 543) are truncated
N_CLASSES = 3


def normalize(text: str) -> str:
    """Links (t.co short links) carry no content: each becomes the token URL."""
    return re.sub(r"\s+", " ", re.sub(r"https?://\S+", " URL ", text)).strip()


def macro_f1(y, pred, k: int = N_CLASSES) -> float:
    """Mean of the F1 measures of all k classes (a class never predicted and absent counts as 0)."""
    cm = np.bincount(k * np.asarray(y) + np.asarray(pred), minlength=k * k).reshape(k, k)
    tp, denom = np.diag(cm), cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)))


def paired_bootstrap(y, pred_a, preds_b, n: int = 2000, seed: int = 0):
    """Percentile interval (95 %) of macro-F1(b) - macro-F1(a), resampling the test texts with replacement.

    ``preds_b`` may hold the predictions of several seeds of model b (one row each); their macro-F1 is averaged, as in
    the table of results.  Also returns the share of resamples in which b is not better."""
    rng = np.random.default_rng(seed)
    y, pred_a, preds_b = np.asarray(y), np.asarray(pred_a), np.atleast_2d(preds_b)
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        d[i] = np.mean([macro_f1(y[idx], p[idx]) for p in preds_b]) - macro_f1(y[idx], pred_a[idx])
    return np.percentile(d, [2.5, 97.5]), float(np.mean(d <= 0))


# ----------------------------------------------------------------------------- bag of words
def naive_bayes(alpha: float):
    """Word and word-pair counts, multinomial naive Bayes with additive smoothing ``alpha``."""
    return make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB(alpha=alpha))


def tfidf_features():
    """TF-IDF of words and word pairs plus character 2-5-grams within words (terms of at least two texts)."""
    return FeatureUnion([("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                         ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                                   min_df=2))])


def tfidf_logistic(c: float, class_weight=None):
    """TF-IDF features and a multinomial logistic regression."""
    return make_pipeline(tfidf_features(), LogisticRegression(C=c, class_weight=class_weight, max_iter=5000))


# ----------------------------------------------------------------------------- tokens and networks
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(LM, revision=LM_REV)


def encode(tok, texts, max_len: int = MAX_LEN) -> list[np.ndarray]:
    """Token ids of each text (with the start and end tokens of the model), truncated to ``max_len``."""
    return [np.asarray(x, np.int64) for x in tok(list(texts), truncation=True, max_length=max_len)["input_ids"]]


def pad(seqs, pad_id: int):
    """Pads the sequences of a batch to the longest of them; returns ids and the attention mask."""
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), pad_id, np.int64)
    mask = np.zeros((len(seqs), L), np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)], mask[i, :len(s)] = s, 1
    return torch.from_numpy(ids), torch.from_numpy(mask)


class Vocabulary:
    """Maps the token ids of the tokenizer to the tokens seen in training; 0 = padding, 1 = unseen token."""

    def __init__(self, seqs, size: int):
        seen = np.unique(np.concatenate(seqs))
        self.lut = np.ones(size, np.int64)
        self.lut[seen] = np.arange(2, len(seen) + 2)
        self.size = len(seen) + 2

    def __call__(self, seqs):
        return [self.lut[s] for s in seqs]


class LSTMClassifier(nn.Module):
    """Embedding, bidirectional LSTM, maximum of the hidden states over the tokens of the text, linear output."""

    def __init__(self, vocab: int, emb: int = 128, hidden: int = 128, dropout: float = 0.3, n_classes: int = N_CLASSES):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, input_ids, attention_mask):
        lengths = attention_mask.sum(1).cpu()
        x = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(input_ids)), lengths, batch_first=True,
                                              enforce_sorted=False)
        h, _ = self.lstm(x)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=input_ids.shape[1])
        h = h.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e4)
        return self.out(self.drop(h.max(1).values))


def transformer(pretrained: bool, seed: int, config=None):
    """DistilRoBERTa with a classification head (two linear layers on the first token).  ``pretrained``: the published
    weights (the head is new); otherwise the same architecture with random weights, or ``config`` (tests)."""
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from transformers.utils import logging
    logging.set_verbosity_error()
    torch.manual_seed(seed)                               # initial weights of the head (and of the network)
    if pretrained:
        return AutoModelForSequenceClassification.from_pretrained(LM, revision=LM_REV, num_labels=N_CLASSES)
    cfg = config or AutoConfig.from_pretrained(LM, revision=LM_REV, num_labels=N_CLASSES)
    return AutoModelForSequenceClassification.from_config(cfg)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _logits(model, ids, mask):
    out = model(input_ids=ids, attention_mask=mask)
    return out.logits if hasattr(out, "logits") else out


@torch.no_grad()
def predict_proba(model, X, device: str, pad_id: int, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        ids, mask = pad(X[i:i + batch], pad_id)
        out.append(torch.softmax(_logits(model, ids.to(device), mask.to(device)).float(), -1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


@torch.no_grad()
def embed(encoder, X, device: str, pad_id: int, batch: int = 256) -> np.ndarray:
    """Mean of the last hidden states over the tokens of each text; the encoder is not changed."""
    encoder.eval().to(device)
    out = []
    for i in range(0, len(X), batch):
        ids, mask = pad(X[i:i + batch], pad_id)
        ids, mask = ids.to(device), mask.to(device)
        h = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.concatenate(out)


def log_loss(y, p) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def train(model, Xtr, ytr, Xva, yva, *, lr: float, max_epochs: int, batch: int = 32, patience: int = 2,
          warmup: float = 0.1, weight_decay: float = 0.01, seed: int = 0, device: str = "cpu", pad_id: int = 1):
    """AdamW with a learning rate that rises linearly over the first ``warmup`` share of the steps and then falls
    linearly to zero at ``max_epochs``.  After every epoch the validation macro-F1 is computed; the weights of the best
    epoch are kept and training stops after ``patience`` epochs without improvement.

    Returns (model with the weights of the best epoch, history of (epoch, training loss, validation loss, validation
    macro-F1), seconds)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device)
    per_epoch = math.ceil(len(Xtr) / batch)
    steps, warm = max_epochs * per_epoch, max(1, int(warmup * max_epochs * per_epoch))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    yt = torch.from_numpy(np.array(ytr, dtype=np.int64))     # a copy: pandas may hand out read-only arrays
    best, best_state, hist, bad = -1.0, None, [], 0
    t0 = time.perf_counter()
    for ep in range(max_epochs):
        model.train()
        tot = torch.zeros((), device=device)
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            ids, mask = pad([Xtr[j] for j in idx], pad_id)
            loss = F.cross_entropy(_logits(model, ids.to(device), mask.to(device)), yt[idx].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.detach() * len(idx)
        p = predict_proba(model, Xva, device, pad_id)
        f1 = macro_f1(yva, p.argmax(1))
        hist.append((ep + 1, float(tot) / len(Xtr), log_loss(np.asarray(yva), p), f1))
        if f1 > best:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    if device == "cuda":
        torch.cuda.synchronize()
    return model, np.array(hist), time.perf_counter() - t0
