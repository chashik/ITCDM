"""Hidden Markov model with Gaussian emissions (Rabiner 1989).

A hidden state q_t in {1, ..., N} follows a Markov chain with initial distribution pi and transition matrix A; the observed
value x_t (a vector of p features) is normal with the mean and the diagonal covariance of the current state. The forward
variables alpha_t(i) = P(x_1..x_t, q_t = i) give the likelihood (p. 262); they are rescaled at every t to stay in floating-
point range, and the log-likelihood is the sum of the logarithms of the scaling factors (p. 272-273). Baum-Welch (EM)
re-estimates pi, A and the emission parameters from the posterior state probabilities gamma and the posterior
transition probabilities xi (p. 264-266); Viterbi finds the single most probable state path (p. 263-264). Several
sequences (one per country) share the parameters: their statistics are summed.
"""
from __future__ import annotations

import numpy as np

VAR_FLOOR = 1e-3


def _emission(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """T x N densities of the observations under each state (diagonal Gaussian)."""
    d = X[:, None, :] - mu[None, :, :]
    logp = -0.5 * (np.log(2 * np.pi * var)[None] + d ** 2 / var[None]).sum(axis=2)
    return np.exp(logp)


def forward_backward(X: np.ndarray, pi: np.ndarray, A: np.ndarray, mu: np.ndarray, var: np.ndarray):
    """Scaled forward-backward. Returns (log-likelihood, gamma T x N, xi summed over t N x N)."""
    B = np.maximum(_emission(X, mu, var), 1e-300)
    T, N = B.shape
    alpha, c = np.zeros((T, N)), np.zeros(T)
    alpha[0] = pi * B[0]
    c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        c[t] = alpha[t].sum(); alpha[t] /= c[t]
    beta = np.ones((T, N))
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((N, N))
    for t in range(T - 1):
        x = alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
        xi += x / x.sum()
    return float(np.log(c).sum()), gamma, xi


def viterbi(X: np.ndarray, pi: np.ndarray, A: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    logB = np.log(np.maximum(_emission(X, mu, var), 1e-300))
    logA = np.log(np.maximum(A, 1e-300))
    T, N = logB.shape
    delta = np.log(np.maximum(pi, 1e-300)) + logB[0]
    back = np.zeros((T, N), dtype=int)
    for t in range(1, T):
        s = delta[:, None] + logA
        back[t] = s.argmax(axis=0)
        delta = s.max(axis=0) + logB[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


class GaussianHMM:
    def __init__(self, n_states: int, n_iter: int = 200, tol: float = 1e-6, n_starts: int = 10, seed: int = 0):
        self.N, self.n_iter, self.tol, self.n_starts, self.seed = n_states, n_iter, tol, n_starts, seed

    def _fit_once(self, seqs: list[np.ndarray], rng) -> tuple[float, tuple]:
        Xall = np.vstack(seqs)
        N, p = self.N, Xall.shape[1]
        mu = Xall[rng.choice(len(Xall), N, replace=False)].astype(float)
        var = np.tile(Xall.var(axis=0), (N, 1))
        A = np.full((N, N), 0.1 / max(N - 1, 1)); np.fill_diagonal(A, 0.9)
        pi = np.full(N, 1 / N)
        ll_old = -np.inf
        for _ in range(self.n_iter):
            ll, g0, xis, gsum, m1, m2 = 0.0, np.zeros(N), np.zeros((N, N)), np.zeros(N), np.zeros((N, p)), np.zeros((N, p))
            for X in seqs:
                l, gamma, xi = forward_backward(X, pi, A, mu, var)
                ll += l; g0 += gamma[0]; xis += xi; gsum += gamma.sum(axis=0)
                m1 += gamma.T @ X; m2 += gamma.T @ X ** 2
            pi = g0 / g0.sum()
            A = xis / np.maximum(xis.sum(axis=1, keepdims=True), 1e-300)
            mu = m1 / gsum[:, None]
            var = np.maximum(m2 / gsum[:, None] - mu ** 2, VAR_FLOOR)
            if ll - ll_old < self.tol:
                break
            ll_old = ll
        return ll, (pi, A, mu, var)

    def fit(self, seqs: list[np.ndarray]) -> "GaussianHMM":
        seqs = [np.asarray(s, float).reshape(len(s), -1) for s in seqs]
        rng = np.random.default_rng(self.seed)
        best = max((self._fit_once(seqs, rng) for _ in range(self.n_starts)), key=lambda r: r[0])
        self.loglik = best[0]
        pi, A, mu, var = best[1]
        order = np.argsort(mu[:, 0])                             # states ordered by the mean of the first feature
        self.pi, self.A, self.mu, self.var = pi[order], A[np.ix_(order, order)], mu[order], var[order]
        self.n_params = (self.N - 1) + self.N * (self.N - 1) + 2 * self.N * mu.shape[1]
        return self

    def decode(self, X) -> np.ndarray:
        return viterbi(np.asarray(X, float).reshape(len(X), -1), self.pi, self.A, self.mu, self.var)

    def posterior(self, X) -> np.ndarray:
        return forward_backward(np.asarray(X, float).reshape(len(X), -1), self.pi, self.A, self.mu, self.var)[1]

    def bic(self, n_obs: int) -> float:
        return -2 * self.loglik + self.n_params * np.log(n_obs)

    def durations(self) -> np.ndarray:
        """Expected duration of a stay in each state, 1 / (1 - a_ii) months."""
        return 1 / (1 - np.diag(self.A))
