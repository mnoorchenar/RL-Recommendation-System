"""
LinUCB contextual bandit — a simple, interpretable RL baseline.
Context = L2-normalised state vector produced by the GRU encoder.
"""
import numpy as np


class LinUCBBandit:
    def __init__(self, n_items: int, context_dim: int, alpha: float = 1.0):
        self.n_items = n_items
        self.alpha = alpha
        self.A = np.stack([np.eye(context_dim)] * n_items)   # (n_items, d, d)
        self.b = np.zeros((n_items, context_dim))             # (n_items, d)
        self.update_count = 0

    def _normalise(self, ctx: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(ctx)
        return ctx / (norm + 1e-8)

    def _ucb_scores(self, ctx: np.ndarray, items: list) -> dict:
        scores = {}
        for i in items:
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            scores[i] = float(theta @ ctx + self.alpha * np.sqrt(ctx @ A_inv @ ctx))
        return scores

    def get_top_k(self, ctx: np.ndarray, k: int = 10, exclude=None) -> list:
        ctx = self._normalise(ctx)
        available = [i for i in range(self.n_items) if i not in (exclude or set())]
        scores = self._ucb_scores(ctx, available)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    def update(self, item_id: int, ctx: np.ndarray, reward: float):
        ctx = self._normalise(ctx)
        self.A[item_id] += np.outer(ctx, ctx)
        self.b[item_id] += reward * ctx
        self.update_count += 1
