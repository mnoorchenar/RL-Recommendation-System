"""
Offline evaluation metrics: Precision@K, Recall@K, NDCG@K, Hit-Rate@K.

Uses the held-out 20% of each user's interactions as the ground-truth test set.
Items rated ≥ 4 (reward == 1) are considered relevant.
"""
import numpy as np


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def _dcg(ranked: list, relevant: set, k: int) -> float:
    return sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(ranked[:k])
        if item in relevant
    )


def _idcg(n_relevant: int, k: int) -> float:
    return sum(1.0 / np.log2(i + 2) for i in range(min(n_relevant, k)))


# ------------------------------------------------------------------

class Evaluator:
    def __init__(self, agent, dataset, ks: tuple = (5, 10, 20)):
        self.agent = agent
        self.dataset = dataset
        self.ks = ks

    def evaluate(self, n_users: int = None) -> dict:
        test_users = self.dataset.test_df['user_id'].unique()
        if n_users:
            test_users = test_users[:n_users]

        buckets = {k: {'p': [], 'r': [], 'ndcg': [], 'hr': []} for k in self.ks}
        max_k = max(self.ks)

        for uid in test_users:
            ut = self.dataset.test_df[self.dataset.test_df['user_id'] == uid]
            relevant = set(ut[ut['reward'] == 1]['item_id'].tolist())
            if not relevant:
                continue

            seq = self.dataset.user_sequences.get(uid, [])
            split = int(len(seq) * 0.8)
            history = seq[:split]

            recs = self.agent.get_top_k_recommendations(history, k=max_k)
            ranked = [item_id for item_id, _ in recs]

            for k in self.ks:
                top_k = set(ranked[:k])
                hits  = len(top_k & relevant)
                buckets[k]['p'].append(hits / k)
                buckets[k]['r'].append(hits / len(relevant))
                buckets[k]['ndcg'].append(
                    _dcg(ranked, relevant, k) / max(1e-9, _idcg(len(relevant), k))
                )
                buckets[k]['hr'].append(int(bool(top_k & relevant)))

        results = {}
        for k in self.ks:
            results[f'P@{k}']    = round(float(np.mean(buckets[k]['p'])),    4)
            results[f'R@{k}']    = round(float(np.mean(buckets[k]['r'])),    4)
            results[f'NDCG@{k}'] = round(float(np.mean(buckets[k]['ndcg'])), 4)
            results[f'HR@{k}']   = round(float(np.mean(buckets[k]['hr'])),   4)
        return results
