"""
Ad-domain evaluation metrics.

Offline ranking metrics  : Precision@K, Recall@K, NDCG@K, Hit-Rate@K
                           (ads clicked in test set = relevant)
Business KPIs             : CTR, CVR, eCPM, simulated Revenue
Coverage                  : fraction of ad categories surfaced in top-K
"""
import numpy as np


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def _dcg(ranked: list, rel_set: set, k: int) -> float:
    return sum(1.0 / np.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in rel_set)


def _idcg(n_rel: int, k: int) -> float:
    return sum(1.0 / np.log2(i + 2) for i in range(min(n_rel, k)))


# ---------------------------------------------------------------------------

class AdEvaluator:
    def __init__(self, agent, dataset, ks: tuple = (5, 10, 20)):
        self.agent   = agent
        self.dataset = dataset
        self.ks      = ks

    def evaluate(self, n_users: int = None) -> dict:
        """
        Returns a flat dict of all metrics suitable for JSON serialisation.
        """
        test_users = self.dataset.test_df['user_id'].unique()
        if n_users:
            test_users = test_users[:n_users]

        ranking_buckets = {k: {'p': [], 'r': [], 'ndcg': [], 'hr': []} for k in self.ks}
        max_k           = max(self.ks)

        # For business KPIs we simulate: for each test impression ask DQN,
        # then check whether the recommended ad was clicked / converted.
        sim_clicks      = []
        sim_converts    = []
        sim_revenues    = []

        from ..data.ad_dataset import CATEGORIES

        all_categories_recommended = set()

        for uid in test_users:
            ut       = self.dataset.test_df[self.dataset.test_df['user_id'] == uid]
            relevant = set(ut[ut['clicked'] == 1]['ad_id'].tolist())
            if not relevant:
                continue

            seq       = self.dataset.user_sequences.get(uid, [])
            split     = int(len(seq) * 0.8)
            history   = seq[:split]
            user_feat = self.dataset.get_user_features(uid)
            ctx_feat  = self.dataset.get_context_features()

            recs    = self.agent.get_top_k_recommendations(history, user_feat, ctx_feat, k=max_k)
            ranked  = [r['ad_id'] for r in recs]

            # Track category coverage
            for ad_id in ranked:
                info = self.dataset.get_ad_info(ad_id)
                all_categories_recommended.add(info['category'])

            # Ranking metrics
            for k in self.ks:
                top_k = set(ranked[:k])
                hits  = len(top_k & relevant)
                ranking_buckets[k]['p'].append(hits / k)
                ranking_buckets[k]['r'].append(hits / len(relevant))
                ranking_buckets[k]['ndcg'].append(
                    _dcg(ranked, relevant, k) / max(1e-9, _idcg(len(relevant), k))
                )
                ranking_buckets[k]['hr'].append(int(bool(top_k & relevant)))

            # Business KPI simulation (top-1 recommendation)
            if ranked:
                top_ad  = ranked[0]
                info    = self.dataset.get_ad_info(top_ad)
                # Use base CTR/CVR from ad meta + relevance boost
                user_interests = set(self.dataset.get_user_profile(uid).get('interests', []))
                match   = info['category'] in user_interests
                p_click = info['ctr_base'] * (2.5 if match else 1.0)
                p_conv  = info['cvr_base'] * (1.5 if match else 1.0) * p_click
                sim_clicks.append(min(p_click, 0.40))
                sim_converts.append(min(p_conv, 0.20))
                sim_revenues.append(info['bid_price'] * p_click)

        # Aggregate ranking metrics
        results = {}
        for k in self.ks:
            results[f'P@{k}']    = round(float(np.mean(ranking_buckets[k]['p'])),    4)
            results[f'R@{k}']    = round(float(np.mean(ranking_buckets[k]['r'])),    4)
            results[f'NDCG@{k}'] = round(float(np.mean(ranking_buckets[k]['ndcg'])), 4)
            results[f'HR@{k}']   = round(float(np.mean(ranking_buckets[k]['hr'])),   4)

        # Business KPIs
        avg_ctr     = float(np.mean(sim_clicks))    if sim_clicks    else 0.0
        avg_cvr     = float(np.mean(sim_converts))  if sim_converts  else 0.0
        avg_ecpm    = float(np.mean(sim_revenues)) * 1000 if sim_revenues else 0.0
        n_cats      = len(CATEGORIES)
        coverage    = len(all_categories_recommended) / n_cats

        results['CTR']      = round(avg_ctr,  4)
        results['CVR']      = round(avg_cvr,  4)
        results['eCPM']     = round(avg_ecpm, 4)
        results['Coverage'] = round(coverage, 4)

        return results
