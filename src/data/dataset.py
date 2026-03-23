"""
Synthetic MovieLens-style dataset with realistic user-item interactions.
Generates data using a latent factor model if no data file exists.
"""
import os
import numpy as np
import pandas as pd


class MovieLensDataset:
    GENRES = [
        'Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi',
        'Romance', 'Thriller', 'Animation', 'Documentary', 'Fantasy',
    ]

    def __init__(self, data_dir='data', n_users=500, n_items=200):
        self.data_dir = data_dir
        self.n_users = n_users
        self.n_items = n_items
        os.makedirs(data_dir, exist_ok=True)

        ratings_path = os.path.join(data_dir, 'ratings.csv')
        items_path = os.path.join(data_dir, 'items.csv')

        if os.path.exists(ratings_path) and os.path.exists(items_path):
            self.ratings_df = pd.read_csv(ratings_path)
            self.items_df = pd.read_csv(items_path)
        else:
            self._generate_synthetic()

        self._preprocess()

    # ------------------------------------------------------------------
    def _generate_synthetic(self):
        np.random.seed(42)
        n_factors = 20
        user_factors = np.random.randn(self.n_users, n_factors)
        item_factors = np.random.randn(self.n_items, n_factors)

        interactions = []
        for user_id in range(self.n_users):
            n_ratings = np.random.randint(20, 80)
            scores = user_factors[user_id] @ item_factors.T
            probs = np.exp(scores - scores.max())
            probs /= probs.sum()
            n_sample = min(n_ratings, self.n_items)
            rated_items = np.random.choice(self.n_items, size=n_sample, replace=False, p=probs)
            for item_id in rated_items:
                score = user_factors[user_id] @ item_factors[item_id]
                rating = int(np.clip(round((score + 2) / 4 * 4 + 1), 1, 5))
                timestamp = np.random.randint(1_000_000, 2_000_000)
                interactions.append((user_id, item_id, rating, timestamp))

        self.ratings_df = pd.DataFrame(
            interactions, columns=['user_id', 'item_id', 'rating', 'timestamp']
        )
        self.ratings_df = self.ratings_df.sort_values('timestamp').reset_index(drop=True)

        items = []
        for item_id in range(self.n_items):
            n_genres = np.random.randint(1, 4)
            item_genres = np.random.choice(self.GENRES, size=n_genres, replace=False)
            items.append({
                'item_id': item_id,
                'title': f'Movie {item_id + 1}',
                'genres': '|'.join(item_genres),
            })
        self.items_df = pd.DataFrame(items)

        self.ratings_df.to_csv(os.path.join(self.data_dir, 'ratings.csv'), index=False)
        self.items_df.to_csv(os.path.join(self.data_dir, 'items.csv'), index=False)

    # ------------------------------------------------------------------
    def _preprocess(self):
        df = self.ratings_df.copy()
        df['reward'] = (df['rating'] >= 4).astype(int)

        self.user_sequences: dict = {}
        for user_id, group in df.groupby('user_id'):
            group_sorted = group.sort_values('timestamp')
            self.user_sequences[int(user_id)] = list(zip(
                group_sorted['item_id'].values.tolist(),
                group_sorted['reward'].values.tolist(),
            ))

        train_rows, test_rows = [], []
        for user_id, seq in self.user_sequences.items():
            split = max(1, int(len(seq) * 0.8))
            for i, (item_id, reward) in enumerate(seq):
                row = {'user_id': user_id, 'item_id': item_id, 'reward': reward}
                (train_rows if i < split else test_rows).append(row)

        self.train_df = pd.DataFrame(train_rows)
        self.test_df = pd.DataFrame(test_rows)
        self.n_users_actual = int(df['user_id'].nunique())
        self.n_items_actual = int(df['item_id'].nunique())

    # ------------------------------------------------------------------
    def get_user_history(self, user_id: int, max_len: int = 20):
        return self.user_sequences.get(user_id, [])[-max_len:]

    def get_item_info(self, item_id: int) -> dict:
        row = self.items_df[self.items_df['item_id'] == item_id]
        if len(row) == 0:
            return {'item_id': item_id, 'title': f'Movie {item_id + 1}', 'genres': 'Unknown'}
        r = row.iloc[0]
        return {'item_id': int(item_id), 'title': str(r['title']), 'genres': str(r['genres'])}
