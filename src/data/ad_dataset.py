"""
Synthetic advertising dataset.

Generates realistic user profiles, ad inventory, and impression logs
using a latent-factor model to simulate interest-based click / conversion
probabilities that mirror real-world RTB (Real-Time Bidding) behaviour.

Key domain concepts implemented
--------------------------------
* Users       – demographic profile + interest vector
* Ads         – advertiser, category, format, bid price, targeting
* Impressions – (user, ad, timestamp, clicked, converted, dwell_time, revenue)
* Fatigue     – click-probability decays with repeated exposure to same ad
* Reward      – composite: click + conversion bonus − fatigue penalty + revenue signal
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES   = ['tech', 'fashion', 'sports', 'travel', 'food',
                 'finance', 'health', 'gaming', 'music', 'automotive']
ADVERTISERS  = ['TechCorp', 'FashionHub', 'SportsPro', 'TravelEase',
                'FoodieDeals', 'FinancePlus', 'HealthFirst', 'GamingZone',
                'MusicStream', 'AutoDrive']
AD_FORMATS   = ['banner', 'video', 'native', 'carousel']
AGE_GROUPS   = ['18-24', '25-34', '35-44', '45-54', '55+']
GENDERS      = ['M', 'F', 'Other']
DEVICES      = ['mobile', 'desktop', 'tablet']

# reward weights
R_CLICK      = 1.0
R_CONVERT    = 3.0
R_FATIGUE    = 0.2   # subtracted per repeated impression beyond 2
R_REVENUE    = 0.001 # per dollar of bid price


# ---------------------------------------------------------------------------
class AdDataset:
    def __init__(self, data_dir: str = 'data', n_users: int = 300, n_ads: int = 100):
        self.data_dir = data_dir
        self.n_users  = n_users
        self.n_ads    = n_ads
        os.makedirs(data_dir, exist_ok=True)

        users_path = os.path.join(data_dir, 'users.csv')
        ads_path   = os.path.join(data_dir, 'ads.csv')
        imp_path   = os.path.join(data_dir, 'impressions.csv')

        if all(os.path.exists(p) for p in (users_path, ads_path, imp_path)):
            self.users_df       = pd.read_csv(users_path)
            self.ads_df         = pd.read_csv(ads_path)
            self.impressions_df = pd.read_csv(imp_path)
        else:
            self._generate()

        self._preprocess()

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    def _generate(self):
        np.random.seed(42)

        # ── Users ──────────────────────────────────────────────────────
        users = []
        for uid in range(self.n_users):
            n_interests = np.random.randint(1, 4)
            interests   = np.random.choice(CATEGORIES, size=n_interests, replace=False)
            users.append({
                'user_id':    uid,
                'age_group':  np.random.choice(AGE_GROUPS),
                'gender':     np.random.choice(GENDERS),
                'interests':  '|'.join(interests),
                'device':     np.random.choice(DEVICES),
            })
        self.users_df = pd.DataFrame(users)

        # ── Ads ────────────────────────────────────────────────────────
        ads = []
        for ad_id in range(self.n_ads):
            cat    = CATEGORIES[ad_id % len(CATEGORIES)]
            adv    = ADVERTISERS[ad_id % len(ADVERTISERS)]
            fmt    = np.random.choice(AD_FORMATS)
            bid    = round(np.random.uniform(0.5, 8.0), 2)
            budget = round(np.random.uniform(50, 500), 0)
            n_tgt  = np.random.randint(2, 5)
            target_ages = '|'.join(np.random.choice(AGE_GROUPS, size=n_tgt, replace=False))
            headlines = {
                'tech': 'Upgrade your tech today!',      'fashion': 'Style up this season!',
                'sports': 'Gear up for greatness!',      'travel': 'Explore the world now!',
                'food': 'Delicious deals await!',        'finance': 'Grow your wealth today!',
                'health': 'Live healthier, longer!',     'gaming': 'Level up your game!',
                'music': 'Discover new sounds!',         'automotive': 'Drive your dream car!',
            }
            ads.append({
                'ad_id':        ad_id,
                'advertiser':   adv,
                'category':     cat,
                'format':       fmt,
                'bid_price':    bid,
                'daily_budget': budget,
                'target_ages':  target_ages,
                'headline':     headlines[cat],
                'ctr_base':     round(np.random.uniform(0.02, 0.06), 4),
                'cvr_base':     round(np.random.uniform(0.05, 0.15), 4),
            })
        self.ads_df = pd.DataFrame(ads)

        # ── Impressions ────────────────────────────────────────────────
        impressions = []
        imp_id      = 0
        for uid in range(self.n_users):
            user       = self.users_df.iloc[uid]
            u_ints     = set(user['interests'].split('|'))
            n_imps     = np.random.randint(30, 100)
            ad_ids     = np.random.choice(self.n_ads, size=n_imps, replace=True)
            freq_count = {}   # ad → exposure count for fatigue

            for t, ad_id in enumerate(ad_ids):
                ad       = self.ads_df.iloc[ad_id]
                freq     = freq_count.get(ad_id, 0)
                freq_count[ad_id] = freq + 1

                # click probability: base + interest boost + fatigue decay
                match    = ad['category'] in u_ints
                ctr      = ad['ctr_base'] * (3.0 if match else 1.0) * (0.8 ** max(0, freq - 1))
                ctr      = min(ctr, 0.35)
                clicked  = int(np.random.rand() < ctr)

                cvr      = ad['cvr_base'] * (1.5 if match else 1.0) if clicked else 0
                converted = int(np.random.rand() < cvr)

                dwell    = np.random.exponential(15.0) if clicked else np.random.exponential(2.0)
                revenue  = ad['bid_price'] if clicked else 0.0

                # reward
                fatigue_pen = max(0, freq - 1)
                reward = (
                    R_CLICK   * clicked
                    + R_CONVERT * converted
                    - R_FATIGUE * fatigue_pen
                    + R_REVENUE * revenue
                )

                hour = np.random.randint(0, 24)
                dow  = np.random.randint(0, 7)

                impressions.append({
                    'impression_id': imp_id,
                    'user_id':       uid,
                    'ad_id':         int(ad_id),
                    'timestamp':     1_000_000 + uid * 1000 + t,
                    'clicked':       clicked,
                    'converted':     converted,
                    'dwell_time':    round(dwell, 2),
                    'revenue':       round(revenue, 4),
                    'reward':        round(reward, 4),
                    'freq_count':    freq,
                    'hour_of_day':   hour,
                    'day_of_week':   dow,
                })
                imp_id += 1

        self.impressions_df = pd.DataFrame(impressions)

        # Save
        self.users_df.to_csv(os.path.join(self.data_dir, 'users.csv'), index=False)
        self.ads_df.to_csv(os.path.join(self.data_dir, 'ads.csv'),   index=False)
        self.impressions_df.to_csv(os.path.join(self.data_dir, 'impressions.csv'), index=False)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self):
        df = self.impressions_df.sort_values('timestamp')

        # Build per-user interaction sequences
        self.user_sequences: dict = {}
        for uid, grp in df.groupby('user_id'):
            self.user_sequences[int(uid)] = list(zip(
                grp['ad_id'].values.tolist(),
                grp['reward'].values.tolist(),
                grp['clicked'].values.tolist(),
                grp['converted'].values.tolist(),
            ))

        # Train / test split (80 / 20 per user)
        train_rows, test_rows = [], []
        for uid, seq in self.user_sequences.items():
            split = max(1, int(len(seq) * 0.8))
            for i, (ad_id, reward, clicked, converted) in enumerate(seq):
                row = {'user_id': uid, 'ad_id': ad_id, 'reward': reward,
                       'clicked': clicked, 'converted': converted}
                (train_rows if i < split else test_rows).append(row)

        self.train_df = pd.DataFrame(train_rows)
        self.test_df  = pd.DataFrame(test_rows)

        self.n_users_actual = int(df['user_id'].nunique())
        self.n_ads_actual   = int(df['ad_id'].nunique())

        # Pre-compute aggregate analytics
        self._compute_analytics()

    def _compute_analytics(self):
        df = self.impressions_df
        total_imp  = len(df)
        total_clk  = df['clicked'].sum()
        total_conv = df['converted'].sum()
        total_rev  = df['revenue'].sum()

        self.analytics = {
            'total_impressions': int(total_imp),
            'total_clicks':      int(total_clk),
            'total_conversions': int(total_conv),
            'total_revenue':     round(float(total_rev), 2),
            'ctr':               round(float(total_clk / max(1, total_imp)), 4),
            'cvr':               round(float(total_conv / max(1, total_clk)), 4),
            'ecpm':              round(float(total_rev / max(1, total_imp) * 1000), 4),
        }

        # By category
        merged = df.merge(self.ads_df[['ad_id', 'category', 'advertiser', 'bid_price']],
                          on='ad_id', how='left')
        cat_stats = []
        for cat, g in merged.groupby('category'):
            cat_stats.append({
                'category':    cat,
                'impressions': int(len(g)),
                'clicks':      int(g['clicked'].sum()),
                'conversions': int(g['converted'].sum()),
                'revenue':     round(float(g['revenue'].sum()), 2),
                'ctr':         round(float(g['clicked'].mean()), 4),
            })
        self.analytics['by_category'] = cat_stats

        # By advertiser
        adv_stats = []
        for adv, g in merged.groupby('advertiser'):
            spend = g['revenue'].sum()
            conv  = g['converted'].sum()
            adv_stats.append({
                'advertiser':  adv,
                'impressions': int(len(g)),
                'clicks':      int(g['clicked'].sum()),
                'conversions': int(conv),
                'spend':       round(float(spend), 2),
                'ctr':         round(float(g['clicked'].mean()), 4),
                'roas':        round(float(conv * 50 / max(0.01, spend)), 2),
            })
        self.analytics['by_advertiser'] = adv_stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def get_user_history(self, user_id: int, max_len: int = 20) -> list:
        """Returns [(ad_id, reward, clicked, converted), ...]."""
        return self.user_sequences.get(user_id, [])[-max_len:]

    def get_user_features(self, user_id: int) -> np.ndarray:
        """Return a 21-dim binary/float feature vector for the user."""
        row = self.users_df[self.users_df['user_id'] == user_id]
        if len(row) == 0:
            return np.zeros(21, dtype=np.float32)
        r       = row.iloc[0]
        age_oh  = np.zeros(5);  age_oh[AGE_GROUPS.index(r['age_group'])]   = 1
        gen_oh  = np.zeros(3);  gen_oh[GENDERS.index(r['gender'])]         = 1
        int_oh  = np.zeros(10)
        for interest in str(r['interests']).split('|'):
            if interest in CATEGORIES:
                int_oh[CATEGORIES.index(interest)] = 1
        dev_oh  = np.zeros(3);  dev_oh[DEVICES.index(r['device'])]         = 1
        return np.concatenate([age_oh, gen_oh, int_oh, dev_oh]).astype(np.float32)

    def get_context_features(self, hour: int = None, dow: int = None) -> np.ndarray:
        """4-dim sinusoidal encoding of hour and day-of-week."""
        if hour is None:
            import datetime
            now  = datetime.datetime.now()
            hour = now.hour
            dow  = now.weekday()
        return np.array([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow  / 7),
            np.cos(2 * np.pi * dow  / 7),
        ], dtype=np.float32)

    def get_ad_info(self, ad_id: int) -> dict:
        row = self.ads_df[self.ads_df['ad_id'] == ad_id]
        if len(row) == 0:
            return {'ad_id': ad_id, 'advertiser': 'Unknown', 'category': 'unknown',
                    'format': 'banner', 'bid_price': 1.0, 'headline': '—'}
        r = row.iloc[0]
        return {
            'ad_id':      int(ad_id),
            'advertiser': str(r['advertiser']),
            'category':   str(r['category']),
            'format':     str(r['format']),
            'bid_price':  float(r['bid_price']),
            'headline':   str(r['headline']),
            'ctr_base':   float(r['ctr_base']),
            'cvr_base':   float(r['cvr_base']),
        }

    def get_user_profile(self, user_id: int) -> dict:
        row = self.users_df[self.users_df['user_id'] == user_id]
        if len(row) == 0:
            return {}
        r = row.iloc[0]
        return {
            'user_id':   int(user_id),
            'age_group': str(r['age_group']),
            'gender':    str(r['gender']),
            'interests': str(r['interests']).split('|'),
            'device':    str(r['device']),
        }
