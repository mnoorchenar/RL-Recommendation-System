"""
DQN agent for ad recommendation.

State  : 52-dim vector from AdStateEncoder (GRU history + user features + context)
Action : ad_id  ∈ {0 … n_ads − 1}
Reward : composite  =  R_CLICK·click + R_CONVERT·convert
                      − R_FATIGUE·fatigue_count + R_REVENUE·bid_price

The target network (Polyak / hard-copy update) provides stable TD targets.
Smooth-L1 (Huber) loss is used for robustness to reward-scale variation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .state_encoder import AdStateEncoder
from .ctr_model import CTRModel
from ..data.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdDQNAgent:
    STATE_DIM = AdStateEncoder.OUTPUT_DIM  # 52

    def __init__(
        self,
        n_ads:            int,
        user_feat_dim:    int   = 21,
        lr:               float = 1e-3,
        gamma:            float = 0.99,
        epsilon:          float = 1.0,
        epsilon_min:      float = 0.05,
        epsilon_decay:    float = 0.995,
        buffer_capacity:  int   = 30_000,
        batch_size:       int   = 64,
        target_update:    int   = 200,
    ):
        self.n_ads         = n_ads
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps         = 0

        # Networks
        self.encoder    = AdStateEncoder(n_ads, user_feat_dim)
        self.q_net      = QNetwork(self.STATE_DIM, n_ads)
        self.target_net = QNetwork(self.STATE_DIM, n_ads)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.ctr_model  = CTRModel(self.STATE_DIM, n_ads)

        params = (list(self.encoder.parameters())
                  + list(self.q_net.parameters())
                  + list(self.ctr_model.parameters()))
        self.optimizer = optim.Adam(params, lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.losses: list = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_state(self, history: list, user_feat, ctx_feat) -> torch.Tensor:
        return self.encoder.encode(history, user_feat, ctx_feat)  # (1, 52)

    def get_top_k_recommendations(
        self,
        history:       list,
        user_feat,
        ctx_feat,
        k:             int  = 10,
        exclude_seen:  bool = False,
    ) -> list:
        """
        Returns list of dicts:
          {ad_id, q_score, p_click, p_convert}
        sorted by q_score descending.
        """
        state = self.get_state(history, user_feat, ctx_feat)
        seen  = {ad_id for ad_id, *_ in history} if exclude_seen else set()
        avail = [i for i in range(self.n_ads) if i not in seen]

        with torch.no_grad():
            q = self.q_net(state)[0].numpy()

        ranked     = sorted(avail, key=lambda i: q[i], reverse=True)[:k]
        ctr_preds  = self.ctr_model.predict(state, ranked)

        return [
            {
                'ad_id':     int(ad_id),
                'q_score':   round(float(q[ad_id]), 4),
                'p_click':   round(ctr_preds[i][0], 4),
                'p_convert': round(ctr_preds[i][1], 4),
            }
            for i, ad_id in enumerate(ranked)
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(
            state.numpy().flatten(),
            action, reward,
            next_state.numpy().flatten(),
            done,
        )

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        S  = torch.FloatTensor(states)
        A  = torch.LongTensor(actions)
        R  = torch.FloatTensor(rewards)
        S_ = torch.FloatTensor(next_states)
        D  = torch.FloatTensor(dones)

        current_q = self.q_net(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q   = self.target_net(S_).max(1)[0]
            target_q = R + self.gamma * next_q * (1 - D)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_net.parameters()) + list(self.encoder.parameters()), 1.0
        )
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps  += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        lv = loss.item()
        self.losses.append(lv)
        return lv

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'encoder':    self.encoder.state_dict(),
            'q_net':      self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'ctr_model':  self.ctr_model.state_dict(),
            'epsilon':    self.epsilon,
            'steps':      self.steps,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location='cpu')
        self.encoder.load_state_dict(ck['encoder'])
        self.q_net.load_state_dict(ck['q_net'])
        self.target_net.load_state_dict(ck['target_net'])
        self.ctr_model.load_state_dict(ck['ctr_model'])
        self.epsilon = ck['epsilon']
        self.steps   = ck['steps']
