"""
DQN agent for sequential recommendation.

State  = GRU encoding of the user's recent interaction history.
Action = item index to recommend (catalogue of n_items).
Reward = 1 if the user liked the item (rating ≥ 4), else 0.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .state_encoder import StateEncoder
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


class DQNAgent:
    def __init__(
        self,
        n_items: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update: int = 200,
    ):
        self.n_items = n_items
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0

        self.encoder = StateEncoder(n_items, embed_dim, hidden_dim)
        state_dim = hidden_dim

        self.q_net     = QNetwork(state_dim, n_items)
        self.target_net = QNetwork(state_dim, n_items)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        params = list(self.encoder.parameters()) + list(self.q_net.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)
        self.losses: list = []

    # ------------------------------------------------------------------
    # State / inference helpers
    # ------------------------------------------------------------------

    def get_state(self, history: list, max_len: int = 10) -> torch.Tensor:
        """Return (1, hidden_dim) state tensor for a user history."""
        return self.encoder.encode(history, max_len)

    def get_top_k_recommendations(
        self, history: list, k: int = 10, exclude_seen: bool = True
    ) -> list:
        """Return [(item_id, q_score), ...] sorted by Q-value descending."""
        state = self.get_state(history)
        seen = {item_id for item_id, _ in history} if exclude_seen else set()
        available = [i for i in range(self.n_items) if i not in seen]
        with torch.no_grad():
            q = self.q_net(state)[0].numpy()
        ranked = sorted([(i, float(q[i])) for i in available], key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store(self, state: torch.Tensor, action: int, reward: float,
              next_state: torch.Tensor, done: bool):
        self.buffer.push(
            state.numpy().flatten(),
            action,
            reward,
            next_state.numpy().flatten(),
            done,
        )

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(states)
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones       = torch.FloatTensor(dones)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q   = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q_net.parameters()) + list(self.encoder.parameters()), 1.0
        )
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'encoder':    self.encoder.state_dict(),
            'q_net':      self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon':    self.epsilon,
            'steps':      self.steps,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location='cpu')
        self.encoder.load_state_dict(ck['encoder'])
        self.q_net.load_state_dict(ck['q_net'])
        self.target_net.load_state_dict(ck['target_net'])
        self.epsilon = ck['epsilon']
        self.steps   = ck['steps']
