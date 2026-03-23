"""GRU-based state encoder: maps user interaction history to a dense state vector."""
import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(self, n_items: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.n_items = n_items
        self.hidden_dim = hidden_dim

        # index 0 is padding; real items are shifted by +1
        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.reward_emb = nn.Embedding(2, 4)
        self.gru = nn.GRU(embed_dim + 4, hidden_dim, batch_first=True)

    def forward(self, item_seq: torch.Tensor, reward_seq: torch.Tensor) -> torch.Tensor:
        """
        item_seq   : (batch, seq_len) LongTensor
        reward_seq : (batch, seq_len) LongTensor  (0 or 1)
        returns    : (batch, hidden_dim)
        """
        x = torch.cat([self.item_emb(item_seq), self.reward_emb(reward_seq)], dim=-1)
        _, h = self.gru(x)
        return h.squeeze(0)  # (batch, hidden_dim)

    @torch.no_grad()
    def encode(self, history: list, max_len: int = 10, device: str = 'cpu') -> torch.Tensor:
        """Encode a single user history list [(item_id, reward), ...] → (1, hidden_dim)."""
        history = history[-max_len:]
        pad = max_len - len(history)
        items   = [0] * pad + [item_id + 1 for item_id, _ in history]
        rewards = [0] * pad + [int(r) for _, r in history]
        item_t   = torch.tensor([items],   dtype=torch.long, device=device)
        reward_t = torch.tensor([rewards], dtype=torch.long, device=device)
        return self.forward(item_t, reward_t)
