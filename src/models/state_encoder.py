"""
Ad-aware state encoder.

State = concat(
    GRU_hidden(ad_interaction_sequence),   # 32-dim
    Linear(user_features),                 # 16-dim
    context_features,                      #  4-dim (sinusoidal hour + dow)
)  →  52-dim state vector
"""
import torch
import torch.nn as nn
import numpy as np


class AdStateEncoder(nn.Module):
    OUTPUT_DIM = 52   # 32 + 16 + 4

    def __init__(self, n_ads: int, user_feat_dim: int = 21):
        super().__init__()
        self.n_ads = n_ads

        # Ad interaction sequence encoder
        # Each step: ad_emb(16) + outcome_emb(4) = 20-dim input
        self.ad_emb      = nn.Embedding(n_ads + 1, 16, padding_idx=0)  # +1 for pad
        self.outcome_emb = nn.Embedding(4, 4)   # 0=no_imp, 1=click, 2=convert, 3=skip
        self.gru         = nn.GRU(20, 32, batch_first=True)

        # User features projection
        self.user_proj = nn.Sequential(
            nn.Linear(user_feat_dim, 16),
            nn.ReLU(),
        )

    def forward(
        self,
        ad_seq:      torch.Tensor,   # (batch, seq_len)  LongTensor
        outcome_seq: torch.Tensor,   # (batch, seq_len)  LongTensor
        user_feat:   torch.Tensor,   # (batch, user_feat_dim)
        ctx_feat:    torch.Tensor,   # (batch, 4)
    ) -> torch.Tensor:
        # Sequence branch
        x = torch.cat([self.ad_emb(ad_seq), self.outcome_emb(outcome_seq)], dim=-1)
        _, h = self.gru(x)
        gru_out = h.squeeze(0)                   # (batch, 32)

        user_out = self.user_proj(user_feat)      # (batch, 16)
        state    = torch.cat([gru_out, user_out, ctx_feat], dim=-1)  # (batch, 52)
        return state

    @torch.no_grad()
    def encode(
        self,
        history:    list,           # [(ad_id, reward, clicked, converted), ...]
        user_feat:  np.ndarray,     # (21,)
        ctx_feat:   np.ndarray,     # (4,)
        max_len:    int = 10,
    ) -> torch.Tensor:
        """Single-sample encoding. Returns (1, 52) state tensor."""
        history = history[-max_len:]
        pad     = max_len - len(history)

        ads      = [0] * pad + [int(ad_id) + 1 for ad_id, *_ in history]
        outcomes = [0] * pad

        for _, _, clicked, converted in history:
            if converted:
                outcomes.append(2)
            elif clicked:
                outcomes.append(1)
            else:
                outcomes.append(3)

        ad_t  = torch.tensor([ads],     dtype=torch.long)
        out_t = torch.tensor([outcomes], dtype=torch.long)
        u_t   = torch.from_numpy(np.asarray(user_feat, dtype=np.float32).reshape(1, -1))
        ctx_t = torch.from_numpy(np.asarray(ctx_feat,  dtype=np.float32).reshape(1, -1))

        return self.forward(ad_t, out_t, u_t, ctx_t)  # (1, 52)
