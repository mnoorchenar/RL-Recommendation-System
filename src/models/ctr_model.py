"""
CTR / CVR prediction model.

Given a user state vector and an ad embedding, predicts:
  P(click | user, ad)       →  CTR estimate
  P(convert | click, ad)    →  CVR estimate

Used at inference time to annotate recommended ads with human-interpretable
click and conversion probability estimates.  The DQN Q-values handle the
*long-term* reward signal; the CTR model provides *immediate* interpretability.
"""
import torch
import torch.nn as nn


class CTRModel(nn.Module):
    def __init__(self, state_dim: int, n_ads: int, ad_embed_dim: int = 16):
        super().__init__()
        self.ad_emb = nn.Embedding(n_ads, ad_embed_dim)

        in_dim = state_dim + ad_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 2),      # [logit_ctr, logit_cvr]
        )

    def forward(self, state: torch.Tensor, ad_ids: torch.Tensor) -> torch.Tensor:
        """
        state  : (batch, state_dim)
        ad_ids : (batch,)  or  (1,) broadcast
        returns: (batch, 2) — [p_click, p_convert]
        """
        ad_feat = self.ad_emb(ad_ids)
        x       = torch.cat([state.expand(len(ad_ids), -1), ad_feat], dim=-1)
        return torch.sigmoid(self.net(x))

    @torch.no_grad()
    def predict(self, state: torch.Tensor, ad_ids: list) -> list:
        """Return list of (p_click, p_convert) for each ad_id."""
        ids = torch.tensor(ad_ids, dtype=torch.long)
        out = self.forward(state, ids)         # (n_ads, 2)
        return [(float(row[0]), float(row[1])) for row in out]
