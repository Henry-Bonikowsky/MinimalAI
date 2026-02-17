"""Neural network with entity attention for combat AI (v2).

Architecture:
  Per-Entity Features (up to 32 x 24)
    -> Entity Encoder: Linear(24, 64) + ReLU (shared weights)
    -> 2-Head Self-Attention (64 dim, 2 layers)
    -> Mean Pool over valid entities -> 64-dim entity summary

  [self_state(38) | entity_summary(64) | combat_ctx(26) | sigils(48) | env(8)]
    = 184 dims
    -> Linear(184, 256) + ReLU
    -> GRU(256, 128)
    -> Policy Head: Linear(128, 35) with action masking
    -> Value Head: Linear(128, 1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


SELF_STATE_DIM = 38
ENTITY_FEATURE_DIM = 24
COMBAT_CTX_DIM = 26
SIGIL_STATE_DIM = 48   # 12 slots * 4 dims
ENV_STATE_DIM = 8
NUM_ACTIONS = 35
MAX_ENTITIES = 32

ENTITY_EMBED_DIM = 64
ATTENTION_HEADS = 2
ATTENTION_LAYERS = 2
HIDDEN_DIM = 256
GRU_HIDDEN_DIM = 128


class EntityAttention(nn.Module):
    """Processes variable-number entities through self-attention."""

    def __init__(self, input_dim=ENTITY_FEATURE_DIM, embed_dim=ENTITY_EMBED_DIM,
                 num_heads=ATTENTION_HEADS, num_layers=ATTENTION_LAYERS):
        super().__init__()

        self.entity_encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_dim = embed_dim

    def forward(self, entity_features: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entity_features: (batch, max_entities, feature_dim)
            entity_mask: (batch, max_entities) - 1.0 for valid, 0.0 for padding

        Returns:
            entity_summary: (batch, embed_dim)
        """
        encoded = self.entity_encoder(entity_features)

        attn_mask = (entity_mask == 0.0)
        all_masked = attn_mask.all(dim=1)
        if all_masked.any():
            attn_mask[all_masked, 0] = False

        attended = self.attention(encoded, src_key_padding_mask=attn_mask)

        mask_expanded = entity_mask.unsqueeze(-1)
        pooled = (attended * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)
        entity_summary = pooled / counts

        return entity_summary


class CombatNetwork(nn.Module):
    """Full combat policy/value network (v2 - 1.8 PvP)."""

    def __init__(self):
        super().__init__()

        self.entity_attention = EntityAttention()

        combined_dim = (
            SELF_STATE_DIM +
            self.entity_attention.output_dim +
            COMBAT_CTX_DIM +
            SIGIL_STATE_DIM +
            ENV_STATE_DIM
        )  # 38 + 64 + 26 + 48 + 8 = 184

        self.feature_net = nn.Sequential(
            nn.Linear(combined_dim, HIDDEN_DIM),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=HIDDEN_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )

        self.policy_head = nn.Linear(GRU_HIDDEN_DIM, NUM_ACTIONS)
        self.value_head = nn.Linear(GRU_HIDDEN_DIM, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: dict, hidden: torch.Tensor = None,
                action_mask: torch.Tensor = None):
        self_state = obs["self_state"]
        entity_features = obs["entity_features"]
        entity_mask = obs["entity_mask"]
        combat_ctx = obs["combat_ctx"]
        sigil_state = obs["sigil_state"]
        env_state = obs["env_state"]

        entity_summary = self.entity_attention(entity_features, entity_mask)

        combined = torch.cat([
            self_state,
            entity_summary,
            combat_ctx,
            sigil_state,
            env_state,
        ], dim=-1)

        features = self.feature_net(combined)
        features = features.unsqueeze(1)

        if hidden is None:
            gru_out, hidden = self.gru(features)
        else:
            gru_out, hidden = self.gru(features, hidden)

        gru_out = gru_out.squeeze(1)

        action_logits = self.policy_head(gru_out)

        if action_mask is not None:
            action_logits = action_logits + (1.0 - action_mask) * (-1e8)

        value = self.value_head(gru_out)

        return action_logits, value, hidden

    def get_action_and_value(self, obs: dict, hidden: torch.Tensor = None,
                             action_mask: torch.Tensor = None,
                             action: torch.Tensor = None):
        logits, value, hidden = self.forward(obs, hidden, action_mask)

        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs=probs)

        if action is None:
            action = dist.sample()

        if action_mask is not None:
            action = action * action_mask

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value.squeeze(-1), hidden

    def get_value(self, obs: dict, hidden: torch.Tensor = None):
        _, value, hidden = self.forward(obs, hidden)
        return value.squeeze(-1), hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(1, batch_size, GRU_HIDDEN_DIM)
