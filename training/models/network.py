"""Neural network with entity attention for combat AI.

Architecture:
  Per-Entity Features (up to 32 x 20)
    -> Entity Encoder: Linear(20, 64) + ReLU (shared weights)
    -> 2-Head Self-Attention (64 dim, 2 layers)
    -> Mean Pool over valid entities -> 64-dim entity summary

  [self_state(30) | entity_summary(64) | combat_ctx(22) | sigils(12) | env(8)]
    = 136 dims
    -> Linear(136, 256) + ReLU
    -> GRU(256, 128)
    -> Policy Head: Linear(128, 28) with action masking
    -> Value Head: Linear(128, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


SELF_STATE_DIM = 30
ENTITY_FEATURE_DIM = 20
COMBAT_CTX_DIM = 22
SIGIL_STATE_DIM = 12
ENV_STATE_DIM = 8
NUM_ACTIONS = 28
MAX_ENTITIES = 32

ENTITY_EMBED_DIM = 64
ATTENTION_HEADS = 2
ATTENTION_LAYERS = 2
HIDDEN_DIM = 256
GRU_HIDDEN_DIM = 128


class EntityAttention(nn.Module):
    """Processes variable-number entities through self-attention.

    Inspired by AlphaStar's entity encoder: shared MLP per entity,
    then multi-head self-attention to learn relationships between entities.
    """

    def __init__(self, input_dim=ENTITY_FEATURE_DIM, embed_dim=ENTITY_EMBED_DIM,
                 num_heads=ATTENTION_HEADS, num_layers=ATTENTION_LAYERS):
        super().__init__()

        # Shared entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,  # no dropout for RL
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_dim = embed_dim

    def forward(self, entity_features: torch.Tensor, entity_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entity_features: (batch, max_entities, feature_dim)
            entity_mask: (batch, max_entities) - 1.0 for valid entities, 0.0 for padding

        Returns:
            entity_summary: (batch, embed_dim) - pooled entity representation
        """
        batch_size = entity_features.shape[0]

        # Encode each entity independently (shared weights)
        encoded = self.entity_encoder(entity_features)  # (batch, max_entities, embed_dim)

        # Create attention mask: True means IGNORE this position
        # TransformerEncoder expects src_key_padding_mask where True = ignore
        attn_mask = (entity_mask == 0.0)  # (batch, max_entities)

        # Check if ALL entities are masked (no valid entities)
        all_masked = attn_mask.all(dim=1)  # (batch,)
        if all_masked.any():
            # For batches with no valid entities, unmask first slot to avoid NaN
            attn_mask[all_masked, 0] = False

        # Self-attention over entities
        attended = self.attention(encoded, src_key_padding_mask=attn_mask)

        # Mean pool over valid entities
        mask_expanded = entity_mask.unsqueeze(-1)  # (batch, max_entities, 1)
        pooled = (attended * mask_expanded).sum(dim=1)  # (batch, embed_dim)
        counts = mask_expanded.sum(dim=1).clamp(min=1.0)  # (batch, 1)
        entity_summary = pooled / counts  # (batch, embed_dim)

        return entity_summary


class CombatNetwork(nn.Module):
    """Full combat policy/value network.

    Entity attention processes variable entities into a fixed summary.
    Combined with self-state, combat context, sigil state, and env state.
    GRU provides temporal memory across ticks.
    Separate policy and value heads.
    """

    def __init__(self):
        super().__init__()

        # Entity attention module
        self.entity_attention = EntityAttention()

        # Combined input dimension
        combined_dim = (
            SELF_STATE_DIM +
            self.entity_attention.output_dim +
            COMBAT_CTX_DIM +
            SIGIL_STATE_DIM +
            ENV_STATE_DIM
        )  # 30 + 64 + 22 + 12 + 8 = 136

        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(combined_dim, HIDDEN_DIM),
            nn.ReLU(),
        )

        # GRU for temporal context
        self.gru = nn.GRU(
            input_size=HIDDEN_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )

        # Policy head (independent Bernoulli for each action)
        self.policy_head = nn.Linear(GRU_HIDDEN_DIM, NUM_ACTIONS)

        # Value head
        self.value_head = nn.Linear(GRU_HIDDEN_DIM, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (standard for PPO)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        # Policy head: small init for initial near-uniform distribution
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

        # Value head: unit gain
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: dict, hidden: torch.Tensor = None,
                action_mask: torch.Tensor = None):
        """Forward pass.

        Args:
            obs: Dict with keys matching CombatEnv observation space.
            hidden: GRU hidden state (1, batch, GRU_HIDDEN_DIM) or None.
            action_mask: (batch, NUM_ACTIONS) - 1.0 for valid, 0.0 for invalid.

        Returns:
            action_logits: (batch, NUM_ACTIONS)
            value: (batch, 1)
            hidden: updated GRU hidden state
        """
        self_state = obs["self_state"]       # (batch, 30)
        entity_features = obs["entity_features"]  # (batch, 32, 20)
        entity_mask = obs["entity_mask"]     # (batch, 32)
        combat_ctx = obs["combat_ctx"]       # (batch, 22)
        sigil_state = obs["sigil_state"]     # (batch, 12)
        env_state = obs["env_state"]         # (batch, 8)

        # Entity attention
        entity_summary = self.entity_attention(entity_features, entity_mask)  # (batch, 64)

        # Concatenate all features
        combined = torch.cat([
            self_state,
            entity_summary,
            combat_ctx,
            sigil_state,
            env_state,
        ], dim=-1)  # (batch, 136)

        # Feature processing
        features = self.feature_net(combined)  # (batch, 256)

        # GRU: expects (batch, seq_len, input_size)
        features = features.unsqueeze(1)  # (batch, 1, 256)

        if hidden is None:
            gru_out, hidden = self.gru(features)
        else:
            gru_out, hidden = self.gru(features, hidden)

        gru_out = gru_out.squeeze(1)  # (batch, 128)

        # Policy head
        action_logits = self.policy_head(gru_out)  # (batch, 28)

        # Apply action mask: set invalid actions to very negative logit
        if action_mask is not None:
            action_logits = action_logits + (1.0 - action_mask) * (-1e8)

        # Value head
        value = self.value_head(gru_out)  # (batch, 1)

        return action_logits, value, hidden

    def get_action_and_value(self, obs: dict, hidden: torch.Tensor = None,
                             action_mask: torch.Tensor = None,
                             action: torch.Tensor = None):
        """Get action, log prob, entropy, and value for PPO.

        Args:
            obs: Observation dict.
            hidden: GRU hidden state.
            action_mask: Valid action mask.
            action: If provided, evaluate this action instead of sampling.

        Returns:
            action: (batch, NUM_ACTIONS) sampled or provided
            log_prob: (batch,) sum of log probs for all action dims
            entropy: (batch,) sum of entropy for all action dims
            value: (batch,)
            hidden: updated hidden state
        """
        logits, value, hidden = self.forward(obs, hidden, action_mask)

        # Independent Bernoulli per action dimension
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs=probs)

        if action is None:
            action = dist.sample()

        # Apply action mask to sampled actions
        if action_mask is not None:
            action = action * action_mask

        log_prob = dist.log_prob(action).sum(dim=-1)  # (batch,)
        entropy = dist.entropy().sum(dim=-1)  # (batch,)

        return action, log_prob, entropy, value.squeeze(-1), hidden

    def get_value(self, obs: dict, hidden: torch.Tensor = None):
        """Get value estimate only (for GAE computation)."""
        _, value, hidden = self.forward(obs, hidden)
        return value.squeeze(-1), hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize GRU hidden state."""
        return torch.zeros(1, batch_size, GRU_HIDDEN_DIM)


# Need numpy for init
import numpy as np
