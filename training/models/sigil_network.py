"""Lightweight sigil timing network.

Small reactive network for deciding when to activate sigil abilities.
No GRU needed — sigil decisions are based on current state snapshot.

Architecture:
  obs(20) → Linear(20, 32) + ReLU → Linear(32, 32) + ReLU
    → Policy: Linear(32, 7) + sigmoid (with mask)
    → Value: Linear(32, 1)

~5K parameters total.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


SIGIL_OBS_DIM = 16
SIGIL_NUM_ACTIONS = 4
SIGIL_HIDDEN_DIM = 32


class SigilNet(nn.Module):
    """Policy/value network for sigil ability timing."""

    def __init__(self, obs_dim=SIGIL_OBS_DIM, hidden_dim=SIGIL_HIDDEN_DIM,
                 num_actions=SIGIL_NUM_ACTIONS):
        super().__init__()
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            obs: (batch, 20)
            mask: (batch, 7) — 1.0 for usable abilities, 0.0 for on cooldown

        Returns:
            probs: (batch, 7) — sigmoid probabilities (masked)
            value: (batch, 1)
        """
        x = self.net(obs)
        logits = self.policy(x)

        if mask is not None:
            logits = logits + (1.0 - mask) * (-1e8)

        probs = torch.sigmoid(logits)
        value = self.value(x)

        return probs, value

    def get_action_and_value(self, obs, mask=None, action=None):
        """PPO-compatible action sampling."""
        probs, value = self.forward(obs, mask)
        dist = Bernoulli(probs=probs)

        if action is None:
            action = dist.sample()

        if mask is not None:
            action = action * mask

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs, mask=None):
        _, value = self.forward(obs, mask)
        return value.squeeze(-1)


class SigilInferenceWrapper(nn.Module):
    """TorchScript-compatible wrapper for Java-side inference.

    Input: obs(1, 20), mask(1, 7)
    Output: probs(1, 7), value(1, 1)
    """

    def __init__(self, model: SigilNet):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        probs, value = self.model(obs, mask)
        return probs, value
