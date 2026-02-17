"""Tests for the neural network architecture (v2)."""

import torch
import numpy as np
import pytest
from models.network import (
    CombatNetwork, EntityAttention, NUM_ACTIONS, GRU_HIDDEN_DIM,
    ENTITY_EMBED_DIM, MAX_ENTITIES, ENTITY_FEATURE_DIM,
    SELF_STATE_DIM, COMBAT_CTX_DIM, SIGIL_STATE_DIM, ENV_STATE_DIM,
)


@pytest.fixture
def network():
    return CombatNetwork()


@pytest.fixture
def batch_obs():
    batch_size = 4
    return {
        "self_state": torch.randn(batch_size, SELF_STATE_DIM),
        "entity_features": torch.randn(batch_size, MAX_ENTITIES, ENTITY_FEATURE_DIM),
        "entity_mask": torch.zeros(batch_size, MAX_ENTITIES),
        "combat_ctx": torch.randn(batch_size, COMBAT_CTX_DIM),
        "sigil_state": torch.randn(batch_size, SIGIL_STATE_DIM),
        "env_state": torch.randn(batch_size, ENV_STATE_DIM),
    }


@pytest.fixture
def single_obs():
    return {
        "self_state": torch.randn(1, SELF_STATE_DIM),
        "entity_features": torch.randn(1, MAX_ENTITIES, ENTITY_FEATURE_DIM),
        "entity_mask": torch.zeros(1, MAX_ENTITIES),
        "combat_ctx": torch.randn(1, COMBAT_CTX_DIM),
        "sigil_state": torch.randn(1, SIGIL_STATE_DIM),
        "env_state": torch.randn(1, ENV_STATE_DIM),
    }


class TestEntityAttention:
    def test_output_shape(self):
        attn = EntityAttention()
        features = torch.randn(4, MAX_ENTITIES, ENTITY_FEATURE_DIM)
        mask = torch.zeros(4, MAX_ENTITIES)
        mask[:, :5] = 1.0
        summary = attn(features, mask)
        assert summary.shape == (4, ENTITY_EMBED_DIM)

    def test_handles_zero_entities(self):
        attn = EntityAttention()
        features = torch.randn(2, MAX_ENTITIES, ENTITY_FEATURE_DIM)
        mask = torch.zeros(2, MAX_ENTITIES)
        summary = attn(features, mask)
        assert summary.shape == (2, ENTITY_EMBED_DIM)
        assert not torch.isnan(summary).any()

    def test_handles_variable_entity_count(self):
        attn = EntityAttention()
        features = torch.randn(3, MAX_ENTITIES, ENTITY_FEATURE_DIM)
        mask = torch.zeros(3, MAX_ENTITIES)
        mask[0, :2] = 1.0
        mask[1, :10] = 1.0
        mask[2, :1] = 1.0
        summary = attn(features, mask)
        assert summary.shape == (3, ENTITY_EMBED_DIM)
        assert not torch.isnan(summary).any()

    def test_handles_max_entities(self):
        attn = EntityAttention()
        features = torch.randn(2, MAX_ENTITIES, ENTITY_FEATURE_DIM)
        mask = torch.ones(2, MAX_ENTITIES)
        summary = attn(features, mask)
        assert summary.shape == (2, ENTITY_EMBED_DIM)
        assert not torch.isnan(summary).any()


class TestCombatNetwork:
    def test_forward_shapes(self, network, batch_obs):
        batch_obs["entity_mask"][:, :3] = 1.0
        logits, value, hidden = network(batch_obs)
        assert logits.shape == (4, NUM_ACTIONS)
        assert value.shape == (4, 1)
        assert hidden.shape == (1, 4, GRU_HIDDEN_DIM)

    def test_forward_no_nan(self, network, batch_obs):
        batch_obs["entity_mask"][:, :3] = 1.0
        logits, value, hidden = network(batch_obs)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(value).any()
        assert not torch.isnan(hidden).any()

    def test_get_action_and_value(self, network, single_obs):
        single_obs["entity_mask"][:, :1] = 1.0
        action_mask = torch.ones(1, NUM_ACTIONS)

        action, log_prob, entropy, value, hidden = network.get_action_and_value(
            single_obs, action_mask=action_mask
        )
        assert action.shape == (1, NUM_ACTIONS)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)
        assert hidden.shape == (1, 1, GRU_HIDDEN_DIM)

    def test_action_mask_zeroes_invalid(self, network, single_obs):
        single_obs["entity_mask"][:, :1] = 1.0
        action_mask = torch.ones(1, NUM_ACTIONS)
        action_mask[0, 14:26] = 0.0  # mask sigils

        action, _, _, _, _ = network.get_action_and_value(
            single_obs, action_mask=action_mask
        )
        assert action[0, 14:26].sum() == 0

    def test_hidden_state_persists(self, network, single_obs):
        single_obs["entity_mask"][:, :1] = 1.0
        hidden = network.init_hidden(1)
        _, _, hidden1 = network(single_obs, hidden)
        assert not torch.allclose(hidden1, hidden)

    def test_init_hidden(self, network):
        hidden = network.init_hidden(batch_size=8)
        assert hidden.shape == (1, 8, GRU_HIDDEN_DIM)
        assert (hidden == 0).all()

    def test_parameter_count(self, network):
        total_params = sum(p.numel() for p in network.parameters())
        assert 50_000 < total_params < 500_000

    def test_get_value(self, network, single_obs):
        single_obs["entity_mask"][:, :1] = 1.0
        value, hidden = network.get_value(single_obs)
        assert value.shape == (1,)

    def test_gradients_flow(self, network, batch_obs):
        batch_obs["entity_mask"][:, :3] = 1.0
        action_mask = torch.ones(4, NUM_ACTIONS)

        action, log_prob, entropy, value, _ = network.get_action_and_value(
            batch_obs, action_mask=action_mask
        )
        loss = -log_prob.mean() + value.mean()
        loss.backward()

        assert network.entity_attention.entity_encoder[0].weight.grad is not None
        assert network.feature_net[0].weight.grad is not None
        assert network.policy_head.weight.grad is not None
        assert network.value_head.weight.grad is not None

    def test_initial_policy_near_uniform(self, network, single_obs):
        single_obs["entity_mask"][:, :1] = 1.0
        logits, _, _ = network(single_obs)
        probs = torch.sigmoid(logits)
        assert (probs > 0.3).all() and (probs < 0.7).all()

    def test_v2_dimensions(self):
        """Verify v2 dimension constants are correct."""
        assert SELF_STATE_DIM == 38
        assert ENTITY_FEATURE_DIM == 24
        assert COMBAT_CTX_DIM == 26
        assert SIGIL_STATE_DIM == 48
        assert NUM_ACTIONS == 35
