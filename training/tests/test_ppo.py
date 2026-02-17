"""Tests for the PPO trainer (v2)."""

import numpy as np
import torch
import pytest
import os
import tempfile
from combat_sim.env import CombatEnv, MAX_ENTITIES, ENTITY_FEATURE_DIM
from models.network import CombatNetwork, SELF_STATE_DIM, COMBAT_CTX_DIM, SIGIL_STATE_DIM, ENV_STATE_DIM, NUM_ACTIONS, GRU_HIDDEN_DIM
from models.ppo import PPO, RolloutBuffer


@pytest.fixture
def env():
    return CombatEnv(num_enemies=1, episode_length=100, seed=42, domain_randomization=False)


@pytest.fixture
def ppo(env):
    network = CombatNetwork()
    return PPO(
        network=network,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=2,
        minibatch_size=64,
        device="cpu",
    )


OBS_SHAPES = {
    "self_state": (SELF_STATE_DIM,),
    "entity_features": (MAX_ENTITIES, ENTITY_FEATURE_DIM),
    "entity_mask": (MAX_ENTITIES,),
    "combat_ctx": (COMBAT_CTX_DIM,),
    "sigil_state": (SIGIL_STATE_DIM,),
    "env_state": (ENV_STATE_DIM,),
}


def _make_obs():
    return {k: np.zeros(v, dtype=np.float32) for k, v in OBS_SHAPES.items()}


class TestRolloutBuffer:
    def test_add_and_length(self):
        buf = RolloutBuffer(capacity=64, obs_shapes=OBS_SHAPES)
        buf.add(_make_obs(), np.zeros(NUM_ACTIONS), np.ones(NUM_ACTIONS), 0.0, 1.0, 0.5, False, None)
        assert len(buf) == 1

    def test_clear(self):
        buf = RolloutBuffer(capacity=64, obs_shapes=OBS_SHAPES)
        buf.add(_make_obs(), np.zeros(NUM_ACTIONS), np.ones(NUM_ACTIONS), 0.0, 1.0, 0.5, False, None)
        buf.clear()
        assert len(buf) == 0

    def test_gae_computation(self):
        buf = RolloutBuffer(capacity=64, obs_shapes=OBS_SHAPES)
        for i in range(10):
            buf.add(_make_obs(), np.zeros(NUM_ACTIONS), np.ones(NUM_ACTIONS), -0.5, 0.1 * i, 0.5, i == 9, None)

        buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        assert buf.advantages is not None
        assert buf.returns is not None
        assert len(buf) == 10
        assert not np.isnan(buf.advantages[:10]).any()

    def test_gae_terminal_episode(self):
        buf = RolloutBuffer(capacity=64, obs_shapes=OBS_SHAPES)
        for i in range(10):
            done = (i == 4)
            buf.add(_make_obs(), np.zeros(NUM_ACTIONS), np.ones(NUM_ACTIONS), -0.5, 0.1, 0.5, done, None)

        buf.compute_gae(last_value=0.3, gamma=0.99, gae_lambda=0.95)
        assert not np.isnan(buf.advantages[:10]).any()


class TestPPO:
    def test_collect_rollout(self, ppo, env):
        stats = ppo.collect_rollout(env, num_steps=128)
        assert "mean_reward" in stats
        assert len(ppo.buffer) == 128
        assert ppo.total_steps == 128

    def test_train_on_buffer(self, ppo, env):
        ppo.collect_rollout(env, num_steps=256)
        train_stats = ppo.train_on_buffer()

        assert "policy_loss" in train_stats
        assert "value_loss" in train_stats
        assert "entropy" in train_stats
        assert not np.isnan(train_stats["policy_loss"])
        assert not np.isnan(train_stats["value_loss"])
        assert train_stats["entropy"] >= 0

    def test_multiple_updates(self, ppo, env):
        for _ in range(3):
            ppo.collect_rollout(env, num_steps=128)
            stats = ppo.train_on_buffer()
            assert not np.isnan(stats["policy_loss"])

    def test_save_load_checkpoint(self, ppo, env):
        ppo.collect_rollout(env, num_steps=128)
        ppo.train_on_buffer()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            ppo.save_checkpoint(path, metadata={"test": True})
            assert os.path.exists(path)

            new_network = CombatNetwork()
            new_ppo = PPO(new_network, device="cpu")
            new_ppo.load_checkpoint(path)

            assert new_ppo.total_steps == ppo.total_steps
            for p1, p2 in zip(ppo.network.parameters(), new_ppo.network.parameters()):
                assert torch.allclose(p1, p2)

    def test_clip_fraction_meaningful(self, ppo, env):
        ppo.collect_rollout(env, num_steps=256)
        stats = ppo.train_on_buffer()
        assert 0.0 <= stats["clip_fraction"] <= 1.0

    def test_entropy_positive(self, ppo, env):
        entropies = []
        for _ in range(3):
            ppo.collect_rollout(env, num_steps=256)
            stats = ppo.train_on_buffer()
            entropies.append(stats["entropy"])
        assert all(e > 0 for e in entropies)


class TestPPOConvergence:
    def test_reward_no_nan(self):
        """Training should not produce NaN rewards."""
        env = CombatEnv(num_enemies=1, episode_length=200, seed=42, domain_randomization=False)
        network = CombatNetwork()
        ppo = PPO(
            network=network,
            lr=3e-4,
            gamma=0.99,
            ppo_epochs=4,
            minibatch_size=64,
            device="cpu",
        )

        for i in range(10):
            stats = ppo.collect_rollout(env, num_steps=512)
            ppo.train_on_buffer()
            if stats["episode_rewards"]:
                assert not np.isnan(np.mean(stats["episode_rewards"]))
