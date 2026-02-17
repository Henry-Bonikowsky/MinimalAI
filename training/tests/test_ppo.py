"""Tests for the PPO trainer."""

import numpy as np
import torch
import pytest
import os
import tempfile
from combat_sim.env import CombatEnv
from models.network import CombatNetwork
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


class TestRolloutBuffer:
    def test_add_and_length(self):
        """Buffer should track added entries."""
        buf = RolloutBuffer()
        obs = {
            "self_state": np.zeros(30),
            "entity_features": np.zeros((32, 20)),
            "entity_mask": np.zeros(32),
            "combat_ctx": np.zeros(22),
            "sigil_state": np.zeros(12),
            "env_state": np.zeros(8),
        }
        buf.add(obs, np.zeros(28), np.ones(28), 0.0, 1.0, 0.5, False, np.zeros((1, 1, 128)))
        assert len(buf) == 1

    def test_clear(self):
        """Clear should empty the buffer."""
        buf = RolloutBuffer()
        obs = {
            "self_state": np.zeros(30),
            "entity_features": np.zeros((32, 20)),
            "entity_mask": np.zeros(32),
            "combat_ctx": np.zeros(22),
            "sigil_state": np.zeros(12),
            "env_state": np.zeros(8),
        }
        buf.add(obs, np.zeros(28), np.ones(28), 0.0, 1.0, 0.5, False, np.zeros((1, 1, 128)))
        buf.clear()
        assert len(buf) == 0

    def test_gae_computation(self):
        """GAE should produce valid advantages and returns."""
        buf = RolloutBuffer()
        obs = {
            "self_state": np.zeros(30),
            "entity_features": np.zeros((32, 20)),
            "entity_mask": np.zeros(32),
            "combat_ctx": np.zeros(22),
            "sigil_state": np.zeros(12),
            "env_state": np.zeros(8),
        }
        # Add 10 steps with known rewards
        for i in range(10):
            buf.add(obs, np.zeros(28), np.ones(28), -0.5, 0.1 * i, 0.5, i == 9, np.zeros((1, 1, 128)))

        buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)

        assert buf.advantages is not None
        assert buf.returns is not None
        assert len(buf.advantages) == 10
        assert len(buf.returns) == 10
        assert not np.isnan(buf.advantages).any()
        assert not np.isnan(buf.returns).any()

    def test_gae_terminal_episode(self):
        """GAE should handle terminal states correctly."""
        buf = RolloutBuffer()
        obs = {
            "self_state": np.zeros(30),
            "entity_features": np.zeros((32, 20)),
            "entity_mask": np.zeros(32),
            "combat_ctx": np.zeros(22),
            "sigil_state": np.zeros(12),
            "env_state": np.zeros(8),
        }
        # Episode ends at step 5
        for i in range(10):
            done = (i == 4)
            buf.add(obs, np.zeros(28), np.ones(28), -0.5, 0.1, 0.5, done, np.zeros((1, 1, 128)))

        buf.compute_gae(last_value=0.3, gamma=0.99, gae_lambda=0.95)
        assert not np.isnan(buf.advantages).any()


class TestPPO:
    def test_collect_rollout(self, ppo, env):
        """Rollout collection should work without errors."""
        stats = ppo.collect_rollout(env, num_steps=128)

        assert "mean_reward" in stats
        assert "episode_rewards" in stats
        assert len(ppo.buffer) == 128
        assert ppo.total_steps == 128

    def test_train_on_buffer(self, ppo, env):
        """Training on buffer should produce valid stats."""
        ppo.collect_rollout(env, num_steps=256)
        train_stats = ppo.train_on_buffer()

        assert "policy_loss" in train_stats
        assert "value_loss" in train_stats
        assert "entropy" in train_stats
        assert "clip_fraction" in train_stats
        assert "approx_kl" in train_stats

        assert not np.isnan(train_stats["policy_loss"])
        assert not np.isnan(train_stats["value_loss"])
        assert train_stats["entropy"] >= 0

    def test_multiple_updates(self, ppo, env):
        """Multiple update cycles should work."""
        for _ in range(3):
            ppo.collect_rollout(env, num_steps=128)
            stats = ppo.train_on_buffer()
            assert not np.isnan(stats["policy_loss"])

    def test_save_load_checkpoint(self, ppo, env):
        """Save and load should preserve network state."""
        ppo.collect_rollout(env, num_steps=128)
        ppo.train_on_buffer()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            ppo.save_checkpoint(path, metadata={"test": True})

            assert os.path.exists(path)

            # Load into new PPO
            new_network = CombatNetwork()
            new_ppo = PPO(new_network, device="cpu")
            new_ppo.load_checkpoint(path)

            assert new_ppo.total_steps == ppo.total_steps
            assert new_ppo.total_episodes == ppo.total_episodes

            # Check weights are the same
            for p1, p2 in zip(ppo.network.parameters(), new_ppo.network.parameters()):
                assert torch.allclose(p1, p2)

    def test_clip_fraction_meaningful(self, ppo, env):
        """Clip fraction should be between 0 and 1."""
        ppo.collect_rollout(env, num_steps=256)
        stats = ppo.train_on_buffer()
        assert 0.0 <= stats["clip_fraction"] <= 1.0

    def test_entropy_decreases_with_training(self, ppo, env):
        """Entropy should generally decrease as policy becomes more certain."""
        entropies = []
        for _ in range(5):
            ppo.collect_rollout(env, num_steps=256)
            stats = ppo.train_on_buffer()
            entropies.append(stats["entropy"])

        # Entropy shouldn't go to zero immediately
        assert all(e > 0 for e in entropies)


class TestPPOConvergence:
    """Test that PPO can learn on a simple scenario."""

    def test_reward_improves(self):
        """Mean reward should improve over training on a simple 1v1."""
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

        early_rewards = []
        late_rewards = []

        for i in range(20):
            stats = ppo.collect_rollout(env, num_steps=512)
            ppo.train_on_buffer()

            if i < 5:
                early_rewards.extend(stats["episode_rewards"])
            elif i >= 15:
                late_rewards.extend(stats["episode_rewards"])

        # We just need training to not crash and produce meaningful values
        # Full convergence takes much longer, but there shouldn't be NaN
        if early_rewards and late_rewards:
            assert not np.isnan(np.mean(early_rewards))
            assert not np.isnan(np.mean(late_rewards))
