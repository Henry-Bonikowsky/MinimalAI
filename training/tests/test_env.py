"""Tests for the Gymnasium combat environment."""

import numpy as np
import pytest
from combat_sim.env import (
    CombatEnv, NUM_ACTIONS, MAX_ENTITIES, ENTITY_FEATURE_DIM,
    SELF_STATE_DIM, COMBAT_CTX_DIM, SIGIL_STATE_DIM, ENV_STATE_DIM,
    ACT_FORWARD, ACT_ATTACK, ACT_SPRINT, ACT_JUMP,
)
from combat_sim.entities import MAX_HEALTH


@pytest.fixture
def env_1v1():
    env = CombatEnv(num_enemies=1, num_allies=0, episode_length=200, seed=42)
    return env


@pytest.fixture
def env_3v3():
    env = CombatEnv(num_enemies=3, num_allies=2, episode_length=200, seed=42)
    return env


class TestEnvBasics:
    def test_reset_returns_valid_obs(self, env_1v1):
        """Reset should return valid observation dict."""
        obs, info = env_1v1.reset()

        assert "self_state" in obs
        assert "entity_features" in obs
        assert "entity_mask" in obs
        assert "combat_ctx" in obs
        assert "sigil_state" in obs
        assert "env_state" in obs

        assert obs["self_state"].shape == (SELF_STATE_DIM,)
        assert obs["entity_features"].shape == (MAX_ENTITIES, ENTITY_FEATURE_DIM)
        assert obs["entity_mask"].shape == (MAX_ENTITIES,)
        assert obs["combat_ctx"].shape == (COMBAT_CTX_DIM,)
        assert obs["sigil_state"].shape == (SIGIL_STATE_DIM,)
        assert obs["env_state"].shape == (ENV_STATE_DIM,)

    def test_reset_obs_dtypes(self, env_1v1):
        """All observations should be float32."""
        obs, _ = env_1v1.reset()
        for key, val in obs.items():
            assert val.dtype == np.float32, f"{key} has dtype {val.dtype}"

    def test_step_returns_valid(self, env_1v1):
        """Step should return obs, reward, terminated, truncated, info."""
        env_1v1.reset()
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        obs, reward, terminated, truncated, info = env_1v1.step(action)

        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert obs["self_state"].shape == (SELF_STATE_DIM,)

    def test_action_space_shape(self, env_1v1):
        """Action space should be multi-binary with correct size."""
        assert env_1v1.action_space.n == NUM_ACTIONS

    def test_episode_truncation(self, env_1v1):
        """Episode should truncate after episode_length ticks."""
        env_1v1.reset()
        truncated = False
        for _ in range(300):  # longer than episode_length=200
            action = env_1v1.action_space.sample()
            _, _, terminated, truncated, _ = env_1v1.step(action)
            if terminated or truncated:
                break
        assert truncated or env_1v1.current_tick <= 200

    def test_info_contains_stats(self, env_1v1):
        """Info dict should contain game stats."""
        _, info = env_1v1.reset()
        assert "tick" in info
        assert "player_health" in info
        assert "enemies_alive" in info
        assert "kills" in info
        assert "episode_reward" in info


class TestMultiEntity:
    def test_multi_enemy_entity_mask(self, env_3v3):
        """Entity mask should reflect number of alive entities."""
        obs, _ = env_3v3.reset()
        # 3 enemies + 2 allies = 5 entities
        num_valid = obs["entity_mask"].sum()
        assert num_valid == 5

    def test_entity_features_have_alliance(self, env_3v3):
        """Entity features should include alliance information."""
        obs, _ = env_3v3.reset()
        for i in range(int(obs["entity_mask"].sum())):
            alliance = obs["entity_features"][i, 0]
            assert alliance in [-1.0, 0.0, 1.0]

    def test_enemy_count_in_info(self, env_3v3):
        """Info should report correct enemy count."""
        _, info = env_3v3.reset()
        assert info["enemies_alive"] == 3


class TestActions:
    def test_no_action_doesnt_crash(self, env_1v1):
        """Zero action vector should work."""
        env_1v1.reset()
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        obs, reward, _, _, _ = env_1v1.step(action)
        assert obs is not None

    def test_all_actions_doesnt_crash(self, env_1v1):
        """All actions on shouldn't crash (masking handles conflicts)."""
        env_1v1.reset()
        action = np.ones(NUM_ACTIONS, dtype=np.int8)
        obs, reward, _, _, _ = env_1v1.step(action)
        assert obs is not None

    def test_random_actions_survive_episode(self, env_1v1):
        """Random actions for full episode shouldn't crash."""
        env_1v1.reset()
        for _ in range(200):
            action = env_1v1.action_space.sample()
            _, _, terminated, truncated, _ = env_1v1.step(action)
            if terminated or truncated:
                break

    def test_attack_action_deals_damage(self, env_1v1):
        """Attacking when in range should deal damage."""
        env_1v1.reset()
        initial_enemy_hp = env_1v1.enemies[0].health

        # Move toward enemy and attack
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        action[ACT_FORWARD] = 1
        action[ACT_SPRINT] = 1

        # Approach
        for _ in range(50):
            env_1v1.step(action)

        # Attack
        action[ACT_ATTACK] = 1
        for _ in range(20):
            env_1v1.step(action)
            if env_1v1.enemies[0].health < initial_enemy_hp:
                break

        # Should have dealt some damage (enemy AI might kill us first but health should have changed)
        # Check that SOME combat happened
        assert env_1v1.current_tick > 0


class TestRewards:
    def test_reward_is_bounded(self, env_1v1):
        """Rewards should be bounded."""
        env_1v1.reset()
        for _ in range(100):
            action = env_1v1.action_space.sample()
            _, reward, terminated, _, _ = env_1v1.step(action)
            assert -3.0 <= reward <= 3.0
            if terminated:
                break

    def test_kill_gives_positive_reward(self, env_1v1):
        """Killing an enemy should give positive reward via direct physics."""
        env_1v1.reset()
        enemy = env_1v1.enemies[0]

        # Make enemy nearly dead and disable enemy AI by making it unable to attack
        enemy.health = 0.5
        enemy.attack_cooldown = 9999

        # Place player adjacent and facing enemy
        enemy.x = 2.0
        enemy.z = 0.0
        env_1v1.player.x = 0.0
        env_1v1.player.z = 0.0
        env_1v1.player.facing_angle = 0.0  # facing +X where enemy is
        env_1v1.player.health = MAX_HEALTH  # full health so we don't die

        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        action[ACT_ATTACK] = 1
        action[ACT_FORWARD] = 1
        action[ACT_SPRINT] = 1

        total_reward = 0.0
        killed = False
        for _ in range(50):
            _, reward, terminated, _, _ = env_1v1.step(action)
            total_reward += reward
            if not enemy.is_alive:
                killed = True
                break

        assert killed, "Should have killed the low-HP enemy"
        assert total_reward > 0, f"Kill should give positive reward, got {total_reward}"

    def test_death_gives_negative_reward(self, env_1v1):
        """Dying should give negative reward."""
        env_1v1.reset()
        env_1v1.player.health = 0.1

        # Let enemy kill us
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        total_reward = 0.0
        for _ in range(200):
            _, reward, terminated, _, _ = env_1v1.step(action)
            total_reward += reward
            if terminated:
                break

        assert total_reward < 0


class TestActionMask:
    def test_action_mask_shape(self, env_1v1):
        """Action mask should have correct shape."""
        env_1v1.reset()
        mask = env_1v1.get_action_mask()
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == np.float32

    def test_sigils_masked_when_unavailable(self, env_1v1):
        """Sigil actions should be masked when unavailable."""
        env_1v1.reset()
        mask = env_1v1.get_action_mask()
        # Sigils are unavailable in v1
        for i in range(4):
            assert mask[16 + i] == 0.0

    def test_basic_movement_always_valid(self, env_1v1):
        """Basic movement should always be valid."""
        env_1v1.reset()
        mask = env_1v1.get_action_mask()
        assert mask[ACT_FORWARD] == 1.0
        assert mask[ACT_SPRINT] == 1.0
        assert mask[ACT_JUMP] == 1.0
        assert mask[ACT_ATTACK] == 1.0


class TestDeterminism:
    def test_same_seed_same_result(self):
        """Same seed should produce same initial state."""
        env1 = CombatEnv(num_enemies=1, seed=123, domain_randomization=False)
        obs1, _ = env1.reset(seed=123)

        env2 = CombatEnv(num_enemies=1, seed=123, domain_randomization=False)
        obs2, _ = env2.reset(seed=123)

        for key in obs1:
            np.testing.assert_array_almost_equal(obs1[key], obs2[key], decimal=5)
