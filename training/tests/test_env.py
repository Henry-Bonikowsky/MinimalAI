"""Tests for the 1.8 PvP combat environment."""

import numpy as np
import pytest
from combat_sim.env import (
    CombatEnv, NUM_ACTIONS, MAX_ENTITIES, ENTITY_FEATURE_DIM,
    SELF_STATE_DIM, COMBAT_CTX_DIM, SIGIL_STATE_DIM, ENV_STATE_DIM,
    ACT_FORWARD, ACT_ATTACK, ACT_SPRINT, ACT_JUMP,
    ACT_EAT_GAP, ACT_THROW_POT, ACT_THROW_PEARL,
    ACT_BLOCK, ACT_SPRINT_RESET, ACT_SWAP_WEAPON,
    ACT_SIGIL_0, ACT_TARGET_0,
)
from combat_sim.entities import MAX_HEALTH, WeaponType


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
        obs, _ = env_1v1.reset()
        for key, val in obs.items():
            assert val.dtype == np.float32, f"{key} has dtype {val.dtype}"

    def test_step_returns_valid(self, env_1v1):
        env_1v1.reset()
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        obs, reward, terminated, truncated, info = env_1v1.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert obs["self_state"].shape == (SELF_STATE_DIM,)

    def test_action_space_shape(self, env_1v1):
        assert env_1v1.action_space.n == NUM_ACTIONS

    def test_episode_truncation(self, env_1v1):
        env_1v1.reset()
        truncated = False
        for _ in range(300):
            action = env_1v1.action_space.sample()
            _, _, terminated, truncated, _ = env_1v1.step(action)
            if terminated or truncated:
                break
        assert truncated or env_1v1.current_tick <= 200

    def test_info_contains_stats(self, env_1v1):
        _, info = env_1v1.reset()
        assert "tick" in info
        assert "player_health" in info
        assert "enemies_alive" in info
        assert "kills" in info
        assert "gaps_remaining" in info
        assert "pots_remaining" in info


class TestMultiEntity:
    def test_multi_enemy_entity_mask(self, env_3v3):
        obs, _ = env_3v3.reset()
        num_valid = obs["entity_mask"].sum()
        assert num_valid == 5  # 3 enemies + 2 allies

    def test_entity_features_have_alliance(self, env_3v3):
        obs, _ = env_3v3.reset()
        for i in range(int(obs["entity_mask"].sum())):
            alliance = obs["entity_features"][i, 0]
            assert alliance in [-1.0, 0.0, 1.0]

    def test_enemy_count_in_info(self, env_3v3):
        _, info = env_3v3.reset()
        assert info["enemies_alive"] == 3


class TestActions:
    def test_no_action_doesnt_crash(self, env_1v1):
        env_1v1.reset()
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        obs, reward, _, _, _ = env_1v1.step(action)
        assert obs is not None

    def test_all_actions_doesnt_crash(self, env_1v1):
        env_1v1.reset()
        action = np.ones(NUM_ACTIONS, dtype=np.int8)
        obs, reward, _, _, _ = env_1v1.step(action)
        assert obs is not None

    def test_random_actions_survive_episode(self, env_1v1):
        env_1v1.reset()
        for _ in range(200):
            action = env_1v1.action_space.sample()
            _, _, terminated, truncated, _ = env_1v1.step(action)
            if terminated or truncated:
                break

    def test_swap_weapon(self, env_1v1):
        """SWAP_WEAPON should toggle sword <-> axe."""
        env_1v1.reset()
        assert env_1v1.player.weapon == WeaponType.SWORD
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        action[ACT_SWAP_WEAPON] = 1
        env_1v1.step(action)
        assert env_1v1.player.weapon == WeaponType.AXE

    def test_eat_gap_action(self, env_1v1):
        """EAT_GAP should start eating."""
        env_1v1.reset()
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        action[ACT_EAT_GAP] = 1
        env_1v1.step(action)
        assert env_1v1.player.kit.is_eating

    def test_throw_pot_action(self, env_1v1):
        """THROW_POT should heal player."""
        env_1v1.reset()
        env_1v1.player.health = 10.0
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        action[ACT_THROW_POT] = 1
        env_1v1.step(action)
        assert env_1v1.player.health > 10.0

    def test_sprint_reset_action(self, env_1v1):
        """SPRINT_RESET should toggle sprint state."""
        env_1v1.reset()
        env_1v1.player.is_sprinting = True
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        action[ACT_SPRINT_RESET] = 1
        action[ACT_SPRINT] = 1  # re-sprint immediately
        env_1v1.step(action)
        # After sprint reset + re-sprint, should be sprinting again
        assert env_1v1.player.is_sprinting

    def test_blockhit(self, env_1v1):
        """BLOCK + ATTACK should both be active (1.8 blockhit)."""
        env_1v1.reset()
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        action[ACT_BLOCK] = 1
        action[ACT_ATTACK] = 1
        env_1v1.step(action)
        # Player should be blocking (1.8 allows block + attack simultaneously)
        assert env_1v1.player.is_blocking


class TestKitItems:
    def test_initial_kit_counts(self, env_1v1):
        """Kit should start with correct item counts."""
        env_1v1.reset()
        from combat_sim.entities import KIT_GOLDEN_APPLES, KIT_HEALTH_POTS, KIT_ENDER_PEARLS, KIT_TOTEMS
        assert env_1v1.player.kit.golden_apples == KIT_GOLDEN_APPLES
        assert env_1v1.player.kit.health_pots == KIT_HEALTH_POTS
        assert env_1v1.player.kit.ender_pearls == KIT_ENDER_PEARLS
        assert env_1v1.player.kit.totems == KIT_TOTEMS


class TestRewards:
    def test_reward_is_bounded(self, env_1v1):
        env_1v1.reset()
        for _ in range(100):
            action = env_1v1.action_space.sample()
            _, reward, terminated, _, _ = env_1v1.step(action)
            assert -6.0 <= reward <= 6.0
            if terminated:
                break

    def test_kill_gives_positive_reward(self, env_1v1):
        env_1v1.reset()
        enemy = env_1v1.enemies[0]
        enemy.health = 0.5
        enemy.attack_tick_cooldown = 9999
        enemy.armor = 0.0
        enemy.armor_toughness = 0.0
        enemy.protection_epf = 0
        enemy.kit.totems = 0

        enemy.x = 2.0
        enemy.z = 0.0
        env_1v1.player.x = 0.0
        env_1v1.player.z = 0.0
        env_1v1.player.facing_angle = 0.0
        env_1v1.player.health = MAX_HEALTH

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

        assert killed
        assert total_reward > 0

    def test_death_gives_negative_reward(self, env_1v1):
        env_1v1.reset()
        env_1v1.player.health = 0.1
        env_1v1.player.kit.totems = 0  # no totem save

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
        env_1v1.reset()
        mask = env_1v1.get_action_mask()
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == np.float32

    def test_basic_movement_always_valid(self, env_1v1):
        env_1v1.reset()
        mask = env_1v1.get_action_mask()
        assert mask[ACT_FORWARD] == 1.0
        assert mask[ACT_SPRINT] == 1.0
        assert mask[ACT_JUMP] == 1.0
        assert mask[ACT_ATTACK] == 1.0

    def test_pot_masked_when_empty(self, env_1v1):
        env_1v1.reset()
        env_1v1.player.kit.health_pots = 0
        mask = env_1v1.get_action_mask()
        assert mask[ACT_THROW_POT] == 0.0

    def test_pearl_masked_on_cooldown(self, env_1v1):
        env_1v1.reset()
        env_1v1.player.kit.pearl_cooldown = 10
        mask = env_1v1.get_action_mask()
        assert mask[ACT_THROW_PEARL] == 0.0

    def test_sprint_reset_masked_when_not_sprinting(self, env_1v1):
        env_1v1.reset()
        env_1v1.player.is_sprinting = False
        mask = env_1v1.get_action_mask()
        assert mask[ACT_SPRINT_RESET] == 0.0

    def test_block_masked_when_holding_axe(self, env_1v1):
        env_1v1.reset()
        env_1v1.player.weapon = WeaponType.AXE
        mask = env_1v1.get_action_mask()
        assert mask[ACT_BLOCK] == 0.0


class TestSigils:
    def test_sigils_randomized_on_reset(self, env_1v1):
        """Sigil loadout should be randomized each episode."""
        env_1v1.reset()
        equipped = [s for s in env_1v1.player.sigil_slots if s.equipped]
        # Should have at least weapon sigils (4) + some armor
        assert len(equipped) >= 4

    def test_weapon_sigils_always_equipped(self, env_1v1):
        """Weapon sigils (slots 4, 5, 9, 10) should always be equipped."""
        env_1v1.reset()
        for idx in [4, 5, 9, 10]:
            assert env_1v1.player.sigil_slots[idx].equipped


class TestSelfPlay:
    def test_self_play_env_creation(self):
        env = CombatEnv(num_enemies=1, self_play=True, seed=42)
        obs, _ = env.reset()
        assert obs is not None

    def test_opponent_obs(self):
        env = CombatEnv(num_enemies=1, self_play=True, seed=42)
        env.reset()
        enemy_id = env.enemies[0].agent_id
        opp_obs = env.get_opponent_obs(enemy_id)
        assert opp_obs is not None
        assert opp_obs["self_state"].shape == (SELF_STATE_DIM,)

    def test_set_opponent_action(self):
        env = CombatEnv(num_enemies=1, self_play=True, seed=42)
        env.reset()
        enemy_id = env.enemies[0].agent_id
        opp_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        opp_action[ACT_FORWARD] = 1
        env.set_opponent_action(enemy_id, opp_action)
        # Should not crash
        obs, _, _, _, _ = env.step(np.zeros(NUM_ACTIONS, dtype=np.int8))
        assert obs is not None


class TestDeterminism:
    def test_same_seed_same_result(self):
        env1 = CombatEnv(num_enemies=1, seed=123, domain_randomization=False)
        obs1, _ = env1.reset(seed=123)

        env2 = CombatEnv(num_enemies=1, seed=123, domain_randomization=False)
        obs2, _ = env2.reset(seed=123)

        for key in obs1:
            np.testing.assert_array_almost_equal(obs1[key], obs2[key], decimal=5)
