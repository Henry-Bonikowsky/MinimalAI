"""Tests for the 1.8 PvP combat physics engine."""

import numpy as np
import pytest
from combat_sim.entities import (
    Agent, Alliance, WeaponType, WEAPON_STATS, MAX_HEALTH, SPRINT_SPEED,
    ARMOR_VALUE, ARMOR_TOUGHNESS, PROTECTION_EPF,
)
from combat_sim.physics import CombatPhysics


@pytest.fixture
def physics():
    return CombatPhysics(domain_randomization=False, rng=np.random.default_rng(42))


@pytest.fixture
def player():
    agent = Agent(agent_id=0, alliance=Alliance.ALLY)
    agent.reset(x=0.0, z=0.0, facing=0.0)
    return agent


@pytest.fixture
def enemy():
    agent = Agent(agent_id=1, alliance=Alliance.ENEMY)
    agent.reset(x=3.0, z=0.0, facing=np.pi)
    return agent


class TestMovement:
    def test_walk_speed(self, physics, player):
        player.is_sprinting = False
        physics.apply_movement(player, forward=1.0, strafe=0.0, jump=False)
        physics.tick_physics(player)
        assert abs(player.vx) > 0 or abs(player.vz) > 0

    def test_sprint_faster_than_walk(self, physics, player):
        walker = Agent(agent_id=10)
        walker.reset(x=0, z=0, facing=0)
        walker.is_sprinting = False
        physics.apply_movement(walker, 1.0, 0.0, False)
        physics.tick_physics(walker)
        walk_speed = np.sqrt(walker.vx**2 + walker.vz**2)

        sprinter = Agent(agent_id=11)
        sprinter.reset(x=0, z=0, facing=0)
        sprinter.is_sprinting = True
        physics.apply_movement(sprinter, 1.0, 0.0, False)
        physics.tick_physics(sprinter)
        sprint_speed = np.sqrt(sprinter.vx**2 + sprinter.vz**2)

        assert sprint_speed > walk_speed

    def test_jump_sets_vertical_velocity(self, physics, player):
        physics.apply_movement(player, 0.0, 0.0, jump=True)
        assert player.vy > 0
        assert not player.on_ground

    def test_gravity_pulls_down(self, physics, player):
        player.vy = 0.42
        player.y = 1.0
        player.on_ground = False
        physics.tick_physics(player)
        assert player.vy < 0.42

    def test_ground_collision(self, physics, player):
        player.y = 0.1
        player.vy = -0.5
        player.on_ground = False
        physics.tick_physics(player)
        assert player.y == 0.0
        assert player.on_ground

    def test_arena_boundaries(self, physics, player):
        player.x = 100.0
        physics.tick_physics(player)
        assert abs(player.x) <= 30.0

    def test_diagonal_movement_normalized(self, physics, player):
        cardinal = Agent(agent_id=10)
        cardinal.reset(x=0, z=0, facing=0)
        cardinal.is_sprinting = True
        physics.apply_movement(cardinal, 1.0, 0.0, False)
        physics.tick_physics(cardinal)
        cardinal_speed = np.sqrt(cardinal.vx**2 + cardinal.vz**2)

        diagonal = Agent(agent_id=11)
        diagonal.reset(x=0, z=0, facing=0)
        diagonal.is_sprinting = True
        physics.apply_movement(diagonal, 1.0, 1.0, False)
        physics.tick_physics(diagonal)
        diagonal_speed = np.sqrt(diagonal.vx**2 + diagonal.vz**2)

        assert abs(diagonal_speed - cardinal_speed) / cardinal_speed < 0.05

    def test_stun_prevents_movement(self, physics, player):
        """Stunned agents can't move."""
        player.add_effect("stun", amplifier=0, duration=20)
        initial_x = player.x
        physics.apply_movement(player, 1.0, 0.0, False)
        assert player.vx == 0.0  # no velocity added


class TestCombat18:
    """1.8 specific combat tests: no cooldown, full damage every hit."""

    def test_attack_in_range_hits(self, physics, player, enemy):
        result = physics.try_attack(player, enemy, current_tick=100)
        assert result["hit"]
        assert result["damage"] > 0

    def test_no_cooldown_full_damage(self, physics, player, enemy):
        """1.8: every hit deals full damage (no cooldown scaling)."""
        result1 = physics.try_attack(player, enemy, current_tick=100)
        enemy.health = MAX_HEALTH
        enemy.hurt_immune_ticks = 0  # clear i-frames
        player.attack_tick_cooldown = 0  # reset 1-tick anti-cheat
        result2 = physics.try_attack(player, enemy, current_tick=101)

        # Both should deal same damage (no cooldown reduction in 1.8)
        assert result1["hit"] and result2["hit"]
        assert abs(result1["damage"] - result2["damage"]) < 0.01

    def test_one_tick_anti_cheat(self, physics, player, enemy):
        """1 tick minimum between attacks (anti-cheat)."""
        physics.try_attack(player, enemy, current_tick=100)
        # Immediate re-attack should fail
        result = physics.try_attack(player, enemy, current_tick=100)
        assert not result["hit"]

    def test_sharpness_v_damage(self, physics, player, enemy):
        """Sharpness V adds +3 to sword damage."""
        enemy.armor = 0.0
        enemy.armor_toughness = 0.0
        enemy.protection_epf = 0
        result = physics.try_attack(player, enemy, current_tick=100)
        # Sword (7) + Sharp V (3) = 10 raw
        assert result["damage"] == pytest.approx(10.0, abs=0.1)

    def test_attack_out_of_range_misses(self, physics, player, enemy):
        enemy.x = 10.0
        result = physics.try_attack(player, enemy, current_tick=100)
        assert not result["hit"]

    def test_attack_not_facing_misses(self, physics, player, enemy):
        player.facing_angle = np.pi
        result = physics.try_attack(player, enemy, current_tick=100)
        assert not result["hit"]

    def test_critical_hit_bonus(self, physics, player, enemy):
        """1.8 crits: falling + not on ground = 1.5x (no cooldown check)."""
        player.on_ground = False
        player.vy = -0.1
        enemy.armor = 0.0
        enemy.armor_toughness = 0.0
        enemy.protection_epf = 0
        result = physics.try_attack(player, enemy, current_tick=100)
        assert result["critical"]
        # 10 * 1.5 = 15 raw
        assert result["damage"] == pytest.approx(15.0, abs=0.1)

    def test_dead_cant_attack(self, physics, player, enemy):
        player.is_alive = False
        result = physics.try_attack(player, enemy, current_tick=100)
        assert not result["hit"]

    def test_stun_prevents_attack(self, physics, player, enemy):
        player.add_effect("stun", amplifier=0, duration=20)
        result = physics.try_attack(player, enemy, current_tick=100)
        assert not result["hit"]


class TestArmorReduction18:
    """1.8 armor formula tests."""

    def test_full_netherite_reduces_damage(self, physics, player, enemy):
        """Full netherite (20 armor, 12 toughness) + Prot IV should reduce damage."""
        result = physics.try_attack(player, enemy, current_tick=100)
        # Raw 10, should be significantly reduced
        assert result["damage"] < 10.0

    def test_no_armor_full_damage(self, physics, player, enemy):
        """No armor = full damage."""
        enemy.armor = 0.0
        enemy.armor_toughness = 0.0
        enemy.protection_epf = 0
        result = physics.try_attack(player, enemy, current_tick=100)
        assert result["damage"] == pytest.approx(10.0, abs=0.1)

    def test_armor_formula_correctness(self, physics):
        """Verify the 1.8 armor reduction formula."""
        # Full netherite: armor=20, toughness=12, raw=10
        reduction = physics._calc_armor_reduction(10.0, 20.0, 12.0)
        # inner = max(20/5, 20 - 10/(2+12/4)) = max(4, 20-2) = max(4, 18) = 18
        # reduction = min(20, 18) / 25 = 18/25 = 0.72
        assert reduction == pytest.approx(0.72, abs=0.01)


class TestBlockhit:
    """1.8 sword blocking tests."""

    def test_block_reduces_damage_50pct(self, physics, player, enemy):
        """1.8 sword block reduces damage by 50%."""
        enemy.armor = 0.0
        enemy.armor_toughness = 0.0
        enemy.protection_epf = 0

        # Unblocked
        result1 = physics.try_attack(player, enemy, current_tick=100)
        dmg_unblocked = result1["damage"]

        # Reset and block
        enemy.health = MAX_HEALTH
        enemy.hurt_immune_ticks = 0  # clear i-frames
        enemy.is_blocking = True
        player.attack_tick_cooldown = 0
        result2 = physics.try_attack(player, enemy, current_tick=101)
        dmg_blocked = result2["damage"]

        assert dmg_blocked == pytest.approx(dmg_unblocked * 0.5, abs=0.1)

    def test_block_reduces_knockback(self, physics, player, enemy):
        """Blocking should halve knockback."""
        e1 = Agent(agent_id=10, alliance=Alliance.ENEMY)
        e1.reset(x=3.0, z=0.0, facing=np.pi)
        e1.armor = 0.0
        e1.is_blocking = False
        physics.try_attack(player, e1, current_tick=100)
        physics.tick_physics(e1)
        normal_x = e1.x

        e2 = Agent(agent_id=11, alliance=Alliance.ENEMY)
        e2.reset(x=3.0, z=0.0, facing=np.pi)
        e2.armor = 0.0
        e2.is_blocking = True
        player.attack_tick_cooldown = 0
        physics.try_attack(player, e2, current_tick=101)
        physics.tick_physics(e2)
        blocked_x = e2.x

        # Blocked should move less
        assert blocked_x < normal_x


class TestSprintReset:
    def test_sprint_reset_clears_sprint(self, physics, player):
        player.is_sprinting = True
        physics.sprint_reset(player)
        assert not player.is_sprinting
        assert player.sprint_reset_ready


class TestKnockback:
    def test_knockback_moves_target(self, physics, player, enemy):
        initial_x = enemy.x
        physics.try_attack(player, enemy, current_tick=100)
        physics.tick_physics(enemy)
        assert enemy.x > initial_x

    def test_sprint_knockback_stronger(self, physics, player, enemy):
        e1 = Agent(agent_id=10, alliance=Alliance.ENEMY)
        e1.reset(x=3.0, z=0.0, facing=np.pi)
        player.is_sprinting = False
        physics.try_attack(player, e1, current_tick=100)
        physics.tick_physics(e1)
        normal_dist = e1.x

        e2 = Agent(agent_id=11, alliance=Alliance.ENEMY)
        e2.reset(x=3.0, z=0.0, facing=np.pi)
        player.attack_tick_cooldown = 0
        player.is_sprinting = True
        physics.try_attack(player, e2, current_tick=101)
        physics.tick_physics(e2)
        sprint_dist = e2.x

        assert sprint_dist > normal_dist

    def test_no_kb_when_quicksand_active(self, physics, player, enemy):
        """Quick Sand: no knockback on target."""
        enemy.no_knockback_ticks = 100
        initial_vx = enemy.vx
        physics.try_attack(player, enemy, current_tick=100)
        # Velocity should not have changed from knockback
        assert enemy.vx == pytest.approx(initial_vx * 0.5, abs=0.01)  # only the halving


class TestKitItems:
    def test_eat_golden_apple(self, physics, player):
        """Eating a gap should grant absorption + regen."""
        initial_gaps = player.kit.golden_apples
        physics.start_eating(player)
        assert player.kit.is_eating

        # Simulate eating time
        for _ in range(GAP_EAT_TICKS + 1):
            physics.tick_physics(player)

        assert not player.kit.is_eating
        assert player.kit.golden_apples == initial_gaps - 1
        assert player.absorption > 0
        assert player.has_effect("regen")

    def test_cant_eat_when_empty(self, physics, player):
        player.kit.golden_apples = 0
        assert not physics.start_eating(player)

    def test_throw_health_pot(self, physics, player):
        player.health = 10.0
        initial_pots = player.kit.health_pots
        result = physics.throw_pot(player)
        assert result
        assert player.health == pytest.approx(18.0, abs=0.1)
        assert player.kit.health_pots == initial_pots - 1

    def test_pot_caps_at_max_health(self, physics, player):
        player.health = 18.0
        physics.throw_pot(player)
        assert player.health == MAX_HEALTH

    def test_throw_pearl(self, physics, player):
        initial_pearls = player.kit.ender_pearls
        initial_x = player.x
        result = physics.throw_pearl(player)
        assert result
        assert player.kit.ender_pearls == initial_pearls - 1
        assert player.kit.pearl_cooldown > 0
        # Should have teleported forward
        assert player.x != initial_x
        # Should have taken self damage
        assert player.health < MAX_HEALTH

    def test_pearl_cooldown(self, physics, player):
        physics.throw_pearl(player)
        assert not physics.throw_pearl(player)  # on cooldown

    def test_totem_saves_from_death(self, physics, player, enemy):
        """Totem should save from lethal damage."""
        player.health = 1.0
        player.kit.totems = 1
        enemy.armor = 0.0
        enemy.armor_toughness = 0.0
        enemy.protection_epf = 0

        # Player attacks enemy to set up state, then enemy kills player
        # Directly set up lethal scenario
        result = physics.try_attack(enemy, player, current_tick=100)
        assert result["hit"]
        # Player should have survived with totem
        assert player.is_alive
        assert player.health > 0
        assert player.kit.totems == 0

    def test_no_totem_means_death(self, physics, player, enemy):
        player.health = 1.0
        player.kit.totems = 0
        enemy.armor = 0.0
        physics.try_attack(enemy, player, current_tick=100)
        assert not player.is_alive


class TestDomainRandomization:
    def test_randomization_changes_params(self):
        physics = CombatPhysics(domain_randomization=True, rng=np.random.default_rng(1))
        physics.randomize_params()
        kb1 = physics.knockback_scale
        assert 0.8 <= kb1 <= 1.2

    def test_no_randomization(self):
        physics = CombatPhysics(domain_randomization=False)
        physics.randomize_params()
        assert physics.knockback_scale == 1.0
        assert physics.damage_scale == 1.0


# Import for GAP_EAT_TICKS
from combat_sim.entities import GAP_EAT_TICKS
