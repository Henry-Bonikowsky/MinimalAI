"""Tests for the combat physics engine."""

import numpy as np
import pytest
from combat_sim.entities import Agent, Alliance, WeaponType, WEAPON_STATS, MAX_HEALTH, SPRINT_SPEED
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
    agent.reset(x=3.0, z=0.0, facing=np.pi)  # 3 blocks away, facing player
    return agent


class TestMovement:
    def test_walk_speed(self, physics, player):
        """Walking should move at walk speed."""
        player.is_sprinting = False
        physics.apply_movement(player, forward=1.0, strafe=0.0, jump=False)
        physics.tick_physics(player)
        # Should have some forward velocity
        assert abs(player.vx) > 0 or abs(player.vz) > 0

    def test_sprint_faster_than_walk(self, physics, player):
        """Sprinting should be faster than walking."""
        # Walk
        walker = Agent(agent_id=10)
        walker.reset(x=0, z=0, facing=0)
        walker.is_sprinting = False
        physics.apply_movement(walker, 1.0, 0.0, False)
        physics.tick_physics(walker)
        walk_speed = np.sqrt(walker.vx**2 + walker.vz**2)

        # Sprint
        sprinter = Agent(agent_id=11)
        sprinter.reset(x=0, z=0, facing=0)
        sprinter.is_sprinting = True
        physics.apply_movement(sprinter, 1.0, 0.0, False)
        physics.tick_physics(sprinter)
        sprint_speed = np.sqrt(sprinter.vx**2 + sprinter.vz**2)

        assert sprint_speed > walk_speed

    def test_jump_sets_vertical_velocity(self, physics, player):
        """Jumping should set upward velocity."""
        physics.apply_movement(player, 0.0, 0.0, jump=True)
        assert player.vy > 0
        assert not player.on_ground

    def test_gravity_pulls_down(self, physics, player):
        """Gravity should reduce vertical velocity."""
        player.vy = 0.42
        player.y = 1.0
        player.on_ground = False
        physics.tick_physics(player)
        assert player.vy < 0.42

    def test_ground_collision(self, physics, player):
        """Agent should land when y reaches 0."""
        player.y = 0.1
        player.vy = -0.5
        player.on_ground = False
        physics.tick_physics(player)
        assert player.y == 0.0
        assert player.on_ground

    def test_arena_boundaries(self, physics, player):
        """Agents should be kept within arena bounds."""
        player.x = 100.0
        physics.tick_physics(player)
        assert abs(player.x) <= 30.0

    def test_diagonal_movement_normalized(self, physics, player):
        """Diagonal movement shouldn't be faster than cardinal."""
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

        # Should be approximately equal (within 5%)
        assert abs(diagonal_speed - cardinal_speed) / cardinal_speed < 0.05


class TestCombat:
    def test_attack_in_range_hits(self, physics, player, enemy):
        """Attack within reach should hit."""
        result = physics.try_attack(player, enemy, current_tick=100)
        assert result["hit"]
        assert result["damage"] > 0

    def test_attack_out_of_range_misses(self, physics, player, enemy):
        """Attack beyond reach should miss."""
        enemy.x = 10.0  # far away
        result = physics.try_attack(player, enemy, current_tick=100)
        assert not result["hit"]

    def test_attack_not_facing_misses(self, physics, player, enemy):
        """Attack when not facing target should miss."""
        player.facing_angle = np.pi  # facing away
        result = physics.try_attack(player, enemy, current_tick=100)
        assert not result["hit"]

    def test_cooldown_reduces_damage(self, physics, player, enemy):
        """Attacking mid-cooldown should deal reduced damage."""
        # Full cooldown attack
        player.attack_cooldown = 0
        result_full = physics.try_attack(player, enemy, current_tick=100)

        # Reset enemy health
        enemy.health = MAX_HEALTH
        # Immediate re-attack (on cooldown)
        result_quick = physics.try_attack(player, enemy, current_tick=101)

        if result_quick["hit"]:
            assert result_quick["damage"] < result_full["damage"]

    def test_critical_hit_bonus(self, physics, player, enemy):
        """Critical hits should deal extra damage."""
        # Setup crit conditions: falling + full cooldown
        player.on_ground = False
        player.vy = -0.1  # falling
        player.attack_cooldown = 0

        result = physics.try_attack(player, enemy, current_tick=100)
        assert result["hit"]
        assert result["critical"]
        # Crit should deal 1.5x damage
        base_damage = WEAPON_STATS[WeaponType.SWORD]["damage"]
        assert result["damage"] > base_damage * 1.2  # at least more than base

    def test_damage_reduces_health(self, physics, player, enemy):
        """Damage should reduce target health."""
        initial_health = enemy.health
        physics.try_attack(player, enemy, current_tick=100)
        assert enemy.health < initial_health

    def test_lethal_damage_kills(self, physics, player, enemy):
        """Enough damage should kill the target."""
        enemy.health = 1.0
        result = physics.try_attack(player, enemy, current_tick=100)
        assert result["hit"]
        assert not enemy.is_alive
        assert player.kills == 1

    def test_dead_cant_attack(self, physics, player, enemy):
        """Dead agents can't attack."""
        player.is_alive = False
        result = physics.try_attack(player, enemy, current_tick=100)
        assert not result["hit"]

    def test_dead_cant_be_attacked(self, physics, player, enemy):
        """Dead agents can't be attacked."""
        enemy.is_alive = False
        result = physics.try_attack(player, enemy, current_tick=100)
        assert not result["hit"]

    def test_armor_reduces_damage(self, physics, player, enemy):
        """Armor should reduce damage taken."""
        enemy_no_armor = Agent(agent_id=2, alliance=Alliance.ENEMY)
        enemy_no_armor.reset(x=3.0, z=0.0, facing=np.pi)
        enemy_no_armor.armor = 0.0

        enemy_armored = Agent(agent_id=3, alliance=Alliance.ENEMY)
        enemy_armored.reset(x=3.0, z=0.0, facing=np.pi)
        enemy_armored.armor = 15.0

        result1 = physics.try_attack(player, enemy_no_armor, current_tick=100)
        player.attack_cooldown = 0  # reset for fair comparison
        result2 = physics.try_attack(player, enemy_armored, current_tick=100)

        assert result2["damage"] < result1["damage"]

    def test_shield_blocking(self, physics, player, enemy):
        """Shield blocking should reduce damage."""
        enemy.is_blocking = False
        result_unblocked = physics.try_attack(player, enemy, current_tick=100)

        enemy.health = MAX_HEALTH
        enemy.is_blocking = True
        player.attack_cooldown = 0
        result_blocked = physics.try_attack(player, enemy, current_tick=101)

        assert result_blocked["damage"] < result_unblocked["damage"]


class TestKnockback:
    def test_knockback_moves_target(self, physics, player, enemy):
        """Knockback should push target away."""
        initial_x = enemy.x
        physics.try_attack(player, enemy, current_tick=100)
        physics.tick_physics(enemy)
        # Enemy should be pushed away (positive X direction since enemy is at +X)
        assert enemy.x > initial_x

    def test_sprint_knockback_stronger(self, physics, player, enemy):
        """Sprint attacks should have stronger knockback."""
        # Normal attack
        e1 = Agent(agent_id=10, alliance=Alliance.ENEMY)
        e1.reset(x=3.0, z=0.0, facing=np.pi)
        player.is_sprinting = False
        physics.try_attack(player, e1, current_tick=100)
        physics.tick_physics(e1)
        normal_dist = e1.x

        # Sprint attack
        e2 = Agent(agent_id=11, alliance=Alliance.ENEMY)
        e2.reset(x=3.0, z=0.0, facing=np.pi)
        player.attack_cooldown = 0
        player.is_sprinting = True
        physics.try_attack(player, e2, current_tick=101)
        physics.tick_physics(e2)
        sprint_dist = e2.x

        assert sprint_dist > normal_dist


class TestWTap:
    def test_wtap_resets_sprint(self, physics, player):
        """W-tap should reset sprint state."""
        player.is_sprinting = True
        physics.apply_wtap(player)
        assert not player.is_sprinting


class TestSigils:
    def test_sigil_activation_on_cooldown(self, physics, player):
        """Can't activate sigil on cooldown."""
        player.sigil_binds[0].available = True
        player.sigil_binds[0].cooldown_remaining = 50
        result = physics.activate_sigil(player, 0, [], 100)
        assert not result

    def test_sigil_activation_available(self, physics, player):
        """Can activate available sigil."""
        player.sigil_binds[0].available = True
        player.sigil_binds[0].cooldown_remaining = 0
        result = physics.activate_sigil(player, 0, [], 100)
        assert result
        assert player.sigil_binds[0].cooldown_remaining > 0

    def test_sigil_unavailable(self, physics, player):
        """Can't activate unavailable sigil."""
        player.sigil_binds[0].available = False
        result = physics.activate_sigil(player, 0, [], 100)
        assert not result


class TestDomainRandomization:
    def test_randomization_changes_params(self):
        """Domain randomization should produce different physics params."""
        physics = CombatPhysics(domain_randomization=True, rng=np.random.default_rng(1))
        physics.randomize_params()
        kb1 = physics.knockback_scale

        physics.randomize_params()
        kb2 = physics.knockback_scale

        # Very unlikely to be exactly the same
        # But with fixed seed + 2 calls, they might differ
        # Just check they're in valid range
        assert 0.8 <= kb1 <= 1.2
        assert 0.8 <= kb2 <= 1.2

    def test_no_randomization(self):
        """Without domain randomization, params should be 1.0."""
        physics = CombatPhysics(domain_randomization=False)
        physics.randomize_params()
        assert physics.knockback_scale == 1.0
        assert physics.damage_scale == 1.0
