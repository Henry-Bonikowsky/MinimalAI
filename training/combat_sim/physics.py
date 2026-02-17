"""Minecraft PvP combat physics engine.

Implements core mechanics: knockback, attack cooldowns, sprint, W-tap,
critical hits, health/armor damage calculation, and movement.
Physics values sourced from Minecraft Wiki for 1.21+ combat.
"""

import numpy as np
from .entities import (
    Agent, WeaponType, WEAPON_STATS, Alliance,
    WALK_SPEED, SPRINT_SPEED, JUMP_VELOCITY, GRAVITY,
    DRAG_HORIZONTAL, DRAG_AIR, DRAG_VERTICAL, MAX_HEALTH, ARENA_SIZE,
)


class CombatPhysics:
    """Simulates Minecraft PvP combat physics."""

    def __init__(self, domain_randomization: bool = True, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.domain_randomization = domain_randomization

        # Domain randomization ranges (applied per episode reset)
        self.knockback_scale = 1.0
        self.cooldown_scale = 1.0
        self.speed_scale = 1.0
        self.damage_scale = 1.0

    def randomize_params(self):
        """Randomize physics parameters for sim-to-real transfer."""
        if self.domain_randomization:
            self.knockback_scale = self.rng.uniform(0.8, 1.2)
            self.cooldown_scale = self.rng.uniform(0.9, 1.1)
            self.speed_scale = self.rng.uniform(0.9, 1.1)
            self.damage_scale = self.rng.uniform(0.9, 1.1)
        else:
            self.knockback_scale = 1.0
            self.cooldown_scale = 1.0
            self.speed_scale = 1.0
            self.damage_scale = 1.0

    def apply_movement(self, agent: Agent, forward: float, strafe: float, jump: bool):
        """Apply movement input to an agent.

        Args:
            agent: The agent to move.
            forward: -1 to 1 (negative = backward).
            strafe: -1 to 1 (negative = left, positive = right).
            jump: Whether to jump this tick.
        """
        if not agent.is_alive:
            return

        speed = agent.move_speed * self.speed_scale

        # Movement direction relative to facing
        move_angle = agent.facing_angle
        fx = np.cos(move_angle) * forward - np.sin(move_angle) * strafe
        fz = np.sin(move_angle) * forward + np.cos(move_angle) * strafe

        # Normalize diagonal movement
        mag = np.sqrt(fx**2 + fz**2)
        if mag > 1e-6:
            fx /= mag
            fz /= mag

        # Apply acceleration
        if agent.on_ground:
            agent.vx += fx * speed * 0.1  # ground acceleration
            agent.vz += fz * speed * 0.1
        else:
            agent.vx += fx * speed * 0.02  # air control (much less)
            agent.vz += fz * speed * 0.02

        # Jump
        if jump and agent.on_ground:
            agent.vy = JUMP_VELOCITY
            agent.on_ground = False
            # Sprint jump boost
            if agent.is_sprinting:
                agent.vx += np.cos(agent.facing_angle) * 0.2
                agent.vz += np.sin(agent.facing_angle) * 0.2

    def tick_physics(self, agent: Agent):
        """Advance one tick of physics for an agent."""
        if not agent.is_alive:
            return

        # Apply drag
        if agent.on_ground:
            agent.vx *= DRAG_HORIZONTAL
            agent.vz *= DRAG_HORIZONTAL
        else:
            agent.vx *= DRAG_AIR
            agent.vz *= DRAG_AIR

        # Gravity
        agent.vy -= GRAVITY
        agent.vy *= DRAG_VERTICAL

        # Update position
        agent.x += agent.vx
        agent.z += agent.vz
        agent.y += agent.vy

        # Ground collision
        if agent.y <= 0.0:
            agent.y = 0.0
            agent.vy = 0.0
            agent.on_ground = True
        else:
            agent.on_ground = False

        # Arena boundaries (soft wall)
        if abs(agent.x) > ARENA_SIZE:
            agent.x = np.clip(agent.x, -ARENA_SIZE, ARENA_SIZE)
            agent.vx *= -0.3
        if abs(agent.z) > ARENA_SIZE:
            agent.z = np.clip(agent.z, -ARENA_SIZE, ARENA_SIZE)
            agent.vz *= -0.3

        # Tick cooldowns
        if agent.attack_cooldown > 0:
            agent.attack_cooldown -= 1

        # Tick sigil cooldowns
        for bind in agent.sigil_binds:
            if bind.cooldown_remaining > 0:
                bind.cooldown_remaining -= 1

        # Reset per-tick damage tracking
        agent.damage_dealt_this_tick = 0.0
        agent.damage_taken_this_tick = 0.0

    def try_attack(self, attacker: Agent, target: Agent, current_tick: int) -> dict:
        """Attempt a melee attack from attacker to target.

        Returns:
            dict with keys: hit (bool), damage (float), critical (bool), knockback (float)
        """
        result = {"hit": False, "damage": 0.0, "critical": False, "knockback": 0.0}

        if not attacker.is_alive or not target.is_alive:
            return result

        # Range check
        dx = target.x - attacker.x
        dz = target.z - attacker.z
        dy = target.y - attacker.y
        dist = np.sqrt(dx**2 + dz**2 + dy**2)

        weapon_stats = WEAPON_STATS[attacker.weapon]
        reach = weapon_stats["reach"]

        if dist > reach:
            return result

        # Facing check (must be roughly facing target, within 90 degrees)
        angle_to_target = np.arctan2(dz, dx)
        angle_diff = abs(attacker.facing_angle - angle_to_target)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        if angle_diff > np.pi / 2:
            return result

        # Cooldown check - can attack anytime but damage scales
        damage_mult = attacker.damage_multiplier

        # Base damage
        base_damage = weapon_stats["damage"] * damage_mult * self.damage_scale

        # Sigil damage amplification
        base_damage *= (1.0 + attacker.sigil_damage_amp)

        # Critical hit: falling + full cooldown
        critical = (not attacker.on_ground and attacker.vy < 0 and
                    attacker.cooldown_progress >= 0.9)
        if critical:
            base_damage *= 1.5
            result["critical"] = True

        # Armor damage reduction
        armor_reduction = target.armor * 0.04  # 4% per armor point
        armor_reduction = min(armor_reduction, 0.8)  # cap at 80%
        actual_damage = base_damage * (1.0 - armor_reduction)

        # Sigil damage reduction on target
        actual_damage *= (1.0 - target.sigil_damage_reduction)

        # Shield blocking
        if target.is_blocking:
            actual_damage *= 0.33  # shields block ~67% in MC

        result["hit"] = True
        result["damage"] = actual_damage

        # Apply damage
        if target.absorption > 0:
            absorbed = min(target.absorption, actual_damage)
            target.absorption -= absorbed
            actual_damage -= absorbed
        target.health -= actual_damage
        target.health = max(0.0, target.health)

        # Track damage
        attacker.damage_dealt_this_tick += result["damage"]
        target.damage_taken_this_tick += result["damage"]
        attacker.last_hit_tick = current_tick
        target.last_hurt_tick = current_tick

        # Combo tracking
        if current_tick - attacker.last_hit_tick < 20:
            attacker.combo_counter += 1
        else:
            attacker.combo_counter = 1

        # Death check
        if target.health <= 0:
            target.is_alive = False
            attacker.kills += 1

        # Knockback
        kb = self._calculate_knockback(attacker, target, dx, dz, dist)
        result["knockback"] = kb

        # Reset attack cooldown
        max_cd = int(20.0 / weapon_stats["speed"] * self.cooldown_scale)
        attacker.attack_cooldown = max_cd

        return result

    def _calculate_knockback(self, attacker: Agent, target: Agent,
                             dx: float, dz: float, dist: float) -> float:
        """Apply knockback to target. Returns knockback magnitude."""
        if dist < 1e-6:
            return 0.0

        # Normalize direction
        nx = dx / dist
        nz = dz / dist

        # Base knockback
        kb_strength = 0.4 * self.knockback_scale

        # Sprint knockback bonus
        if attacker.is_sprinting:
            kb_strength *= 1.5

        # Halve target velocity first (MC mechanic)
        target.vx *= 0.5
        target.vz *= 0.5

        # Apply knockback force
        target.vx += nx * kb_strength
        target.vz += nz * kb_strength

        # Vertical knockback
        target.vy = min(target.vy + 0.4 * self.knockback_scale, 0.5)
        target.on_ground = False

        return kb_strength

    def apply_wtap(self, agent: Agent):
        """W-tap: brief sprint release then re-sprint for knockback reset.

        In MC, releasing forward briefly resets sprint state, so the next
        hit applies sprint knockback even mid-combo.
        """
        agent.is_sprinting = False
        # Next tick when sprint is re-enabled, knockback bonus applies

    def turn_agent(self, agent: Agent, delta_angle: float):
        """Rotate agent's facing direction."""
        agent.facing_angle += delta_angle
        # Normalize to [-pi, pi]
        agent.facing_angle = (agent.facing_angle + np.pi) % (2 * np.pi) - np.pi

    def face_toward(self, agent: Agent, target: Agent, max_turn: float = np.pi / 4):
        """Turn agent toward target, limited by max_turn per tick."""
        dx = target.x - agent.x
        dz = target.z - agent.z
        desired_angle = np.arctan2(dz, dx)

        diff = desired_angle - agent.facing_angle
        # Normalize to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        # Clamp turn rate
        turn = np.clip(diff, -max_turn, max_turn)
        agent.facing_angle += turn
        agent.facing_angle = (agent.facing_angle + np.pi) % (2 * np.pi) - np.pi

    def activate_sigil(self, agent: Agent, bind_index: int, targets: list[Agent],
                       current_tick: int) -> bool:
        """Activate a sigil bind. Returns True if activated successfully.

        In v1, this is a no-op (sigils not available). v2 will add effects.
        """
        if bind_index < 0 or bind_index >= 4:
            return False

        bind = agent.sigil_binds[bind_index]
        if not bind.available or bind.cooldown_remaining > 0:
            return False

        # v1: sigils are structurally present but do nothing
        # v2 will add: damage, buffs, AoE effects here
        bind.cooldown_remaining = bind.cooldown_max
        return True

    def distance_between(self, a: Agent, b: Agent) -> float:
        """Euclidean distance between two agents."""
        dx = a.x - b.x
        dz = a.z - b.z
        dy = a.y - b.y
        return np.sqrt(dx**2 + dz**2 + dy**2)
