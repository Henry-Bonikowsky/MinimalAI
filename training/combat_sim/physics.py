"""Minecraft 1.8 PvP combat physics engine.

Key differences from 1.9+:
- No attack cooldown: every click deals full damage
- Sword blocking: reduces incoming damage by 50% and knockback
- Sprint reset: toggling sprint off/on between hits gives full KB
- No sweep attacks, no shields (1.9+ mechanic)

Also handles kit items: golden apples, health pots, ender pearls, totems.
"""

import numpy as np
from .entities import (
    Agent, WeaponType, WEAPON_STATS, Alliance,
    WALK_SPEED, SPRINT_SPEED, JUMP_VELOCITY, GRAVITY,
    DRAG_HORIZONTAL, DRAG_AIR, DRAG_VERTICAL, MAX_HEALTH, ARENA_SIZE,
    ARMOR_VALUE, ARMOR_TOUGHNESS, PROTECTION_EPF,
    GAP_EAT_TICKS, GAP_ABSORPTION, GAP_REGEN_DURATION,
    PEARL_COOLDOWN, PEARL_SELF_DAMAGE, HEALTH_POT_HEAL, POT_COOLDOWN,
    TOTEM_REGEN_DURATION, HURT_IMMUNE_TICKS,
)
from .sigils import SigilEngine


class CombatPhysics:
    """Simulates Minecraft 1.8 PvP combat physics."""

    def __init__(self, domain_randomization: bool = True, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.domain_randomization = domain_randomization
        self.sigil_engine = SigilEngine(rng=self.rng)

        # Domain randomization ranges
        self.knockback_scale = 1.0
        self.speed_scale = 1.0
        self.damage_scale = 1.0

    def randomize_params(self):
        """Randomize physics parameters for sim-to-real transfer."""
        if self.domain_randomization:
            self.knockback_scale = self.rng.uniform(0.8, 1.2)
            self.speed_scale = self.rng.uniform(0.9, 1.1)
            self.damage_scale = self.rng.uniform(0.9, 1.1)
        else:
            self.knockback_scale = 1.0
            self.speed_scale = 1.0
            self.damage_scale = 1.0

    def apply_movement(self, agent: Agent, forward: float, strafe: float, jump: bool):
        """Apply movement input to an agent."""
        if not agent.is_alive:
            return
        # Stun prevents movement
        if agent.has_effect("stun"):
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
            agent.vx += fx * speed * 0.1
            agent.vz += fz * speed * 0.1
        else:
            agent.vx += fx * speed * 0.02
            agent.vz += fz * speed * 0.02

        # Jump
        if jump and agent.on_ground:
            agent.vy = JUMP_VELOCITY
            agent.on_ground = False
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

        # Arena boundaries
        if abs(agent.x) > ARENA_SIZE:
            agent.x = np.clip(agent.x, -ARENA_SIZE, ARENA_SIZE)
            agent.vx *= -0.3
        if abs(agent.z) > ARENA_SIZE:
            agent.z = np.clip(agent.z, -ARENA_SIZE, ARENA_SIZE)
            agent.vz *= -0.3

        # Tick attack anti-cheat cooldown (1 tick between hits)
        if agent.attack_tick_cooldown > 0:
            agent.attack_tick_cooldown -= 1

        # Tick hurt immunity frames
        if agent.hurt_immune_ticks > 0:
            agent.hurt_immune_ticks -= 1

        # Tick kit cooldowns
        if agent.kit.pot_cooldown > 0:
            agent.kit.pot_cooldown -= 1
        if agent.kit.pearl_cooldown > 0:
            agent.kit.pearl_cooldown -= 1

        # Tick eating progress
        if agent.kit.is_eating:
            agent.kit.eating_ticks += 1
            if agent.kit.eating_ticks >= GAP_EAT_TICKS:
                self._finish_eating(agent)

        # Tick status effects
        agent.tick_effects()

        # Tick sigil cooldowns
        self.sigil_engine.tick_sigil_cooldowns(agent)

        # Tick no-KB timer is handled in sigil_engine.tick_sigil_cooldowns

        # Reset per-tick damage tracking
        agent.damage_dealt_this_tick = 0.0
        agent.damage_taken_this_tick = 0.0

    def try_attack(self, attacker: Agent, target: Agent, current_tick: int) -> dict:
        """Attempt a melee attack (1.8: no cooldown, full damage every hit).

        Returns dict: hit, damage, critical, knockback
        """
        result = {"hit": False, "damage": 0.0, "critical": False, "knockback": 0.0}

        if not attacker.is_alive or not target.is_alive:
            return result
        if attacker.has_effect("stun"):
            return result

        # Anti-cheat: 1 tick minimum between attacks
        if attacker.attack_tick_cooldown > 0:
            return result

        # Hurt immunity: target can't take damage during i-frames
        if target.hurt_immune_ticks > 0:
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

        # Facing check (within 90 degrees)
        angle_to_target = np.arctan2(dz, dx)
        angle_diff = abs(attacker.facing_angle - angle_to_target)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        if angle_diff > np.pi / 2:
            return result

        # --- 1.8 damage calculation ---
        # Base damage (full damage every hit, no cooldown scaling)
        base_damage = weapon_stats["damage"]
        sharpness_bonus = weapon_stats.get("sharpness_v", 0.0)
        raw_damage = (base_damage + sharpness_bonus) * self.damage_scale

        # Critical hit: falling + not on ground (no cooldown check in 1.8)
        critical = not attacker.on_ground and attacker.vy < 0
        if critical:
            raw_damage *= 1.5
            result["critical"] = True

        # 1.8 Armor reduction
        armor_reduction = self._calc_armor_reduction(raw_damage, target.armor,
                                                     target.armor_toughness)

        # Protection enchant reduction (separate multiplier)
        epf = min(20, target.protection_epf)
        prot_reduction = epf / 25.0

        # Sigil modifiers
        sigil_dr = target.sigil_damage_reduction
        sigil_amp = attacker.sigil_damage_amp

        # Apply all reductions
        actual_damage = raw_damage * (1.0 - armor_reduction) * (1.0 - prot_reduction)
        actual_damage *= (1.0 - sigil_dr) * (1.0 + sigil_amp)

        # 1.8 sword block: 50% damage reduction
        if target.is_blocking:
            actual_damage *= 0.5

        # Weakness effect
        if attacker.has_effect("weakness"):
            eff = attacker.get_effect("weakness")
            actual_damage *= max(0.0, 1.0 - 0.2 * (eff.amplifier + 1))

        # Resistance effect on target
        if target.has_effect("resistance"):
            eff = target.get_effect("resistance")
            actual_damage *= max(0.0, 1.0 - 0.2 * (eff.amplifier + 1))

        actual_damage = max(0.0, actual_damage)

        # Process sigil auto-defense on target
        defense_result = self.sigil_engine.on_take_damage(
            target, attacker, actual_damage, current_tick
        )
        actual_damage = max(0.0, actual_damage - defense_result["damage_reduction"])

        result["hit"] = True
        result["damage"] = actual_damage

        # Apply damage (absorption absorbs first)
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
        target.hurt_immune_ticks = HURT_IMMUNE_TICKS

        # Combo tracking
        if current_tick - attacker.last_hit_tick < 20:
            attacker.combo_counter += 1
        else:
            attacker.combo_counter = 1

        # Process sigil auto-attack on attacker
        attack_result = self.sigil_engine.on_attack_hit(
            attacker, target, result["damage"], current_tick
        )
        # Apply bonus sigil damage
        if attack_result["bonus_damage"] > 0:
            bonus = attack_result["bonus_damage"]
            if target.absorption > 0:
                absorbed = min(target.absorption, bonus)
                target.absorption -= absorbed
                bonus -= absorbed
            target.health = max(0.0, target.health - bonus)
            result["damage"] += attack_result["bonus_damage"]
            attacker.damage_dealt_this_tick += attack_result["bonus_damage"]
            target.damage_taken_this_tick += attack_result["bonus_damage"]

        # Death check (before totem)
        if target.health <= 0:
            if self._try_totem(target):
                pass  # Totem saved them
            else:
                target.is_alive = False
                attacker.kills += 1

        # Knockback (1.8 style)
        kb = self._calculate_knockback(attacker, target, dx, dz, dist)
        result["knockback"] = kb

        # 1 tick anti-cheat cooldown
        attacker.attack_tick_cooldown = 1

        # Sprint reset consumed
        attacker.sprint_reset_ready = False

        return result

    def _calc_armor_reduction(self, damage: float, armor: float, toughness: float) -> float:
        """1.8 armor damage reduction formula."""
        if armor <= 0:
            return 0.0
        # MC 1.8 formula: min(20, max(armor/5, armor - damage/(2 + toughness/4))) / 25
        inner = max(armor / 5.0, armor - damage / (2.0 + toughness / 4.0))
        defense = min(20.0, inner)
        return defense / 25.0

    def _calculate_knockback(self, attacker: Agent, target: Agent,
                             dx: float, dz: float, dist: float) -> float:
        """Apply 1.8 knockback. Returns magnitude."""
        if dist < 1e-6:
            return 0.0

        # No knockback if target has Quick Sand active
        if target.no_knockback_ticks > 0:
            return 0.0

        # Normalize direction
        nx = dx / dist
        nz = dz / dist

        # Base knockback
        kb_strength = 0.4 * self.knockback_scale

        # Sprint knockback bonus (1.8: only if sprint was active AND sprint reset was done)
        if attacker.is_sprinting:
            kb_strength *= 1.5

        # 1.8 sword block reduces knockback received
        if target.is_blocking:
            kb_strength *= 0.5

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

    def sprint_reset(self, agent: Agent):
        """1.8 sprint reset: toggle sprint off/on for max KB on next hit."""
        agent.is_sprinting = False
        agent.sprint_reset_ready = True
        # Re-sprint happens next tick when sprint action is taken

    def start_eating(self, agent: Agent) -> bool:
        """Start eating a golden apple. Returns True if started."""
        if agent.kit.golden_apples <= 0 or agent.kit.is_eating:
            return False
        agent.kit.is_eating = True
        agent.kit.eating_ticks = 0
        return True

    def cancel_eating(self, agent: Agent):
        """Cancel eating (e.g., if hit or action changes)."""
        agent.kit.is_eating = False
        agent.kit.eating_ticks = 0

    def _finish_eating(self, agent: Agent):
        """Apply golden apple effects."""
        agent.kit.golden_apples -= 1
        agent.kit.is_eating = False
        agent.kit.eating_ticks = 0
        # Absorption 2 hearts (4 HP)
        agent.absorption = min(20.0, agent.absorption + GAP_ABSORPTION)
        # Regen II for 5s
        agent.add_effect("regen", amplifier=1, duration=GAP_REGEN_DURATION)

    def throw_pot(self, agent: Agent) -> bool:
        """Throw a splash health potion. Instant heal. Returns True if thrown."""
        if agent.kit.health_pots <= 0 or agent.kit.pot_cooldown > 0:
            return False
        agent.kit.health_pots -= 1
        agent.kit.pot_cooldown = POT_COOLDOWN
        # Instant Health II: heal 8 HP
        agent.health = min(MAX_HEALTH, agent.health + HEALTH_POT_HEAL)
        return True

    def throw_pearl(self, agent: Agent, target_x: float = None,
                    target_z: float = None) -> bool:
        """Throw an ender pearl. Returns True if thrown.

        Simplified: instant teleport to target location (or 15 blocks forward).
        """
        if agent.kit.ender_pearls <= 0 or agent.kit.pearl_cooldown > 0:
            return False
        agent.kit.ender_pearls -= 1
        agent.kit.pearl_cooldown = PEARL_COOLDOWN

        if target_x is not None and target_z is not None:
            agent.x = target_x
            agent.z = target_z
        else:
            # Default: teleport 15 blocks in facing direction
            agent.x += np.cos(agent.facing_angle) * 15.0
            agent.z += np.sin(agent.facing_angle) * 15.0

        # Clamp to arena
        agent.x = np.clip(agent.x, -ARENA_SIZE, ARENA_SIZE)
        agent.z = np.clip(agent.z, -ARENA_SIZE, ARENA_SIZE)

        # Self damage
        agent.health = max(0.0, agent.health - PEARL_SELF_DAMAGE)
        agent.y = 0.0
        agent.vy = 0.0
        agent.on_ground = True
        return True

    def _try_totem(self, agent: Agent) -> bool:
        """Try to pop a totem of undying. Returns True if saved."""
        if agent.kit.totems <= 0:
            return False
        agent.kit.totems -= 1
        agent.health = 1.0
        agent.absorption = min(20.0, agent.absorption + 8.0)  # 4 absorption hearts
        agent.add_effect("regen", amplifier=1, duration=TOTEM_REGEN_DURATION)
        agent.add_effect("resistance", amplifier=0, duration=TOTEM_REGEN_DURATION)
        # Clear negative effects
        agent.remove_effect("wither")
        agent.remove_effect("slowness")
        agent.remove_effect("weakness")
        return True

    def turn_agent(self, agent: Agent, delta_angle: float):
        """Rotate agent's facing direction."""
        agent.facing_angle += delta_angle
        agent.facing_angle = (agent.facing_angle + np.pi) % (2 * np.pi) - np.pi

    def face_toward(self, agent: Agent, target: Agent, max_turn: float = np.pi / 4):
        """Turn agent toward target, limited by max_turn per tick."""
        dx = target.x - agent.x
        dz = target.z - agent.z
        desired_angle = np.arctan2(dz, dx)

        diff = desired_angle - agent.facing_angle
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        turn = np.clip(diff, -max_turn, max_turn)
        agent.facing_angle += turn
        agent.facing_angle = (agent.facing_angle + np.pi) % (2 * np.pi) - np.pi

    def distance_between(self, a: Agent, b: Agent) -> float:
        """Euclidean distance between two agents."""
        dx = a.x - b.x
        dz = a.z - b.z
        dy = a.y - b.y
        return np.sqrt(dx**2 + dz**2 + dy**2)
