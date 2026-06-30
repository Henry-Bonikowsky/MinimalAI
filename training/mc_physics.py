"""Minecraft 1.21 physics simulator calibrated from server recordings.

Server runs Paper-Kitara (1.8 combat on 1.21) with custom knockback, damage, and
enchant formulas. All combat constants match the Kitara config exactly.

Movement: vanilla LivingEntity.travel() (verified from recordings)
Combat: Kitara Advanced KB + legacy 1.8 damage/sharpness/armor/protection
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Movement constants (verified from server recordings via linear regression) ──

GRAVITY = 0.08            # blocks/tick^2 (confirmed: residual 3.7e-7)
AIR_DRAG = 0.98           # vertical drag multiplier
GROUND_FRICTION = 0.6     # default block friction (stone, dirt, etc.)
DRAG_FACTOR = 0.91        # horizontal drag = friction * 0.91
JUMP_IMPULSE = 0.42       # vy set on jump
SPRINT_JUMP_KICK = 0.2    # forward boost on sprint-jump
SNEAK_SPEED_MULT = 0.3    # sneaking speed multiplier
SPRINT_SPEED_MULT = 1.3   # sprinting speed multiplier vs walk
PLAYER_WALK_SPEED = 0.1   # attribute base value

# ── Kitara combat constants (from KitaraConfig.java + 0034-kitara-into-paper.patch) ──

ATTACK_REACH = 3.0        # blocks (player reach)
BASE_SWORD_DAMAGE = 4.0   # Kitara: swordDamageBase = 4.0 (overrides modern 3.0)
SHARPNESS_PER_LEVEL = 1.25  # Kitara legacy sharpness: level * 1.25
CRIT_MULTIPLIER = 1.5     # Kitara: critMultiplier = 1.5
I_FRAME_TICKS = 10        # Server direct damage uses invulnerableTime = 10 (0.5s)
SWORD_BLOCKING_DMG_MULT = 0.5  # Kitara: swordBlockingMultiplier = 0.5

# Kitara: disableArmorToughness = true
# Armor formula: damage * (25 - armor) / 25 (no toughness)

# Arcane Sigils enchant scaling (overrides Kitara legacy EPF)
# Config: enchantment-scaling.protection per level = DR% per armor piece
PROTECTION_DR_TABLE = {5: 0.0}  # Server direct damage doesn't apply protection enchant

# ── Kitara Advanced Knockback (from AdvancedKnockback.java defaults) ──

KB_HORIZONTAL = 0.38         # base horizontal KB magnitude
KB_HORIZONTAL_ON_GROUND = 1.1   # multiplier when victim on ground
KB_HORIZONTAL_SPRINTING = 1.35  # multiplier when attacker sprint-hitting
KB_HORIZONTAL_INHERIT = True    # inherit victim's existing horizontal motion
KB_HORIZONTAL_FRICTION = 1.5    # friction applied to inherited motion
KB_HORIZONTAL_ENCHANT_DEDUCTION = 0.8  # KB enchant: motX *= enchLvl / deduction

KB_VERTICAL = 0.35           # base vertical KB
KB_VERTICAL_ON_GROUND = 1.0  # multiplier when victim on ground
KB_VERTICAL_IN_AIR = 1.0     # multiplier when victim in air
KB_VERTICAL_SPRINTING = 1.0  # multiplier when attacker sprinting
KB_VERTICAL_INHERIT = False   # don't inherit victim's vertical motion
KB_VERTICAL_LIMIT_ENABLE = False  # vertical capping disabled
KB_VERTICAL_LIMIT = 1.2      # (unused since disabled)

KB_SWORD_BLOCK_HORIZONTAL = 0.85  # Kitara: swordBlockingHorizontalMultiplier
KB_SWORD_BLOCK_VERTICAL = 0.95    # Kitara: swordBlockingVerticalMultiplier

# 1.7 back-hit mechanic
KB_ONE_POINT_SEVEN_ENABLED = True
KB_ONE_POINT_SEVEN_HORIZONTAL = 0.25

# Attacker effects on hit
KB_SLOWDOWN = 0.6         # attacker horizontal velocity *= slowdown on hit
KB_CANCEL_SPRINT = True   # cancel attacker sprint after hit


@dataclass
class PlayerState:
    """Full state of a player at one tick."""

    # Position & velocity
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Orientation
    yaw: float = 0.0       # degrees
    pitch: float = 0.0     # degrees

    # Flags
    on_ground: bool = True
    sprinting: bool = False
    sneaking: bool = False
    jumping: bool = False

    # Health & defense
    health: float = 20.0
    absorption: float = 0.0
    armor: int = 20          # full diamond/netherite = 20
    hurt_time: int = 0       # i-frames countdown

    # Effects & enchants
    speed_amplifier: int = -1       # -1 = no effect, 0 = Speed I, 1 = Speed II
    strength_amplifier: int = -1    # -1 = no effect
    sharpness_level: int = 6        # Sharp VI (server standard)
    kb_enchant_level: int = 0       # Knockback enchant on weapon
    protection_level: int = 5       # Prot V per piece (server standard)

    # Sword blocking (1.8 mechanic)
    sword_blocking: bool = False

    # Collision (from server recording)
    horizontal_collision: bool = False
    vertical_collision: bool = False
    fall_distance: float = 0.0

    def copy(self) -> 'PlayerState':
        """Shallow copy."""
        return PlayerState(**self.__dict__)


@dataclass
class CombatEvent:
    """Result of a combat interaction in one tick."""
    damage: float = 0.0
    knockback_applied: bool = False
    critical: bool = False
    attacker_sprinting: bool = False


class MCSimulator:
    """Vanilla MC 1.21 physics + combat simulator.

    All constants are tuneable for calibration against server recordings.
    """

    def __init__(
        self,
        gravity: float = GRAVITY,
        air_drag: float = AIR_DRAG,
        ground_friction: float = GROUND_FRICTION,
        drag_factor: float = DRAG_FACTOR,
        jump_impulse: float = JUMP_IMPULSE,
        sprint_jump_kick: float = SPRINT_JUMP_KICK,
        arena_half_size: float = 30.0,
        arena_floor_y: float = 0.0,
    ):
        self.gravity = gravity
        self.air_drag = air_drag
        self.ground_friction = ground_friction
        self.drag_factor = drag_factor
        self.jump_impulse = jump_impulse
        self.sprint_jump_kick = sprint_jump_kick
        self.arena_half_size = arena_half_size
        self.arena_floor_y = arena_floor_y

    def get_params(self) -> dict:
        """Return current physics parameters as a dict."""
        return {
            "gravity": self.gravity,
            "air_drag": self.air_drag,
            "ground_friction": self.ground_friction,
            "drag_factor": self.drag_factor,
            "jump_impulse": self.jump_impulse,
            "sprint_jump_kick": self.sprint_jump_kick,
        }

    def set_params(self, params: dict):
        """Set physics parameters from a dict (for calibration tuning)."""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    # ─────────────────────────────────────────────────────────
    #  Movement: implements vanilla LivingEntity.travel()
    # ─────────────────────────────────────────────────────────

    def step_movement(
        self,
        state: PlayerState,
        actions: dict,
        friction: float = None,
    ) -> PlayerState:
        """Advance one tick of movement physics.

        Args:
            state: current player state (not mutated)
            actions: dict with keys: forward, strafe, jump, sprint, sneak
                     values are floats in [-1, 1] for movement, bool for others
            friction: block friction below player (default: self.ground_friction)

        Returns:
            New PlayerState after physics tick.
        """
        s = state.copy()
        if friction is None:
            friction = self.ground_friction

        fwd = float(actions.get("forward", 0.0))
        strafe = float(actions.get("strafe", 0.0))
        jump = bool(actions.get("jump", False))
        sprint = bool(actions.get("sprint", False))
        sneak = bool(actions.get("sneak", False))

        # Save initial on_ground state (vanilla uses this for BOTH
        # input acceleration AND post-move drag, computed at tick start)
        was_on_ground = s.on_ground

        # Update sprint/sneak state
        s.sprinting = sprint and not sneak and fwd > 0
        s.sneaking = sneak

        # ── Jump (aiStep, before travel) ──
        if jump and was_on_ground:
            s.vy = self.jump_impulse
            s.on_ground = False
            s.jumping = True
            # Sprint jump forward boost
            if s.sprinting:
                yaw_rad = math.radians(s.yaw)
                s.vx -= math.sin(yaw_rad) * self.sprint_jump_kick
                s.vz += math.cos(yaw_rad) * self.sprint_jump_kick
        else:
            s.jumping = False

        # ── Input acceleration (travel: moveRelative) ──
        move_speed = PLAYER_WALK_SPEED
        if s.speed_amplifier >= 0:
            move_speed *= 1.0 + 0.2 * (s.speed_amplifier + 1)

        if s.sprinting:
            move_speed *= SPRINT_SPEED_MULT
        elif s.sneaking:
            move_speed *= SNEAK_SPEED_MULT

        # Slip factor determines input acceleration
        # Computed from on_ground state at START of tick
        slip = friction * self.drag_factor if was_on_ground else self.drag_factor
        if was_on_ground:
            accel = move_speed * (0.6 / slip) ** 3
        else:
            accel = 0.02

        # Resolve input direction relative to yaw
        yaw_rad = math.radians(s.yaw)
        sin_yaw = math.sin(yaw_rad)
        cos_yaw = math.cos(yaw_rad)

        # Normalize diagonal input
        dist_sq = fwd * fwd + strafe * strafe
        if dist_sq > 1.0:
            inv_len = 1.0 / math.sqrt(dist_sq)
            fwd *= inv_len
            strafe *= inv_len

        # MC coordinate system: -sin(yaw) is forward X, cos(yaw) is forward Z
        input_x = -sin_yaw * fwd - cos_yaw * strafe
        input_z = cos_yaw * fwd - sin_yaw * strafe

        s.vx += input_x * accel
        s.vz += input_z * accel

        # ── Position update (travel: move) ──
        # Vanilla: position += velocity BEFORE drag/gravity
        s.x += s.vx
        s.y += s.vy
        s.z += s.vz

        # ── Ground collision (after move) ──
        if s.y <= self.arena_floor_y:
            s.y = self.arena_floor_y
            if s.vy < 0:
                if s.fall_distance > 3.0:
                    pass  # Fall damage would apply here
                s.vy = 0.0
            s.on_ground = True
            s.fall_distance = 0.0
            s.vertical_collision = True
        else:
            s.on_ground = False
            s.vertical_collision = False
            if s.vy < 0:
                s.fall_distance -= s.vy

        # ── Arena wall collision ──
        half = self.arena_half_size
        s.horizontal_collision = False
        if abs(s.x) > half:
            s.x = max(-half, min(half, s.x))
            s.vx = 0.0
            s.horizontal_collision = True
        if abs(s.z) > half:
            s.z = max(-half, min(half, s.z))
            s.vz = 0.0
            s.horizontal_collision = True

        # ── Post-move drag + gravity ──
        # Vanilla: vx = vx * slip; vy = (vy - gravity) * 0.98; vz = vz * slip
        # Uses slip computed from on_ground at START of tick
        s.vx *= slip
        s.vz *= slip
        s.vy = (s.vy - self.gravity) * self.air_drag

        return s

    # ─────────────────────────────────────────────────────────
    #  Combat: implements Player.attack() damage + knockback
    # ─────────────────────────────────────────────────────────

    def step_combat(
        self,
        attacker: PlayerState,
        target: PlayerState,
        attack_action: bool,
    ) -> tuple[PlayerState, PlayerState, CombatEvent]:
        """Process one attack attempt using Kitara combat model.

        Kitara uses 1.8-style combat: no attack cooldown for swords,
        legacy sharpness (level * 1.25), legacy armor (no toughness),
        and Advanced Knockback with configurable multipliers.
        """
        a = attacker.copy()
        t = target.copy()
        event = CombatEvent()

        if not attack_action:
            return a, t, event

        # Range check
        dx = t.x - a.x
        dy = t.y - a.y
        dz = t.z - a.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist > ATTACK_REACH:
            return a, t, event

        # I-frame check: server blocks ALL hits during i-frames
        if t.hurt_time > 0:
            return a, t, event

        # ── Damage calculation ──
        base_damage = BASE_SWORD_DAMAGE

        # Kitara legacy sharpness: level * 1.25
        enchant_damage = 0.0
        if a.sharpness_level > 0:
            enchant_damage = SHARPNESS_PER_LEVEL * a.sharpness_level

        # Strength effect (1.8: +3 per level of Strength)
        if a.strength_amplifier >= 0:
            base_damage += 3.0 * (a.strength_amplifier + 1)

        # Critical hit: falling + not on ground
        critical = (a.fall_distance > 0.0 and not a.on_ground and not a.sneaking
                    and a.vy < 0)
        if critical:
            base_damage *= CRIT_MULTIPLIER
            event.critical = True

        base_damage += enchant_damage
        raw_damage = base_damage

        # ── Sword blocking damage reduction ──
        # Kitara: damage = (1 + damage) * swordBlockingMultiplier
        if t.sword_blocking:
            raw_damage = (1.0 + raw_damage) * SWORD_BLOCKING_DMG_MULT

        # ── Legacy armor reduction (Kitara: no toughness) ──
        # damage = damage * (25 - armor) / 25
        if t.armor > 0:
            raw_damage = raw_damage * (25 - t.armor) / 25.0

        # ── Protection enchant (Arcane Sigils override) ──
        # AS enchant-scaling overrides Kitara legacy EPF with per-level DR%.
        # Config: prot 5 = 0.5% DR per piece, 4 pieces = 2% total.
        if t.protection_level > 0:
            dr_per_piece = PROTECTION_DR_TABLE.get(t.protection_level, 0.0)
            total_dr = min(dr_per_piece * 4, 100.0) / 100.0
            raw_damage *= (1.0 - total_dr)

        actual_damage = max(0.0, raw_damage)
        event.damage = actual_damage

        # ── Apply damage ──
        if t.absorption > 0:
            absorbed = min(t.absorption, actual_damage)
            t.absorption -= absorbed
            actual_damage -= absorbed
        t.health = max(0.0, t.health - actual_damage)
        t.hurt_time = I_FRAME_TICKS

        # ── Kitara Advanced Knockback ──
        event.attacker_sprinting = a.sprinting
        a_yaw_rad = math.radians(a.yaw)

        # Step 1: compute KB direction from attacker yaw
        if KB_HORIZONTAL_INHERIT:
            # Inherit victim motion with friction, then add yaw direction
            mot_x = t.vx * KB_HORIZONTAL_FRICTION + math.sin(a_yaw_rad) * -1.0
            mot_z = t.vz * KB_HORIZONTAL_FRICTION + math.cos(a_yaw_rad)
        else:
            mot_x = math.sin(a_yaw_rad) * -1.0
            mot_z = math.cos(a_yaw_rad)

        # Step 2: 1.7 back-hit check (attacker and victim facing same direction)
        one_point_seven = False
        if KB_ONE_POINT_SEVEN_ENABLED:
            a_dir_x = -math.sin(a_yaw_rad)
            a_dir_z = math.cos(a_yaw_rad)
            t_yaw_rad = math.radians(t.yaw)
            t_dir_x = -math.sin(t_yaw_rad)
            t_dir_z = math.cos(t_yaw_rad)
            dot = a_dir_x * t_dir_x + a_dir_z * t_dir_z
            if dot > 0.1 and t.sprinting:
                one_point_seven = True

        # Step 3: scale by horizontal base
        h_scale = KB_ONE_POINT_SEVEN_HORIZONTAL if one_point_seven else KB_HORIZONTAL
        mot_x *= h_scale
        mot_z *= h_scale

        # Step 4: vertical
        if KB_VERTICAL_INHERIT:
            mot_y = t.vy * 1.0 + KB_VERTICAL  # verticalFriction default 1.0
        else:
            mot_y = KB_VERTICAL

        # Step 5: ground/air multipliers
        if t.on_ground:
            mot_x *= KB_HORIZONTAL_ON_GROUND
            mot_y *= KB_VERTICAL_ON_GROUND
            mot_z *= KB_HORIZONTAL_ON_GROUND
        else:
            mot_y *= KB_VERTICAL_IN_AIR

        # Step 6: KB enchant level
        if a.kb_enchant_level > 0:
            ench_lvl = a.kb_enchant_level + 1
            mot_x *= ench_lvl
            mot_z *= ench_lvl
            mot_x /= KB_HORIZONTAL_ENCHANT_DEDUCTION
            mot_z /= KB_HORIZONTAL_ENCHANT_DEDUCTION

        # Step 7: sprint multiplier (Kitara: shouldDealSprintKnockback)
        if a.sprinting:
            mot_x *= KB_HORIZONTAL_SPRINTING
            mot_y *= KB_VERTICAL_SPRINTING
            mot_z *= KB_HORIZONTAL_SPRINTING

        # Step 8: sword blocking KB reduction
        if t.sword_blocking:
            mot_x *= KB_SWORD_BLOCK_HORIZONTAL
            mot_y *= KB_SWORD_BLOCK_VERTICAL
            mot_z *= KB_SWORD_BLOCK_HORIZONTAL

        # Step 9: vertical limit (disabled by default)
        if KB_VERTICAL_LIMIT_ENABLE:
            y_off = t.y - a.y
            if y_off > KB_VERTICAL_LIMIT:
                mot_y = 0.0

        # Apply KB to target
        t.vx = mot_x
        t.vy = mot_y
        t.vz = mot_z
        t.on_ground = False
        event.knockback_applied = True

        # Step 10: attacker slowdown + sprint cancel
        if a.sprinting:
            a.vx *= KB_SLOWDOWN
            a.vz *= KB_SLOWDOWN
            if KB_CANCEL_SPRINT:
                a.sprinting = False

        return a, t, event

    # ─────────────────────────────────────────────────────────
    #  Full tick: both players move + optional combat
    # ─────────────────────────────────────────────────────────

    def step(
        self,
        state_a: PlayerState,
        state_b: PlayerState,
        actions_a: dict,
        actions_b: dict,
        friction: float = None,
    ) -> tuple[PlayerState, PlayerState, list[CombatEvent]]:
        """Full 1v1 tick: both players move, then combat.

        Args:
            state_a, state_b: player states
            actions_a, actions_b: action dicts with keys:
                forward, strafe, jump, sprint, sneak, attack (bool)
            friction: block friction

        Returns:
            (new_a, new_b, events)
        """
        events = []

        # 1) Tick i-frames down
        a = state_a.copy()
        b = state_b.copy()
        if a.hurt_time > 0:
            a.hurt_time -= 1
        if b.hurt_time > 0:
            b.hurt_time -= 1

        # 2) Movement
        a = self.step_movement(a, actions_a, friction)
        b = self.step_movement(b, actions_b, friction)

        # 3) Combat (both can attack)
        attack_a = bool(actions_a.get("attack", False))
        attack_b = bool(actions_b.get("attack", False))

        if attack_a:
            a, b, ev = self.step_combat(a, b, True)
            events.append(ev)
        if attack_b:
            b, a, ev = self.step_combat(b, a, True)
            events.append(ev)

        return a, b, events


# ── Helper functions ──


def state_from_csv_row(row: dict, prefix: str = "pre") -> PlayerState:
    """Create a PlayerState from a CSV row dict (pandas row or dict).

    Args:
        row: dict-like with column names
        prefix: "pre" or "post" for which state snapshot to read
    """
    p = prefix
    state = PlayerState(
        x=float(row.get(f"{p}_x", 0)),
        y=float(row.get(f"{p}_y", 0)),
        z=float(row.get(f"{p}_z", 0)),
        vx=float(row.get(f"{p}_vx", 0)),
        vy=float(row.get(f"{p}_vy", 0)),
        vz=float(row.get(f"{p}_vz", 0)),
        yaw=float(row.get(f"{p}_yaw", 0)),
        pitch=float(row.get(f"{p}_pitch", 0)),
        on_ground=bool(int(row.get(f"{p}_onGround", 1))),
        sprinting=bool(int(row.get(f"{p}_sprinting", 0))),
        sneaking=bool(int(row.get(f"{p}_sneaking", 0))),
        health=float(row.get(f"{p}_health", 20)),
        absorption=float(row.get(f"{p}_absorption", 0)),
        armor=int(row.get(f"{p}_armor", 20)),
        hurt_time=int(row.get(f"{p}_hurt_time", 0)),
    )

    # Enhanced columns (may not exist in old recordings)
    if f"{p}_horizontal_collision" in row:
        state.horizontal_collision = bool(int(row[f"{p}_horizontal_collision"]))
    if f"{p}_vertical_collision" in row:
        state.vertical_collision = bool(int(row[f"{p}_vertical_collision"]))
    if f"{p}_fall_distance" in row:
        state.fall_distance = float(row[f"{p}_fall_distance"])
    if f"{p}_speed_amplifier" in row:
        state.speed_amplifier = int(row[f"{p}_speed_amplifier"])
    if f"{p}_strength_amplifier" in row:
        state.strength_amplifier = int(row[f"{p}_strength_amplifier"])
    if f"{p}_sharpness_level" in row:
        state.sharpness_level = int(row[f"{p}_sharpness_level"])

    return state


def actions_from_csv_row(row: dict) -> dict:
    """Extract action dict from a CSV row."""
    return {
        "forward": float(row.get("act_fwd", 0)) - float(row.get("act_back", 0)),
        "strafe": float(row.get("act_right", 0)) - float(row.get("act_left", 0)),
        "jump": bool(int(row.get("act_jump", 0))),
        "sprint": bool(int(row.get("act_sprint", 0))),
        "sneak": bool(int(row.get("act_sneak", 0))),
        "attack": bool(int(row.get("act_attack", 0))),
    }
