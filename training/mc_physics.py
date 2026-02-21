"""Vanilla Minecraft 1.21 physics simulator calibrated from server recordings.

Implements the core movement and combat equations from vanilla MC source:
- LivingEntity.travel() movement physics
- Player.attack() damage + knockback
- Armor/enchant damage reduction
- I-frame (invulnerability) system

Used for fast batch RL training (thousands of episodes/sec) without a live server.
The existing training/combat_sim/ is a 1.8-style sim; this targets real 1.21 mechanics.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Vanilla constants (defaults, tunable via calibrate_physics.py) ──

GRAVITY = 0.08            # blocks/tick^2
AIR_DRAG = 0.98           # vertical drag multiplier
GROUND_FRICTION = 0.6     # default block friction (stone, dirt, etc.)
DRAG_FACTOR = 0.91        # horizontal drag = friction * 0.91 (vanilla)
CALIBRATED_DRAG_FACTOR = 0.95  # calibrated from server recordings (1% lower RMSE)
JUMP_IMPULSE = 0.42       # vy set on jump
SPRINT_JUMP_KICK = 0.2    # forward boost on sprint-jump
SNEAK_SPEED_MULT = 0.3    # sneaking speed multiplier
SPRINT_SPEED_MULT = 1.3   # sprinting speed multiplier vs walk

# Base movement acceleration (blocks/tick^2 on ground, attribute-based)
BASE_WALK_ACCEL = 0.1     # generic.movement_speed base * 10
PLAYER_WALK_SPEED = 0.1   # attribute base value

# Combat constants
ATTACK_REACH = 3.0        # blocks (player reach)
BASE_SWORD_DAMAGE = 7.0   # netherite sword base
SHARPNESS_BONUS_PER_LEVEL = 0.5  # + 0.5 * level + 0.5 (1.9+ formula)
CRIT_MULTIPLIER = 1.5     # falling + not on ground
I_FRAME_TICKS = 10        # vanilla 1.9+ invulnerability ticks (0.5s)

# Knockback
KB_BASE = 0.4             # base horizontal KB
KB_SPRINT_BONUS = 0.6     # extra KB when attacker sprinting
KB_VERTICAL = 0.4         # vertical KB impulse
KB_VERTICAL_CAP = 0.4     # max vertical KB


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
    armor: int = 20          # full netherite = 20
    armor_toughness: float = 12.0  # full netherite = 12
    hurt_time: int = 0       # i-frames countdown

    # Effects & enchants
    speed_amplifier: int = -1       # -1 = no effect, 0 = Speed I, 1 = Speed II
    strength_amplifier: int = -1    # -1 = no effect
    sharpness_level: int = 5        # Sharpness V default

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
        """Process one attack attempt.

        Args:
            attacker: attacker state (not mutated)
            target: target state (not mutated)
            attack_action: whether the attack button is pressed

        Returns:
            (new_attacker, new_target, event)
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

        # I-frame check
        if t.hurt_time > 0:
            return a, t, event

        # ── Damage calculation (1.9+) ──
        base_damage = BASE_SWORD_DAMAGE
        # Sharpness: +0.5 * level + 0.5 for level >= 1
        if a.sharpness_level > 0:
            base_damage += 0.5 * a.sharpness_level + 0.5

        # Strength effect
        if a.strength_amplifier >= 0:
            base_damage += 3.0 * (a.strength_amplifier + 1)

        # Critical hit: falling + not on ground + not sprinting (vanilla crit)
        # Note: in practice, sprint-crits work on servers. Keep it simple.
        critical = not a.on_ground and a.vy < 0
        if critical:
            base_damage *= CRIT_MULTIPLIER
            event.critical = True

        # ── Armor reduction ──
        raw_damage = base_damage
        armor_defense = _armor_reduction(raw_damage, t.armor, t.armor_toughness)
        actual_damage = raw_damage * (1.0 - armor_defense)

        actual_damage = max(0.0, actual_damage)
        event.damage = actual_damage

        # ── Apply damage ──
        if t.absorption > 0:
            absorbed = min(t.absorption, actual_damage)
            t.absorption -= absorbed
            actual_damage -= absorbed
        t.health = max(0.0, t.health - actual_damage)
        t.hurt_time = I_FRAME_TICKS

        # ── Knockback ──
        event.attacker_sprinting = a.sprinting
        if dist > 1e-6:
            nx = dx / dist
            nz = dz / dist

            kb_h = KB_BASE
            if a.sprinting:
                kb_h += KB_SPRINT_BONUS

            # Halve target velocity first (vanilla mechanic)
            t.vx *= 0.5
            t.vz *= 0.5

            t.vx += nx * kb_h
            t.vz += nz * kb_h
            t.vy = min(t.vy + KB_VERTICAL, KB_VERTICAL_CAP)
            t.on_ground = False
            event.knockback_applied = True

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

def _armor_reduction(damage: float, armor: int, toughness: float) -> float:
    """Vanilla armor damage reduction formula (1.9+).

    reduction = min(20, max(armor/5, armor - 4*damage/(2+toughness))) / 25
    """
    if armor <= 0:
        return 0.0
    inner = max(armor / 5.0, armor - 4.0 * damage / (2.0 + toughness))
    defense = min(20.0, inner)
    return defense / 25.0


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
        armor_toughness=float(row.get(f"{p}_armor_toughness", 12)),
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
