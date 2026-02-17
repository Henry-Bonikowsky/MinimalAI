"""Agent/entity state management for the combat simulator."""

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum


class Alliance(IntEnum):
    ENEMY = -1
    NEUTRAL = 0
    ALLY = 1


class WeaponType(IntEnum):
    FIST = 0
    SWORD = 1
    AXE = 2


WEAPON_STATS = {
    WeaponType.FIST: {"damage": 1.0, "speed": 4.0, "reach": 3.0},
    WeaponType.SWORD: {"damage": 7.0, "speed": 1.6, "reach": 3.0},
    WeaponType.AXE: {"damage": 9.0, "speed": 1.0, "reach": 3.0},
}

# Minecraft movement constants (blocks per tick)
WALK_SPEED = 0.2158  # 4.317 m/s / 20 TPS
SPRINT_SPEED = 0.2806  # 5.612 m/s / 20 TPS
JUMP_VELOCITY = 0.42  # blocks/tick upward
GRAVITY = 0.08  # blocks/tick^2 downward
DRAG_HORIZONTAL = 0.91  # velocity multiplier per tick (on ground)
DRAG_AIR = 0.91  # air drag
DRAG_VERTICAL = 0.98  # vertical drag

MAX_HEALTH = 20.0
ARENA_SIZE = 30.0  # blocks, half-width of arena


@dataclass
class SigilBind:
    """Represents one Arcane Sigils ability bind slot."""
    available: bool = False
    cooldown_remaining: int = 0
    cooldown_max: int = 100  # ticks
    damage: float = 0.0
    buff_amplifier: float = 0.0
    is_aoe: bool = False
    aoe_radius: float = 0.0


@dataclass
class Agent:
    """A combat entity in the simulator."""

    # Identity
    agent_id: int = 0
    alliance: Alliance = Alliance.ENEMY

    # Position & movement (2D for v1, Y for jump/crit)
    x: float = 0.0
    z: float = 0.0
    y: float = 0.0  # vertical
    vx: float = 0.0
    vz: float = 0.0
    vy: float = 0.0
    facing_angle: float = 0.0  # radians, 0 = +Z direction

    # State
    health: float = MAX_HEALTH
    armor: float = 0.0
    absorption: float = 0.0
    on_ground: bool = True
    is_sprinting: bool = False
    is_sneaking: bool = False
    is_blocking: bool = False
    is_alive: bool = True

    # Combat
    weapon: WeaponType = WeaponType.SWORD
    attack_cooldown: int = 0  # ticks until next full-power attack
    damage_dealt_this_tick: float = 0.0
    damage_taken_this_tick: float = 0.0
    combo_counter: int = 0
    last_hit_tick: int = -100
    last_hurt_tick: int = -100
    kills: int = 0

    # Sigils (4 bind slots, always present)
    sigil_binds: list = field(default_factory=lambda: [SigilBind() for _ in range(4)])
    sigil_damage_amp: float = 0.0  # multiplicative bonus
    sigil_damage_reduction: float = 0.0  # percentage reduction

    # Targeting
    target_id: int = -1  # which entity this agent is targeting

    def reset(self, x: float = 0.0, z: float = 0.0, facing: float = 0.0):
        """Reset agent to starting state."""
        self.x = x
        self.z = z
        self.y = 0.0
        self.vx = 0.0
        self.vz = 0.0
        self.vy = 0.0
        self.facing_angle = facing
        self.health = MAX_HEALTH
        self.absorption = 0.0
        self.on_ground = True
        self.is_sprinting = False
        self.is_sneaking = False
        self.is_blocking = False
        self.is_alive = True
        self.attack_cooldown = 0
        self.damage_dealt_this_tick = 0.0
        self.damage_taken_this_tick = 0.0
        self.combo_counter = 0
        self.last_hit_tick = -100
        self.last_hurt_tick = -100
        self.kills = 0
        self.target_id = -1
        self.sigil_damage_amp = 0.0
        self.sigil_damage_reduction = 0.0
        for bind in self.sigil_binds:
            bind.cooldown_remaining = 0

    @property
    def cooldown_progress(self) -> float:
        """Returns 0.0 (just attacked) to 1.0 (full cooldown)."""
        weapon_stats = WEAPON_STATS[self.weapon]
        max_cooldown = int(20.0 / weapon_stats["speed"])
        if max_cooldown == 0:
            return 1.0
        elapsed = max_cooldown - self.attack_cooldown
        return min(1.0, max(0.0, (elapsed + 0.5) / max_cooldown))

    @property
    def damage_multiplier(self) -> float:
        """Damage multiplier based on attack cooldown progress."""
        progress = self.cooldown_progress
        return 0.2 + (progress ** 2) * 0.8

    @property
    def move_speed(self) -> float:
        """Current movement speed in blocks/tick."""
        if self.is_sneaking:
            return WALK_SPEED * 0.3
        if self.is_sprinting:
            return SPRINT_SPEED
        return WALK_SPEED

    def get_self_state(self) -> np.ndarray:
        """Return the 30-dim self state vector."""
        state = np.zeros(30, dtype=np.float32)

        # Health/defense (4)
        state[0] = self.health / MAX_HEALTH
        state[1] = 1.0  # hunger (always full in sim)
        state[2] = self.absorption / 20.0
        state[3] = self.armor / 20.0

        # Velocity/ground (2)
        speed = np.sqrt(self.vx**2 + self.vz**2)
        state[4] = speed / SPRINT_SPEED
        state[5] = 1.0 if self.on_ground else 0.0

        # Look direction (4)
        state[6] = np.cos(self.facing_angle)  # facing X
        state[7] = np.sin(self.facing_angle)  # facing Z
        state[8] = 0.0  # pitch X (flat in 2D sim)
        state[9] = 0.0  # pitch Y

        # Combat state (4)
        state[10] = self.cooldown_progress
        state[11] = 1.0 if self.is_sprinting else 0.0
        state[12] = 1.0 if self.is_sneaking else 0.0
        state[13] = 1.0 if self.is_blocking else 0.0

        # Potion effects placeholder (12) - zeros for now
        # state[14:26] = 0.0

        # Sigil self-buffs (4)
        state[26] = self.sigil_damage_amp
        state[27] = self.sigil_damage_reduction
        state[28] = 0.0  # reserved
        state[29] = 0.0  # reserved

        return state

    def get_entity_features(self, observer: 'Agent') -> np.ndarray:
        """Return the 20-dim feature vector of this entity relative to observer."""
        features = np.zeros(20, dtype=np.float32)

        # Alliance (1)
        features[0] = float(self.alliance)

        # Relative position XYZ (3)
        dx = self.x - observer.x
        dz = self.z - observer.z
        dy = self.y - observer.y
        features[1] = dx / ARENA_SIZE
        features[2] = dz / ARENA_SIZE
        features[3] = dy / 10.0

        # Distance (1)
        dist = np.sqrt(dx**2 + dz**2 + dy**2)
        features[4] = dist / ARENA_SIZE

        # Health, armor (2)
        features[5] = self.health / MAX_HEALTH
        features[6] = self.armor / 20.0

        # Velocity (3)
        features[7] = self.vx / SPRINT_SPEED
        features[8] = self.vz / SPRINT_SPEED
        features[9] = self.vy / JUMP_VELOCITY if JUMP_VELOCITY > 0 else 0.0

        # Facing direction (2)
        features[10] = np.cos(self.facing_angle)
        features[11] = np.sin(self.facing_angle)

        # Is facing observer (1)
        angle_to_observer = np.arctan2(
            observer.z - self.z, observer.x - self.x
        )
        angle_diff = abs(self.facing_angle - angle_to_observer)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        features[12] = 1.0 if angle_diff < np.pi / 4 else 0.0  # within 45 degrees

        # Weapon type, blocking (2)
        features[13] = float(self.weapon) / 2.0
        features[14] = 1.0 if self.is_blocking else 0.0

        # Attack cooldown (1)
        features[15] = self.cooldown_progress

        # Sigil damage amp/reduction on this entity (2)
        features[16] = self.sigil_damage_amp
        features[17] = self.sigil_damage_reduction

        # Is current target (1)
        features[18] = 1.0 if observer.target_id == self.agent_id else 0.0

        # On ground (1)
        features[19] = 1.0 if self.on_ground else 0.0

        return features

    def get_sigil_state(self) -> np.ndarray:
        """Return the 12-dim sigil state vector (4 binds x 3)."""
        state = np.zeros(12, dtype=np.float32)
        for i, bind in enumerate(self.sigil_binds):
            state[i * 3] = 1.0 if bind.available and bind.cooldown_remaining == 0 else 0.0
            state[i * 3 + 1] = bind.cooldown_remaining / max(bind.cooldown_max, 1)
            state[i * 3 + 2] = bind.cooldown_max / 200.0  # normalize
        return state
