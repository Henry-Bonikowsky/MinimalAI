"""Agent/entity state management for the combat simulator (v2 - 1.8 PvP).

Implements kit inventory, status effects, 1.8 weapon stats (no cooldown),
netherite armor with Protection IV, and expanded sigil state for 12 slots.
"""

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


# 1.8 PvP: no attack cooldown, damage is flat per hit
WEAPON_STATS = {
    WeaponType.FIST: {"damage": 1.0, "reach": 3.0},
    WeaponType.SWORD: {"damage": 7.0, "reach": 3.0, "sharpness_v": 3.0},  # 7 + 3 = 10
    WeaponType.AXE: {"damage": 9.0, "reach": 3.0, "sharpness_v": 3.0},    # 9 + 3 = 12
}

# Netherite armor stats
ARMOR_VALUE = 20.0      # full netherite set
ARMOR_TOUGHNESS = 12.0  # full netherite set
PROTECTION_EPF = 16     # Protection IV on all 4 pieces (4 * 4)

# Minecraft movement constants (blocks per tick)
WALK_SPEED = 0.2158     # 4.317 m/s / 20 TPS
SPRINT_SPEED = 0.2806   # 5.612 m/s / 20 TPS
JUMP_VELOCITY = 0.42    # blocks/tick upward
GRAVITY = 0.08          # blocks/tick^2 downward
DRAG_HORIZONTAL = 0.91  # velocity multiplier per tick (on ground)
DRAG_AIR = 0.91         # air drag
DRAG_VERTICAL = 0.98    # vertical drag

MAX_HEALTH = 20.0
ARENA_SIZE = 30.0       # blocks, half-width of arena

# Kit constants
KIT_GOLDEN_APPLES = 8   # golden apples per kit
KIT_HEALTH_POTS = 6     # health pots per kit
KIT_ENDER_PEARLS = 4    # ender pearls per kit
KIT_TOTEMS = 1           # totems per kit
GAP_EAT_TICKS = 32      # 1.6s to eat a golden apple
GAP_ABSORPTION = 4.0    # 2 absorption hearts = 4 HP
GAP_REGEN_DURATION = 100 # Regen II for 5s = 100 ticks
PEARL_COOLDOWN = 20     # ticks between pearl throws
PEARL_SPEED = 1.0       # blocks/tick travel speed (simplified)
PEARL_SELF_DAMAGE = 3.0 # damage to self on pearl land
HEALTH_POT_HEAL = 8.0   # Instant Health II
POT_COOLDOWN = 20       # ticks between pot throws (1 second)
TOTEM_REGEN_DURATION = 100  # 5s regen after totem pop
HURT_IMMUNE_TICKS = 5       # invulnerability frames after taking a hit

# Sigil constants
NUM_SIGIL_SLOTS = 12    # 6 Pharaoh + 6 Seasonal


@dataclass
class StatusEffect:
    """A timed status effect on an agent."""
    effect_type: str  # "speed", "regen", "resistance", "wither", "slowness", "weakness", "stun", "absorption"
    amplifier: int = 0  # 0 = level I, 1 = level II, etc.
    duration_ticks: int = 0

    def tick(self) -> bool:
        """Tick down duration. Returns True if still active."""
        if self.duration_ticks > 0:
            self.duration_ticks -= 1
        return self.duration_ticks > 0


@dataclass
class Kit:
    """Inventory items for PvP kit."""
    golden_apples: int = KIT_GOLDEN_APPLES
    health_pots: int = KIT_HEALTH_POTS
    ender_pearls: int = KIT_ENDER_PEARLS
    totems: int = KIT_TOTEMS

    # Eating state
    eating_ticks: int = 0     # counts up to GAP_EAT_TICKS
    is_eating: bool = False

    # Pot state
    pot_cooldown: int = 0     # ticks until next pot allowed

    # Pearl state
    pearl_cooldown: int = 0   # ticks until next pearl allowed

    # Totem state
    totem_in_offhand: bool = True  # simplified: always in offhand if available

    def reset(self):
        self.golden_apples = KIT_GOLDEN_APPLES
        self.health_pots = KIT_HEALTH_POTS
        self.ender_pearls = KIT_ENDER_PEARLS
        self.totems = KIT_TOTEMS
        self.eating_ticks = 0
        self.is_eating = False
        self.pot_cooldown = 0
        self.pearl_cooldown = 0
        self.totem_in_offhand = True


@dataclass
class SigilSlot:
    """One of 12 sigil slots (see sigils.py for implementations)."""
    sigil_type: str = ""       # e.g. "pharaoh_curse", "sandstorm", etc.
    equipped: bool = False
    cooldown_remaining: int = 0
    cooldown_max: int = 200    # ticks
    tier: int = 1              # 1-5, affects effect strength
    activation_type: str = ""  # "passive", "auto_attack", "auto_defense", "ability"
    last_used_tick: int = -1000


@dataclass
class Agent:
    """A combat entity in the 1.8 PvP simulator."""

    # Identity
    agent_id: int = 0
    alliance: Alliance = Alliance.ENEMY

    # Position & movement
    x: float = 0.0
    z: float = 0.0
    y: float = 0.0  # vertical
    vx: float = 0.0
    vz: float = 0.0
    vy: float = 0.0
    facing_angle: float = 0.0  # radians, 0 = +X direction

    # State
    health: float = MAX_HEALTH
    armor: float = ARMOR_VALUE
    armor_toughness: float = ARMOR_TOUGHNESS
    protection_epf: int = PROTECTION_EPF
    absorption: float = 0.0
    on_ground: bool = True
    is_sprinting: bool = False
    is_sneaking: bool = False
    is_blocking: bool = False  # 1.8 sword block
    is_alive: bool = True

    # Combat (1.8: no cooldown)
    weapon: WeaponType = WeaponType.SWORD
    attack_tick_cooldown: int = 0  # 1 tick minimum between hits (anti-cheat)
    hurt_immune_ticks: int = 0     # invulnerability frames after taking damage
    damage_dealt_this_tick: float = 0.0
    damage_taken_this_tick: float = 0.0
    combo_counter: int = 0
    last_hit_tick: int = -100
    last_hurt_tick: int = -100
    kills: int = 0
    sprint_reset_ready: bool = False  # True after toggling sprint off/on between hits
    cps: float = 10.0  # clicks per second (for observation, updated by env)

    # Kit
    kit: Kit = field(default_factory=Kit)

    # Status effects
    status_effects: list = field(default_factory=list)

    # Sigil state (12 slots)
    sigil_slots: list = field(default_factory=lambda: [SigilSlot() for _ in range(NUM_SIGIL_SLOTS)])
    sigil_damage_amp: float = 0.0
    sigil_damage_reduction: float = 0.0

    # Sigil-specific combat state
    invuln_hits: int = 0              # Divine Intervention remaining invuln hits
    kings_brace_charges: int = 0      # King's Brace charge count (max 100)
    pharaoh_marks: set = field(default_factory=set)  # set of agent_ids marked by Pharaoh's Mark
    no_knockback_ticks: int = 0       # Quick Sand: no KB received
    negative_effect_immunity: float = 0.0  # Ancient Crown: 0-1.0 chance to resist
    mummies: list = field(default_factory=list)  # Royal Guard spawned mummies

    # Targeting
    target_id: int = -1

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
        self.armor = ARMOR_VALUE
        self.armor_toughness = ARMOR_TOUGHNESS
        self.protection_epf = PROTECTION_EPF
        self.absorption = 0.0
        self.on_ground = True
        self.is_sprinting = False
        self.is_sneaking = False
        self.is_blocking = False
        self.is_alive = True
        self.weapon = WeaponType.SWORD
        self.attack_tick_cooldown = 0
        self.hurt_immune_ticks = 0
        self.damage_dealt_this_tick = 0.0
        self.damage_taken_this_tick = 0.0
        self.combo_counter = 0
        self.last_hit_tick = -100
        self.last_hurt_tick = -100
        self.kills = 0
        self.sprint_reset_ready = False
        self.cps = 10.0
        self.target_id = -1
        self.sigil_damage_amp = 0.0
        self.sigil_damage_reduction = 0.0
        self.invuln_hits = 0
        self.kings_brace_charges = 0
        self.pharaoh_marks = set()
        self.no_knockback_ticks = 0
        self.negative_effect_immunity = 0.0
        self.mummies = []
        self.status_effects = []
        self.kit.reset()
        for slot in self.sigil_slots:
            slot.cooldown_remaining = 0
            slot.last_used_tick = -1000

    @property
    def move_speed(self) -> float:
        """Current movement speed in blocks/tick."""
        base = WALK_SPEED
        if self.is_sneaking:
            base = WALK_SPEED * 0.3
        elif self.is_sprinting:
            base = SPRINT_SPEED
        # Speed effect
        for eff in self.status_effects:
            if eff.effect_type == "speed":
                base *= 1.0 + 0.2 * (eff.amplifier + 1)
            elif eff.effect_type == "slowness":
                base *= max(0.0, 1.0 - 0.15 * (eff.amplifier + 1))
        return base

    def has_effect(self, effect_type: str) -> bool:
        return any(e.effect_type == effect_type and e.duration_ticks > 0 for e in self.status_effects)

    def get_effect(self, effect_type: str):
        for e in self.status_effects:
            if e.effect_type == effect_type and e.duration_ticks > 0:
                return e
        return None

    def add_effect(self, effect_type: str, amplifier: int = 0, duration: int = 0):
        """Add or refresh a status effect."""
        # Check negative effect immunity (Ancient Crown)
        if effect_type in ("slowness", "weakness", "wither", "stun"):
            if self.negative_effect_immunity > 0 and np.random.random() < self.negative_effect_immunity:
                return  # Resisted
        for e in self.status_effects:
            if e.effect_type == effect_type:
                e.amplifier = max(e.amplifier, amplifier)
                e.duration_ticks = max(e.duration_ticks, duration)
                return
        self.status_effects.append(StatusEffect(effect_type, amplifier, duration))

    def remove_effect(self, effect_type: str):
        self.status_effects = [e for e in self.status_effects if e.effect_type != effect_type]

    def clear_defensive_buffs(self):
        """Strip all defensive buffs (for Cleopatra sigil)."""
        defensive = {"resistance", "absorption", "regen", "speed"}
        self.status_effects = [e for e in self.status_effects if e.effect_type not in defensive]
        self.absorption = 0.0
        self.sigil_damage_reduction = 0.0
        self.invuln_hits = 0
        self.kings_brace_charges = 0

    def tick_effects(self):
        """Tick all status effects and apply per-tick effects."""
        remaining = []
        for eff in self.status_effects:
            if eff.effect_type == "regen" and eff.duration_ticks > 0:
                # Regen II heals 1 HP every 12 ticks, Regen III every 6 ticks
                interval = max(1, 25 // (eff.amplifier + 1))
                if eff.duration_ticks % interval == 0:
                    self.health = min(MAX_HEALTH, self.health + 1.0)
            elif eff.effect_type == "wither" and eff.duration_ticks > 0:
                # Wither deals 1 HP every 40 ticks (level I)
                interval = max(1, 40 // (eff.amplifier + 1))
                if eff.duration_ticks % interval == 0:
                    self.health = max(0.0, self.health - 1.0)
            elif eff.effect_type == "stun" and eff.duration_ticks > 0:
                # Stun prevents actions (handled in env)
                pass
            if eff.tick():
                remaining.append(eff)
        self.status_effects = remaining

    def get_self_state(self) -> np.ndarray:
        """Return the 38-dim self state vector."""
        state = np.zeros(38, dtype=np.float32)

        # Health/defense (4)
        state[0] = self.health / MAX_HEALTH
        state[1] = 1.0  # hunger (always full)
        state[2] = self.absorption / 20.0
        state[3] = self.armor / 20.0

        # Velocity/ground (2)
        speed = np.sqrt(self.vx**2 + self.vz**2)
        state[4] = speed / SPRINT_SPEED if SPRINT_SPEED > 0 else 0.0
        state[5] = 1.0 if self.on_ground else 0.0

        # Look direction (4)
        state[6] = np.cos(self.facing_angle)
        state[7] = np.sin(self.facing_angle)
        state[8] = 0.0  # pitch (flat arena)
        state[9] = 0.0

        # Combat state (4)
        state[10] = 1.0  # 1.8: always ready to attack (no cooldown)
        state[11] = 1.0 if self.is_sprinting else 0.0
        state[12] = 1.0 if self.is_sneaking else 0.0
        state[13] = 1.0 if self.is_blocking else 0.0

        # Status effects summary (6)
        state[14] = 1.0 if self.has_effect("speed") else 0.0
        state[15] = 1.0 if self.has_effect("regen") else 0.0
        state[16] = 1.0 if self.has_effect("resistance") else 0.0
        state[17] = 1.0 if self.has_effect("slowness") else 0.0
        state[18] = 1.0 if self.has_effect("weakness") else 0.0
        state[19] = 1.0 if self.has_effect("stun") else 0.0

        # Kit inventory (4)
        state[20] = self.kit.golden_apples / max(KIT_GOLDEN_APPLES, 1)
        state[21] = self.kit.health_pots / max(KIT_HEALTH_POTS, 1)
        state[22] = self.kit.ender_pearls / max(KIT_ENDER_PEARLS, 1)
        state[23] = self.kit.totems / max(KIT_TOTEMS, 1)

        # Kit state (2)
        state[24] = self.kit.eating_ticks / GAP_EAT_TICKS if GAP_EAT_TICKS > 0 else 0.0
        state[25] = self.kit.pearl_cooldown / PEARL_COOLDOWN if PEARL_COOLDOWN > 0 else 0.0

        # Sigil combat state (4)
        state[26] = self.sigil_damage_amp
        state[27] = self.sigil_damage_reduction
        state[28] = min(1.0, self.kings_brace_charges / 100.0)
        state[29] = min(1.0, self.invuln_hits / 3.0)

        # Weapon (2)
        state[30] = float(self.weapon) / 2.0
        state[31] = 1.0 if self.sprint_reset_ready else 0.0

        # Combo/CPS (2)
        state[32] = min(1.0, self.combo_counter / 10.0)
        state[33] = self.cps / 20.0

        # No-KB and effect immunity (2)
        state[34] = min(1.0, self.no_knockback_ticks / 120.0)
        state[35] = self.negative_effect_immunity

        # Reserved (2)
        state[36] = 0.0
        state[37] = 0.0

        return state

    def get_entity_features(self, observer: 'Agent') -> np.ndarray:
        """Return the 24-dim feature vector of this entity relative to observer."""
        features = np.zeros(24, dtype=np.float32)

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

        # Health, armor, absorption (3)
        features[5] = self.health / MAX_HEALTH
        features[6] = self.armor / 20.0
        features[7] = self.absorption / 20.0

        # Velocity (3)
        features[8] = self.vx / SPRINT_SPEED if SPRINT_SPEED > 0 else 0.0
        features[9] = self.vz / SPRINT_SPEED if SPRINT_SPEED > 0 else 0.0
        features[10] = self.vy / JUMP_VELOCITY if JUMP_VELOCITY > 0 else 0.0

        # Facing direction (2)
        features[11] = np.cos(self.facing_angle)
        features[12] = np.sin(self.facing_angle)

        # Is facing observer (1)
        angle_to_observer = np.arctan2(
            observer.z - self.z, observer.x - self.x
        )
        angle_diff = abs(self.facing_angle - angle_to_observer)
        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
        features[13] = 1.0 if angle_diff < np.pi / 4 else 0.0

        # Weapon type, blocking (2)
        features[14] = float(self.weapon) / 2.0
        features[15] = 1.0 if self.is_blocking else 0.0

        # On ground (1)
        features[16] = 1.0 if self.on_ground else 0.0

        # Is current target (1)
        features[17] = 1.0 if observer.target_id == self.agent_id else 0.0

        # Status effects on this entity (4)
        features[18] = 1.0 if self.has_effect("speed") else 0.0
        features[19] = 1.0 if self.has_effect("stun") else 0.0
        features[20] = 1.0 if self.has_effect("slowness") else 0.0
        features[21] = 1.0 if self.has_effect("resistance") else 0.0

        # Pharaoh mark on this entity (1)
        features[22] = 1.0 if observer.agent_id in self.pharaoh_marks else 0.0

        # Sigil damage reduction (1)
        features[23] = self.sigil_damage_reduction

        return features

    def get_sigil_state(self) -> np.ndarray:
        """Return the 48-dim sigil state vector (12 slots x 4 dims each)."""
        state = np.zeros(NUM_SIGIL_SLOTS * 4, dtype=np.float32)
        for i, slot in enumerate(self.sigil_slots):
            base = i * 4
            # Ready to use (1 if equipped, off cooldown, and is ability type)
            state[base] = 1.0 if (slot.equipped and slot.cooldown_remaining == 0) else 0.0
            # Cooldown progress (0 = ready, 1 = full cooldown)
            state[base + 1] = slot.cooldown_remaining / max(slot.cooldown_max, 1)
            # Tier normalized (0-1)
            state[base + 2] = slot.tier / 5.0 if slot.equipped else 0.0
            # Time since last used (normalized, 0 = just used, 1 = long ago)
            state[base + 3] = 0.0  # updated by env with current tick
        return state
