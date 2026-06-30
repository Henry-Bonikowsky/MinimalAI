"""Numpy-vectorized 1v1 PvP simulator for batch RL training.

Runs N parallel fights simultaneously using pure numpy operations.
10-50x faster than the per-env Python loop in sim_env.py.

All physics match mc_physics.py (Kitara combat model).

v2: Requires FACE_TARGET action for aim (no auto-aim), adds golden apple
eating mechanics with cooldown, adds angle check for attacks.

v3: ENGAGE action (bit 26) replaces separate FACE_TARGET + ATTACK.
When ENGAGE=1, bot faces target and auto-attacks when within configurable reach.
AI only learns movement, sprint, spacing — attack timing is automated.
"""

import numpy as np

from .mc_physics import (
    GRAVITY, AIR_DRAG, GROUND_FRICTION, DRAG_FACTOR, JUMP_IMPULSE,
    SPRINT_JUMP_KICK, SPRINT_SPEED_MULT, SNEAK_SPEED_MULT, PLAYER_WALK_SPEED,
    BASE_SWORD_DAMAGE, SHARPNESS_PER_LEVEL, CRIT_MULTIPLIER,
    SWORD_BLOCKING_DMG_MULT, ATTACK_REACH,
    PROTECTION_DR_TABLE,
    KB_HORIZONTAL, KB_HORIZONTAL_ON_GROUND, KB_HORIZONTAL_SPRINTING,
    KB_HORIZONTAL_INHERIT, KB_HORIZONTAL_FRICTION, KB_HORIZONTAL_ENCHANT_DEDUCTION,
    KB_VERTICAL, KB_VERTICAL_ON_GROUND, KB_VERTICAL_IN_AIR, KB_VERTICAL_SPRINTING,
    KB_VERTICAL_INHERIT, KB_SLOWDOWN, KB_CANCEL_SPRINT,
    KB_SWORD_BLOCK_HORIZONTAL, KB_SWORD_BLOCK_VERTICAL,
)
from .config import (
    # Action indices
    ACT_FORWARD, ACT_BACKWARD, ACT_LEFT, ACT_RIGHT, ACT_JUMP, ACT_SNEAK, ACT_SPRINT,
    ACT_ATTACK, ACT_BLOCK, ACT_EAT_GAP, ACT_THROW_POT, ACT_THROW_PEARL,
    ACT_FACE_TARGET, ACT_ENGAGE, ACT_SIGIL_0,
    NUM_ACTIONS, OBS_DIM,
    # Consumables
    GAP_EAT_TICKS, GAP_COOLDOWN_TICKS, GAP_ABSORPTION, GAP_REGEN_TICKS,
    GAP_REGEN_RATE, GAP_START_COUNT,
    POT_HEAL_AMOUNT, POT_SPLASH_RADIUS, POT_THROW_LOCKOUT, POT_START_COUNT,
    PEARL_COOLDOWN_TICKS, PEARL_SELF_DAMAGE, PEARL_START_COUNT,
    # Combat
    MELEE_ANGLE_COS, I_FRAME_TICKS,
    # Sigils
    SIGIL_BRACE_CHARGE_REQ, SIGIL_BRACE_DR, SIGIL_BRACE_DURATION, SIGIL_BRACE_COOLDOWN,
    SIGIL_CLEO_DMG_AMP, SIGIL_CLEO_DURATION, SIGIL_CLEO_COOLDOWN,
    SIGIL_SAND_DMG_AMP, SIGIL_SAND_DURATION, SIGIL_SAND_COOLDOWN,
    SIGIL_GRACE_DR, SIGIL_GRACE_DURATION, SIGIL_GRACE_COOLDOWN,
)


# ── Combo style functions ──
# Each overrides movement inputs for 1 tick after a sprint-hit
# to automatically reset sprint. Swappable per training run.

def _combo_stap(actions, just_hit):
    """S-tap: backward for 1 tick after hit → creates spacing + rising edge."""
    actions[just_hit, ACT_BACKWARD] = 1
    actions[just_hit, ACT_FORWARD] = 0

def _combo_wtap(actions, just_hit):
    """W-tap: release forward for 1 tick after hit → rising edge on re-press."""
    actions[just_hit, ACT_FORWARD] = 0

def _combo_none(actions, just_hit):
    """No override — raw NN control of movement."""
    pass

COMBO_STYLES = {"stap": _combo_stap, "wtap": _combo_wtap, "none": _combo_none}


class VecPvPSim:
    """Numpy-vectorized N-parallel 1v1 PvP simulator.

    State is stored as parallel numpy arrays of shape (N,).
    All physics computations are batched.

    v3: ENGAGE (bit 26) = face target + auto-attack when in reach.
    Model controls movement/spacing/sprint; attack timing is automated.
    Configurable attack_reach per bot (default 3.0 = max range).
    """

    def __init__(self, n_envs: int, arena_half: float = 30.0,
                 episode_length: int = 1800, seed: int = 0,
                 style: str = "combo", reward_mode: str = "default",
                 attack_reach: float = ATTACK_REACH,
                 combo_style: str = "stap",
                 gauntlet: bool = False):
        self.n = n_envs
        self.arena_half = arena_half
        self.floor_y = 0.0
        self.episode_length = episode_length
        self.style = style
        self.reward_mode = reward_mode  # "default" or "hybrid"
        self.attack_reach = attack_reach  # per-bot engage auto-attack range
        self.gauntlet = gauntlet  # if True, killing opponent respawns them, survivor keeps state
        self.gauntlet_survivor = 'b'  # which player survives: 'a' or 'b'
        if combo_style not in COMBO_STYLES:
            raise ValueError(f"Unknown combo_style '{combo_style}'. Options: {list(COMBO_STYLES.keys())}")
        self.combo_style = combo_style
        self._combo_fn = COMBO_STYLES[combo_style]
        self._reward_breakdown = self._empty_breakdown()
        self.rng = np.random.default_rng(seed)

        # Player A (agent) state arrays
        self.ax = np.zeros(n_envs)
        self.ay = np.zeros(n_envs)
        self.az = np.zeros(n_envs)
        self.avx = np.zeros(n_envs)
        self.avy = np.zeros(n_envs)
        self.avz = np.zeros(n_envs)
        self.ayaw = np.zeros(n_envs)
        self.a_on_ground = np.ones(n_envs, dtype=bool)
        self.a_sprinting = np.zeros(n_envs, dtype=bool)
        self.a_health = np.full(n_envs, 20.0)
        self.a_absorption = np.zeros(n_envs)
        self.a_hurt_time = np.zeros(n_envs, dtype=np.int32)
        self.a_fall_dist = np.zeros(n_envs)
        self.a_prev_forward = np.zeros(n_envs, dtype=bool)  # W-tap tracking

        # Player A eating state
        self.a_eating_ticks = np.zeros(n_envs, dtype=np.int32)
        self.a_eat_cooldown = np.zeros(n_envs, dtype=np.int32)
        self.a_regen_ticks = np.zeros(n_envs, dtype=np.int32)
        self.a_gapple_count = np.full(n_envs, GAP_START_COUNT, dtype=np.int32)

        # Player A splash pot state
        self.a_pot_count = np.full(n_envs, POT_START_COUNT, dtype=np.int32)
        self.a_pot_lockout = np.zeros(n_envs, dtype=np.int32)  # can't attack during throw

        # Player A ender pearl state
        self.a_pearl_count = np.full(n_envs, PEARL_START_COUNT, dtype=np.int32)
        self.a_pearl_cooldown = np.zeros(n_envs, dtype=np.int32)

        # Player A blocking state (1.8 sword block)
        self.a_blocking = np.zeros(n_envs, dtype=bool)

        # Player B (opponent) state arrays
        self.bx = np.zeros(n_envs)
        self.by = np.zeros(n_envs)
        self.bz = np.zeros(n_envs)
        self.bvx = np.zeros(n_envs)
        self.bvy = np.zeros(n_envs)
        self.bvz = np.zeros(n_envs)
        self.byaw = np.zeros(n_envs)
        self.b_on_ground = np.ones(n_envs, dtype=bool)
        self.b_sprinting = np.zeros(n_envs, dtype=bool)
        self.b_health = np.full(n_envs, 20.0)
        self.b_absorption = np.zeros(n_envs)
        self.b_hurt_time = np.zeros(n_envs, dtype=np.int32)
        self.b_fall_dist = np.zeros(n_envs)
        self.b_prev_forward = np.zeros(n_envs, dtype=bool)  # W-tap tracking

        # Player B eating state
        self.b_eating_ticks = np.zeros(n_envs, dtype=np.int32)
        self.b_eat_cooldown = np.zeros(n_envs, dtype=np.int32)
        self.b_regen_ticks = np.zeros(n_envs, dtype=np.int32)
        self.b_gapple_count = np.full(n_envs, GAP_START_COUNT, dtype=np.int32)

        # Player B splash pot state
        self.b_pot_count = np.full(n_envs, POT_START_COUNT, dtype=np.int32)
        self.b_pot_lockout = np.zeros(n_envs, dtype=np.int32)

        # Player B ender pearl state
        self.b_pearl_count = np.full(n_envs, PEARL_START_COUNT, dtype=np.int32)
        self.b_pearl_cooldown = np.zeros(n_envs, dtype=np.int32)

        # Player B blocking state (1.8 sword block)
        self.b_blocking = np.zeros(n_envs, dtype=bool)

        # Sigil state — King's Brace
        self.a_brace_charges = np.full(n_envs, 15, dtype=np.int32)
        self.b_brace_charges = np.full(n_envs, 15, dtype=np.int32)
        self.a_brace_timer = np.zeros(n_envs, dtype=np.int32)
        self.b_brace_timer = np.zeros(n_envs, dtype=np.int32)
        self.a_brace_cd = np.zeros(n_envs, dtype=np.int32)
        self.b_brace_cd = np.zeros(n_envs, dtype=np.int32)

        # Sigil state — Cleopatra (debuff on target)
        self.a_cleo_on_b = np.zeros(n_envs, dtype=np.int32)
        self.b_cleo_on_a = np.zeros(n_envs, dtype=np.int32)
        self.a_cleo_cd = np.zeros(n_envs, dtype=np.int32)
        self.b_cleo_cd = np.zeros(n_envs, dtype=np.int32)

        # Sigil state — Quick Sand (buff on self)
        self.a_sand_timer = np.zeros(n_envs, dtype=np.int32)
        self.b_sand_timer = np.zeros(n_envs, dtype=np.int32)
        self.a_sand_cd = np.zeros(n_envs, dtype=np.int32)
        self.b_sand_cd = np.zeros(n_envs, dtype=np.int32)

        # Sigil state — Nile's Grace (DR on self)
        self.a_grace_timer = np.zeros(n_envs, dtype=np.int32)
        self.b_grace_timer = np.zeros(n_envs, dtype=np.int32)
        self.a_grace_cd = np.zeros(n_envs, dtype=np.int32)
        self.b_grace_cd = np.zeros(n_envs, dtype=np.int32)

        # Episode tracking
        self.tick = np.zeros(n_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(n_envs)
        self.kill_count = np.zeros(n_envs, dtype=np.int32)  # gauntlet kills this episode

        # Combo chain tracking: how many consecutive sprint-hits without opponent hitting back
        self.a_combo_streak = np.zeros(n_envs, dtype=np.int32)
        self.b_combo_streak = np.zeros(n_envs, dtype=np.int32)

        # Counter-hit tracking: tick when player was last hit (for "first to ground" tactic)
        self.a_last_hit_tick = np.full(n_envs, -100, dtype=np.int32)
        self.b_last_hit_tick = np.full(n_envs, -100, dtype=np.int32)

        # Combo style tracking: did this player land a sprint-hit last tick?
        self.a_did_sprint_hit = np.zeros(n_envs, dtype=bool)
        self.b_did_sprint_hit = np.zeros(n_envs, dtype=bool)

        # Last dealt hit tick (for combo tracking)
        self.a_last_dealt_tick = np.full(n_envs, -100, dtype=np.int32)
        self.b_last_dealt_tick = np.full(n_envs, -100, dtype=np.int32)

        # Previous state for reward shaping
        self._prev_dist = np.zeros(n_envs)
        self._prev_health_diff = np.zeros(n_envs)

        # Combat config (constant across all envs)
        self.armor = 20
        self.sharpness = 6
        # Server direct damage uses simplified armor: DR = armor * 0.04, capped at 80%
        self.armor_mult = 1.0 - min(self.armor * 0.04, 0.8)

        # Damage: calibrated to server measurements (Sharp 6 + Prot 5 diamond)
        # Server actual: 1.5 hearts (3.0 HP) per hit, 2 hearts (4.0 HP) per crit
        self.base_dmg_after_armor = 3.0
        self.crit_dmg_after_armor = 4.0

        # Sword blocking: Kitara formula = (1 + raw_damage) * 0.5
        # Using approximate raw damage back-calculated from after-armor values
        self.base_dmg_blocked = self.base_dmg_after_armor * SWORD_BLOCKING_DMG_MULT
        self.crit_dmg_blocked = self.crit_dmg_after_armor * SWORD_BLOCKING_DMG_MULT

    def reset_all(self) -> np.ndarray:
        """Reset all environments. Returns obs (n, OBS_DIM)."""
        self.tick[:] = 0
        self.episode_rewards[:] = 0.0
        self._reset_players(np.arange(self.n))
        return self._get_obs()

    def reset_envs(self, mask: np.ndarray):
        """Reset specific environments by boolean mask."""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return
        self._reset_players(indices)
        self.tick[indices] = 0
        self.episode_rewards[indices] = 0.0

    REWARD_COMPONENTS = [
        "kill_death", "counter_hit", "combo", "clean_trade", "combo_broken",
        "non_sprint_hit", "trade_win", "raw_dmg", "time_pressure",
        "max_range_hit", "spacing",
        "jump_spam", "consumables",
    ]

    def _empty_breakdown(self):
        return {k: 0.0 for k in self.REWARD_COMPONENTS}

    def get_reward_breakdown(self, reset=True):
        """Return accumulated reward breakdown and optionally reset."""
        result = dict(self._reward_breakdown)
        if reset:
            self._reward_breakdown = self._empty_breakdown()
        return result

    def _reset_players(self, idx):
        """Reset player positions for given env indices."""
        n = len(idx)

        # Player A at center
        self.ax[idx] = 0.0
        self.ay[idx] = 0.0
        self.az[idx] = 0.0
        self.avx[idx] = 0.0
        self.avy[idx] = 0.0
        self.avz[idx] = 0.0
        self.ayaw[idx] = self.rng.uniform(-180, 180, n)
        self.a_on_ground[idx] = True
        self.a_sprinting[idx] = False
        self.a_health[idx] = 20.0
        self.a_absorption[idx] = 0.0
        self.a_hurt_time[idx] = 0
        self.a_fall_dist[idx] = 0.0
        self.a_prev_forward[idx] = False
        self.a_eating_ticks[idx] = 0
        self.a_eat_cooldown[idx] = 0
        self.a_regen_ticks[idx] = 0
        self.a_gapple_count[idx] = GAP_START_COUNT
        self.a_pot_count[idx] = POT_START_COUNT
        self.a_pot_lockout[idx] = 0
        self.a_pearl_count[idx] = PEARL_START_COUNT
        self.a_pearl_cooldown[idx] = 0
        self.a_blocking[idx] = False

        # Player B at random distance
        angle = self.rng.uniform(0, 2 * np.pi, n)
        dist = self.rng.uniform(4.0, 8.0, n)
        self.bx[idx] = np.cos(angle) * dist
        self.by[idx] = 0.0
        self.bz[idx] = np.sin(angle) * dist
        self.bvx[idx] = 0.0
        self.bvy[idx] = 0.0
        self.bvz[idx] = 0.0
        self.byaw[idx] = np.degrees(angle + np.pi)
        self.b_on_ground[idx] = True
        self.b_sprinting[idx] = False
        self.b_health[idx] = 20.0
        self.b_absorption[idx] = 0.0
        self.b_hurt_time[idx] = 0
        self.b_fall_dist[idx] = 0.0
        self.b_prev_forward[idx] = False
        self.b_eating_ticks[idx] = 0
        self.b_eat_cooldown[idx] = 0
        self.b_regen_ticks[idx] = 0
        self.b_gapple_count[idx] = GAP_START_COUNT
        self.b_pot_count[idx] = POT_START_COUNT
        self.b_pot_lockout[idx] = 0
        self.b_pearl_count[idx] = PEARL_START_COUNT
        self.b_pearl_cooldown[idx] = 0
        self.b_blocking[idx] = False

        # Sigil state reset
        self.a_brace_charges[idx] = 15; self.b_brace_charges[idx] = 15
        self.a_brace_timer[idx] = 0; self.b_brace_timer[idx] = 0
        self.a_brace_cd[idx] = 0; self.b_brace_cd[idx] = 0
        self.a_cleo_on_b[idx] = 0; self.b_cleo_on_a[idx] = 0
        self.a_cleo_cd[idx] = 0; self.b_cleo_cd[idx] = 0
        self.a_sand_timer[idx] = 0; self.b_sand_timer[idx] = 0
        self.a_sand_cd[idx] = 0; self.b_sand_cd[idx] = 0
        self.a_grace_timer[idx] = 0; self.b_grace_timer[idx] = 0
        self.a_grace_cd[idx] = 0; self.b_grace_cd[idx] = 0

        self.a_combo_streak[idx] = 0
        self.b_combo_streak[idx] = 0
        self.a_last_hit_tick[idx] = -100
        self.b_last_hit_tick[idx] = -100
        self.a_did_sprint_hit[idx] = False
        self.b_did_sprint_hit[idx] = False
        self.a_last_dealt_tick[idx] = -100
        self.b_last_dealt_tick[idx] = -100
        self._prev_dist[idx] = dist
        self._prev_health_diff[idx] = 0.0
        self.kill_count[idx] = 0

    def _reset_b_only(self, idx):
        """Reset only player B (opponent respawn in gauntlet mode). A keeps state."""
        n = len(idx)
        angle = self.rng.uniform(0, 2 * np.pi, n)
        dist = self.rng.uniform(4.0, 8.0, n)
        self.bx[idx] = self.ax[idx] + np.cos(angle) * dist
        self.bz[idx] = self.az[idx] + np.sin(angle) * dist
        self.by[idx] = 0.0
        self.bvx[idx] = 0.0
        self.bvy[idx] = 0.0
        self.bvz[idx] = 0.0
        self.byaw[idx] = np.degrees(angle + np.pi)
        self.b_on_ground[idx] = True
        self.b_sprinting[idx] = False
        self.b_health[idx] = 20.0
        self.b_absorption[idx] = 0.0
        self.b_hurt_time[idx] = 0
        self.b_fall_dist[idx] = 0.0
        self.b_prev_forward[idx] = False
        self.b_eating_ticks[idx] = 0
        self.b_eat_cooldown[idx] = 0
        self.b_regen_ticks[idx] = 0
        self.b_gapple_count[idx] = GAP_START_COUNT
        self.b_pot_count[idx] = POT_START_COUNT
        self.b_pot_lockout[idx] = 0
        self.b_pearl_count[idx] = PEARL_START_COUNT
        self.b_pearl_cooldown[idx] = 0
        self.b_blocking[idx] = False
        # B sigil state reset
        self.b_brace_charges[idx] = 15
        self.b_brace_timer[idx] = 0; self.b_brace_cd[idx] = 0
        self.b_cleo_on_a[idx] = 0; self.b_cleo_cd[idx] = 0
        self.b_sand_timer[idx] = 0; self.b_sand_cd[idx] = 0
        self.b_grace_timer[idx] = 0; self.b_grace_cd[idx] = 0
        # Clear A's debuffs on B
        self.a_cleo_on_b[idx] = 0
        self.a_combo_streak[idx] = 0
        self.b_combo_streak[idx] = 0
        self.b_last_hit_tick[idx] = -100
        self.b_did_sprint_hit[idx] = False
        self.a_did_sprint_hit[idx] = False
        dx = self.ax[idx] - self.bx[idx]
        dz = self.az[idx] - self.bz[idx]
        self._prev_dist[idx] = np.sqrt(dx**2 + dz**2)

    def _reset_a_only(self, idx):
        """Reset only player A (opponent respawn in gauntlet mode). B (model) keeps state."""
        n = len(idx)
        # Respawn A at random position relative to B
        angle = self.rng.uniform(0, 2 * np.pi, n)
        dist = self.rng.uniform(4.0, 8.0, n)
        self.ax[idx] = self.bx[idx] + np.cos(angle) * dist
        self.az[idx] = self.bz[idx] + np.sin(angle) * dist
        self.ay[idx] = 0.0
        self.avx[idx] = 0.0
        self.avy[idx] = 0.0
        self.avz[idx] = 0.0
        self.ayaw[idx] = np.degrees(angle + np.pi)
        self.a_on_ground[idx] = True
        self.a_sprinting[idx] = False
        self.a_health[idx] = 20.0
        self.a_absorption[idx] = 0.0
        self.a_hurt_time[idx] = 0
        self.a_fall_dist[idx] = 0.0
        self.a_prev_forward[idx] = False
        self.a_eating_ticks[idx] = 0
        self.a_eat_cooldown[idx] = 0
        self.a_regen_ticks[idx] = 0
        self.a_gapple_count[idx] = GAP_START_COUNT
        self.a_pot_count[idx] = POT_START_COUNT
        self.a_pot_lockout[idx] = 0
        self.a_pearl_count[idx] = PEARL_START_COUNT
        self.a_pearl_cooldown[idx] = 0
        self.a_blocking[idx] = False
        # A sigil state reset
        self.a_brace_charges[idx] = 15
        self.a_brace_timer[idx] = 0; self.a_brace_cd[idx] = 0
        self.a_cleo_cd[idx] = 0
        self.a_sand_timer[idx] = 0; self.a_sand_cd[idx] = 0
        self.a_grace_timer[idx] = 0; self.a_grace_cd[idx] = 0
        # Clear B's debuffs on A
        self.b_cleo_on_a[idx] = 0
        # Reset combo/hit state for both (fresh fight)
        self.a_combo_streak[idx] = 0
        self.b_combo_streak[idx] = 0
        self.a_last_hit_tick[idx] = -100
        self.a_did_sprint_hit[idx] = False
        self.b_did_sprint_hit[idx] = False
        # Update distance tracking
        dx = self.ax[idx] - self.bx[idx]
        dz = self.az[idx] - self.bz[idx]
        self._prev_dist[idx] = np.sqrt(dx**2 + dz**2)

    def step(self, a_actions: np.ndarray, b_actions: np.ndarray):
        """Step all envs simultaneously.

        Args:
            a_actions: (n, NUM_ACTIONS) int8 MultiBinary for player A
            b_actions: (n, NUM_ACTIONS) int8 MultiBinary for player B

        Returns:
            obs: (n, OBS_DIM) float32
            rewards: (n,) float32  (reward for player A)
            dones: (n,) bool
            infos: dict of arrays
        """
        self.tick += 1

        # ── Mask dead bits ──
        USED_BITS = [0,2,3,4,9,10,11,14,15,16,17,26]  # no backward(1), sneak(5), sprint(6), attack(7=auto), block(8=disabled)
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        for b in USED_BITS:
            mask[b] = 1
        a_actions = a_actions * mask
        b_actions = b_actions * mask

        # Force SPRINT input always on (toggle sprint — but sprint STATE tracks separately)
        # SNEAK off (no benefit in 1v1)
        # ENGAGE (bit 26) is a MODEL DECISION — face target + auto-attack when in reach
        a_actions[:, ACT_SPRINT] = 1
        a_actions[:, ACT_SNEAK] = 0
        b_actions[:, ACT_SPRINT] = 1
        b_actions[:, ACT_SNEAK] = 0

        # ── Combo style override: auto sprint-reset after landing a hit ──
        self._combo_fn(a_actions, self.a_did_sprint_hit)
        self._combo_fn(b_actions, self.b_did_sprint_hit)
        self.a_did_sprint_hit[:] = False
        self.b_did_sprint_hit[:] = False

        # FWD+BACK cancel each other
        both_fb_a = a_actions[:, ACT_FORWARD].astype(bool) & a_actions[:, ACT_BACKWARD].astype(bool)
        a_actions[both_fb_a, ACT_FORWARD] = 0
        a_actions[both_fb_a, ACT_BACKWARD] = 0
        both_fb_b = b_actions[:, ACT_FORWARD].astype(bool) & b_actions[:, ACT_BACKWARD].astype(bool)
        b_actions[both_fb_b, ACT_FORWARD] = 0
        b_actions[both_fb_b, ACT_BACKWARD] = 0

        # ── ENGAGE implies FACE_TARGET (face + auto-attack) ──
        engage_face_a = a_actions[:, ACT_ENGAGE].astype(bool)
        engage_face_b = b_actions[:, ACT_ENGAGE].astype(bool)
        a_actions[engage_face_a, ACT_FACE_TARGET] = 1
        b_actions[engage_face_b, ACT_FACE_TARGET] = 1

        # ── Facing: only update yaw when FACE_TARGET action is active ──
        dx_ab = self.bx - self.ax
        dz_ab = self.bz - self.az

        face_a = a_actions[:, ACT_FACE_TARGET].astype(bool)
        if np.any(face_a):
            target_yaw_a = np.degrees(np.arctan2(-dx_ab, dz_ab))
            self.ayaw[face_a] = target_yaw_a[face_a]

        face_b = b_actions[:, ACT_FACE_TARGET].astype(bool)
        if np.any(face_b):
            target_yaw_b = np.degrees(np.arctan2(dx_ab, -dz_ab))
            self.byaw[face_b] = target_yaw_b[face_b]

        # Decrement i-frames
        self.a_hurt_time = np.maximum(0, self.a_hurt_time - 1)
        self.b_hurt_time = np.maximum(0, self.b_hurt_time - 1)

        # Decrement cooldowns
        self.a_eat_cooldown = np.maximum(0, self.a_eat_cooldown - 1)
        self.b_eat_cooldown = np.maximum(0, self.b_eat_cooldown - 1)
        self.a_pot_lockout = np.maximum(0, self.a_pot_lockout - 1)
        self.b_pot_lockout = np.maximum(0, self.b_pot_lockout - 1)
        self.a_pearl_cooldown = np.maximum(0, self.a_pearl_cooldown - 1)
        self.b_pearl_cooldown = np.maximum(0, self.b_pearl_cooldown - 1)

        # ── Sigil cooldown tick ──
        self.a_brace_cd = np.maximum(0, self.a_brace_cd - 1)
        self.b_brace_cd = np.maximum(0, self.b_brace_cd - 1)
        self.a_cleo_cd = np.maximum(0, self.a_cleo_cd - 1)
        self.b_cleo_cd = np.maximum(0, self.b_cleo_cd - 1)
        self.a_sand_cd = np.maximum(0, self.a_sand_cd - 1)
        self.b_sand_cd = np.maximum(0, self.b_sand_cd - 1)
        self.a_grace_cd = np.maximum(0, self.a_grace_cd - 1)
        self.b_grace_cd = np.maximum(0, self.b_grace_cd - 1)
        self.a_brace_timer = np.maximum(0, self.a_brace_timer - 1)
        self.b_brace_timer = np.maximum(0, self.b_brace_timer - 1)
        self.a_cleo_on_b = np.maximum(0, self.a_cleo_on_b - 1)
        self.b_cleo_on_a = np.maximum(0, self.b_cleo_on_a - 1)
        self.a_sand_timer = np.maximum(0, self.a_sand_timer - 1)
        self.b_sand_timer = np.maximum(0, self.b_sand_timer - 1)
        self.a_grace_timer = np.maximum(0, self.a_grace_timer - 1)
        self.b_grace_timer = np.maximum(0, self.b_grace_timer - 1)

        # ── Sigil activations ──
        self._step_sigils(a_actions, b_actions)

        # ── Sword blocking DISABLED (ArcaneSigils combat.yml: sword-blocking.enabled=false) ──
        # Blocking action is ignored — no damage reduction or KB reduction
        self.a_blocking[:] = False
        self.b_blocking[:] = False

        # ── Consumable mutual exclusion ──
        a_raw_eat = a_actions[:, ACT_EAT_GAP].astype(bool)
        a_raw_pot = a_actions[:, ACT_THROW_POT].astype(bool)
        a_raw_pearl = a_actions[:, ACT_THROW_PEARL].astype(bool)
        b_raw_eat = b_actions[:, ACT_EAT_GAP].astype(bool)
        b_raw_pot = b_actions[:, ACT_THROW_POT].astype(bool)
        b_raw_pearl = b_actions[:, ACT_THROW_PEARL].astype(bool)

        # If eating, suppress pot and pearl
        a_actions[:, ACT_THROW_POT] = (a_raw_pot & ~a_raw_eat).astype(np.int8)
        a_actions[:, ACT_THROW_PEARL] = (a_raw_pearl & ~a_raw_eat & ~a_raw_pot).astype(np.int8)
        b_actions[:, ACT_THROW_POT] = (b_raw_pot & ~b_raw_eat).astype(np.int8)
        b_actions[:, ACT_THROW_PEARL] = (b_raw_pearl & ~b_raw_eat & ~b_raw_pot).astype(np.int8)

        # ── Golden apple eating ──
        # Can't start eating while blocking (already using item)
        a_eat_action = a_raw_eat & ~self.a_blocking
        b_eat_action = b_raw_eat & ~self.b_blocking
        self._step_eating(
            a_eat_action,
            self.a_eating_ticks, self.a_eat_cooldown, self.a_regen_ticks,
            self.a_gapple_count, self.a_health, self.a_absorption
        )
        self._step_eating(
            b_eat_action,
            self.b_eating_ticks, self.b_eat_cooldown, self.b_regen_ticks,
            self.b_gapple_count, self.b_health, self.b_absorption
        )

        # ── Regen tick (from golden apple effect) ──
        self._tick_regen(self.a_regen_ticks, self.a_health)
        self._tick_regen(self.b_regen_ticks, self.b_health)

        # ── Splash pots (instant heal, AoE splash) ──
        # Can't throw pot while eating
        dist_2d_pre = np.sqrt(dx_ab**2 + dz_ab**2)  # dx_ab/dz_ab computed above for facing
        a_pot_action = a_actions[:, ACT_THROW_POT].astype(bool) & (self.a_eating_ticks <= 0)
        b_pot_action = b_actions[:, ACT_THROW_POT].astype(bool) & (self.b_eating_ticks <= 0)
        a_pot_splash = self._step_splash_pot(
            a_pot_action,
            self.a_pot_count, self.a_pot_lockout,
            self.a_health, self.b_health, dist_2d_pre
        )
        b_pot_splash = self._step_splash_pot(
            b_pot_action,
            self.b_pot_count, self.b_pot_lockout,
            self.b_health, self.a_health, dist_2d_pre
        )

        # ── Ender pearls (BACK+pearl = escape, else = aggressive) ──
        # Can't throw pearl while eating or pot lockout
        a_pearl_action = a_actions[:, ACT_THROW_PEARL].astype(bool) & (self.a_eating_ticks <= 0) & (self.a_pot_lockout <= 0)
        b_pearl_action = b_actions[:, ACT_THROW_PEARL].astype(bool) & (self.b_eating_ticks <= 0) & (self.b_pot_lockout <= 0)
        a_pearled = self._step_pearl(
            a_pearl_action,
            a_actions[:, ACT_BACKWARD].astype(bool),
            self.a_pearl_count, self.a_pearl_cooldown,
            self.ax, self.ay, self.az, self.avx, self.avy, self.avz,
            self.bx, self.by, self.bz, self.a_health
        )
        b_pearled = self._step_pearl(
            b_pearl_action,
            b_actions[:, ACT_BACKWARD].astype(bool),
            self.b_pearl_count, self.b_pearl_cooldown,
            self.bx, self.by, self.bz, self.bvx, self.bvy, self.bvz,
            self.ax, self.ay, self.az, self.b_health
        )

        # Movement for both players (pass prev_forward for sprint reset tracking)
        self._step_movement_batch(
            self.ax, self.ay, self.az, self.avx, self.avy, self.avz,
            self.ayaw, self.a_on_ground, self.a_sprinting, self.a_fall_dist,
            a_actions, self.a_prev_forward
        )
        self._step_movement_batch(
            self.bx, self.by, self.bz, self.bvx, self.bvy, self.bvz,
            self.byaw, self.b_on_ground, self.b_sprinting, self.b_fall_dist,
            b_actions, self.b_prev_forward
        )

        # ── ENGAGE auto-attack: face target + attack when in reach ──
        # Pre-combat random blocking prevents same-tick trades (models server packet ordering).
        dx_engage = self.bx - self.ax
        dz_engage = self.bz - self.az
        dy_engage = self.by - self.ay
        dist_engage = np.sqrt(dx_engage**2 + dz_engage**2 + dy_engage**2)
        engage_a = a_actions[:, ACT_ENGAGE].astype(bool)
        engage_b = b_actions[:, ACT_ENGAGE].astype(bool)
        in_range = dist_engage <= self.attack_reach

        # When both would attack on the same tick, randomly block one (50/50).
        # Models server-side packet ordering — one player's hit processes first,
        # KB pushes the other out of range before they can swing.
        both_would_attack = engage_a & engage_b & in_range
        a_wins_priority = self.rng.random(self.n) < 0.5
        a_blocked = both_would_attack & ~a_wins_priority
        b_blocked = both_would_attack & a_wins_priority

        a_actions[:, ACT_ATTACK] = (engage_a & in_range & ~a_blocked).astype(np.int8)
        b_actions[:, ACT_ATTACK] = (engage_b & in_range & ~b_blocked).astype(np.int8)

        # ── Sigil damage multipliers (applied inside _step_combat_batch) ──
        # A→B: A's sand amp, A's cleo on B, B's brace DR, B's grace DR
        a_to_b_mult = np.ones(self.n)
        a_to_b_mult *= np.where(self.a_sand_timer > 0, SIGIL_SAND_DMG_AMP, 1.0)
        a_to_b_mult *= np.where(self.a_cleo_on_b > 0, SIGIL_CLEO_DMG_AMP, 1.0)
        a_to_b_mult *= np.where(self.b_brace_timer > 0, 1.0 - SIGIL_BRACE_DR, 1.0)
        a_to_b_mult *= np.where(self.b_grace_timer > 0, 1.0 - SIGIL_GRACE_DR, 1.0)
        # B→A: B's sand amp, B's cleo on A, A's brace DR, A's grace DR
        b_to_a_mult = np.ones(self.n)
        b_to_a_mult *= np.where(self.b_sand_timer > 0, SIGIL_SAND_DMG_AMP, 1.0)
        b_to_a_mult *= np.where(self.b_cleo_on_a > 0, SIGIL_CLEO_DMG_AMP, 1.0)
        b_to_a_mult *= np.where(self.a_brace_timer > 0, 1.0 - SIGIL_BRACE_DR, 1.0)
        b_to_a_mult *= np.where(self.a_grace_timer > 0, 1.0 - SIGIL_GRACE_DR, 1.0)

        # Combat: A attacks B (blocked while eating or pot throw)
        a_can_attack = a_actions[:, ACT_ATTACK].astype(bool) & (self.a_eating_ticks <= 0) & (self.a_pot_lockout <= 0)
        a_dmg, a_sprint_hits, a_hit_dist = self._step_combat_batch(
            self.ax, self.ay, self.az, self.avx, self.avz,
            self.ayaw, self.a_on_ground, self.a_sprinting, self.a_fall_dist,
            self.bx, self.by, self.bz, self.bvx, self.bvy, self.bvz,
            self.byaw, self.b_on_ground, self.b_sprinting, self.b_blocking,
            self.b_health, self.b_absorption, self.b_hurt_time,
            a_can_attack, dmg_mult=a_to_b_mult,
        )

        # Combat: B attacks A (blocked while eating or pot throw)
        b_can_attack = b_actions[:, ACT_ATTACK].astype(bool) & (self.b_eating_ticks <= 0) & (self.b_pot_lockout <= 0)
        b_dmg, b_sprint_hits, b_hit_dist = self._step_combat_batch(
            self.bx, self.by, self.bz, self.bvx, self.bvz,
            self.byaw, self.b_on_ground, self.b_sprinting, self.b_fall_dist,
            self.ax, self.ay, self.az, self.avx, self.avy, self.avz,
            self.ayaw, self.a_on_ground, self.a_sprinting, self.a_blocking,
            self.a_health, self.a_absorption, self.a_hurt_time,
            b_can_attack, dmg_mult=b_to_a_mult,
        )

        # Brace charges: gain 1 per hit taken
        self.a_brace_charges += (b_dmg > 0).astype(np.int32)
        self.b_brace_charges += (a_dmg > 0).astype(np.int32)
        self.a_brace_charges = np.minimum(self.a_brace_charges, 100)
        self.b_brace_charges = np.minimum(self.b_brace_charges, 100)

        # Kill threshold: health < 1.0 → dead (matches server)
        self.a_health[self.a_health < 1.0] = 0.0
        self.b_health[self.b_health < 1.0] = 0.0

        # Track sprint-hits for combo style override next tick
        self.a_did_sprint_hit[:] = a_sprint_hits
        self.b_did_sprint_hit[:] = b_sprint_hits

        # Track last-dealt-hit tick for combo priority
        self.a_last_dealt_tick[a_sprint_hits] = self.tick[a_sprint_hits]
        self.b_last_dealt_tick[b_sprint_hits] = self.tick[b_sprint_hits]

        # ── Counter-hit detection (before combo update) ──
        # A counter-hit: A was hit recently, A is on ground, A sprint-hits back
        COUNTER_WINDOW = 15  # ticks after being hit to qualify as counter
        a_counter = (a_sprint_hits
                     & self.a_on_ground
                     & ((self.tick - self.a_last_hit_tick) <= COUNTER_WINDOW)
                     & ((self.tick - self.a_last_hit_tick) > 0))
        b_counter = (b_sprint_hits
                     & self.b_on_ground
                     & ((self.tick - self.b_last_hit_tick) <= COUNTER_WINDOW)
                     & ((self.tick - self.b_last_hit_tick) > 0))

        # Update last-hit-tick tracking
        self.a_last_hit_tick[b_dmg > 0] = self.tick[b_dmg > 0]  # A was hit by B
        self.b_last_hit_tick[a_dmg > 0] = self.tick[a_dmg > 0]  # B was hit by A

        # ── Combo streak tracking ──
        # Save pre-update streaks for combo-break detection in rewards
        a_combo_pre = self.a_combo_streak.copy()
        # A lands a sprint-hit → A's streak grows, B's streak resets
        self.a_combo_streak[a_sprint_hits] += 1
        self.b_combo_streak[a_sprint_hits] = 0
        # B lands a sprint-hit → B's streak grows, A's streak resets
        self.b_combo_streak[b_sprint_hits] += 1
        self.a_combo_streak[b_sprint_hits] = 0

        # Rewards — separate for charger (A) and counter (B)
        a_rewards = self._compute_rewards_charger(a_actions, b_actions, a_dmg, b_dmg,
                                                   a_sprint_hits, b_sprint_hits,
                                                   a_pot_splash, b_pot_splash,
                                                   a_pearled, b_pearled,
                                                   a_combo_pre)
        b_combo_pre = self.b_combo_streak.copy()
        if self.reward_mode == "hybrid":
            b_rewards = self._compute_rewards_hybrid(b_actions, a_actions, b_dmg, a_dmg,
                                                      b_sprint_hits, a_sprint_hits,
                                                      b_pot_splash, a_pot_splash,
                                                      b_pearled, a_pearled,
                                                      b_combo_pre, b_counter)
        else:
            b_rewards = self._compute_rewards_counter(b_actions, a_actions, b_dmg, a_dmg,
                                                       b_sprint_hits, a_sprint_hits,
                                                       b_pot_splash, a_pot_splash,
                                                       b_pearled, a_pearled,
                                                       b_counter)
        self.episode_rewards += a_rewards

        # Termination
        a_dead = self.a_health <= 0
        b_dead = self.b_health <= 0

        if self.gauntlet:
            # Gauntlet: opponent dying respawns them, survivor keeps state
            if self.gauntlet_survivor == 'b':
                # B survives, A respawns (training: A=rule, B=model)
                a_killed = np.where(a_dead)[0]
                if len(a_killed) > 0:
                    self.kill_count[a_killed] += 1
                    self._reset_a_only(a_killed)
                terminated = b_dead
            else:
                # A survives, B respawns (evaluate: A=model, B=rule)
                b_killed = np.where(b_dead)[0]
                if len(b_killed) > 0:
                    self.kill_count[b_killed] += 1
                    self._reset_b_only(b_killed)
                terminated = a_dead
        else:
            terminated = a_dead | b_dead

        truncated = self.tick >= self.episode_length
        dones = terminated | truncated

        obs = self._get_obs()

        infos = {
            "a_health": self.a_health.copy(),
            "b_health": self.b_health.copy(),
            "a_dmg_dealt": a_dmg.copy(),
            "b_dmg_dealt": b_dmg.copy(),
            "a_sprint_hits": a_sprint_hits.copy(),
            "b_sprint_hits": b_sprint_hits.copy(),
            "a_hit_dist": a_hit_dist.copy(),
            "b_hit_dist": b_hit_dist.copy(),
            "a_counter_hit": a_counter.copy(),
            "b_counter_hit": b_counter.copy(),
            "a_pot_splash_enemy": a_pot_splash.copy(),
            "b_pot_splash_enemy": b_pot_splash.copy(),
            "a_pearled": a_pearled.copy(),
            "b_pearled": b_pearled.copy(),
            "a_rewards": a_rewards.copy(),
            "b_rewards": b_rewards.copy(),
            "episode_reward": self.episode_rewards.copy(),
            "tick": self.tick.copy(),
            "kill_count": self.kill_count.copy(),
        }

        # Auto-reset done envs
        self.reset_envs(dones)

        return obs, a_rewards, dones, infos

    def _step_sigils(self, a_actions, b_actions):
        """Process sigil ability activations for both players."""
        # King's Brace (bit 14 = ACT_SIGIL_0)
        for act, charges, timer, cd in [
            (a_actions, self.a_brace_charges, self.a_brace_timer, self.a_brace_cd),
            (b_actions, self.b_brace_charges, self.b_brace_timer, self.b_brace_cd),
        ]:
            want = act[:, ACT_SIGIL_0].astype(bool)
            can = want & (cd <= 0) & (charges >= SIGIL_BRACE_CHARGE_REQ)
            timer[can] = SIGIL_BRACE_DURATION
            charges[can] = 0
            cd[can] = SIGIL_BRACE_COOLDOWN

        # Cleopatra (bit 15 = ACT_SIGIL_0+1)
        for act, cd, target_brace, target_grace, cleo_on_target in [
            (a_actions, self.a_cleo_cd, self.b_brace_timer, self.b_grace_timer, self.a_cleo_on_b),
            (b_actions, self.b_cleo_cd, self.a_brace_timer, self.a_grace_timer, self.b_cleo_on_a),
        ]:
            want = act[:, ACT_SIGIL_0 + 1].astype(bool)
            can = want & (cd <= 0)
            if np.any(can):
                target_brace[can] = 0
                target_grace[can] = 0
                resisted = self.rng.random(self.n) < 0.5
                cleo_on_target[can & ~resisted] = SIGIL_CLEO_DURATION
                cd[can] = SIGIL_CLEO_COOLDOWN

        # Quick Sand (bit 16 = ACT_SIGIL_0+2)
        for act, timer, cd in [
            (a_actions, self.a_sand_timer, self.a_sand_cd),
            (b_actions, self.b_sand_timer, self.b_sand_cd),
        ]:
            want = act[:, ACT_SIGIL_0 + 2].astype(bool)
            can = want & (cd <= 0)
            timer[can] = SIGIL_SAND_DURATION
            cd[can] = SIGIL_SAND_COOLDOWN

        # Nile's Grace (bit 17 = ACT_SIGIL_0+3)
        for act, timer, cd in [
            (a_actions, self.a_grace_timer, self.a_grace_cd),
            (b_actions, self.b_grace_timer, self.b_grace_cd),
        ]:
            want = act[:, ACT_SIGIL_0 + 3].astype(bool)
            can = want & (cd <= 0)
            timer[can] = SIGIL_GRACE_DURATION
            cd[can] = SIGIL_GRACE_COOLDOWN

    def _step_eating(self, eat_action, eating_ticks, eat_cooldown, regen_ticks,
                     gapple_count, health, absorption):
        """Handle golden apple eating mechanics."""
        # Start eating: action active, not already eating, cooldown done, have gapples
        can_start = eat_action & (eating_ticks <= 0) & (eat_cooldown <= 0) & (gapple_count > 0)
        eating_ticks[can_start] = GAP_EAT_TICKS

        # Tick down eating
        is_eating = eating_ticks > 0
        eating_ticks[is_eating] -= 1

        # Finish eating: ticks just hit 0
        just_finished = is_eating & (eating_ticks <= 0)
        if np.any(just_finished):
            # Apply golden apple effects
            absorption[just_finished] = np.minimum(
                absorption[just_finished] + GAP_ABSORPTION, 20.0
            )
            regen_ticks[just_finished] = GAP_REGEN_TICKS
            eat_cooldown[just_finished] = GAP_COOLDOWN_TICKS
            gapple_count[just_finished] -= 1

    def _tick_regen(self, regen_ticks, health):
        """Apply regen II effect: 1 HP every 25 ticks."""
        active = regen_ticks > 0
        regen_ticks[active] -= 1

        # Heal on regen ticks (every GAP_REGEN_RATE ticks)
        heal_tick = active & (regen_ticks % GAP_REGEN_RATE == 0) & (regen_ticks > 0)
        health[heal_tick] = np.minimum(health[heal_tick] + 1.0, 20.0)

    def _step_splash_pot(self, throw_action, pot_count, pot_lockout,
                          self_health, enemy_health, dist_2d):
        """Handle splash healing potion mechanics.

        Instant Health II: heals 8 HP instantly.
        AoE splash: if enemy is within POT_SPLASH_RADIUS, they get healed too
        (reduced by distance: heal * (1 - dist/radius)).

        Returns: bool array of whether enemy got splashed (for reward penalty).
        """
        can_throw = throw_action & (pot_count > 0) & (pot_lockout <= 0)
        enemy_splashed = np.zeros(len(self_health), dtype=bool)

        if np.any(can_throw):
            # Heal self (full amount, capped at 20)
            self_health[can_throw] = np.minimum(self_health[can_throw] + POT_HEAL_AMOUNT, 20.0)

            # Splash enemy if they're within radius
            in_splash = can_throw & (dist_2d < POT_SPLASH_RADIUS)
            if np.any(in_splash):
                # Heal reduces with distance: full heal at feet, zero at edge
                heal_frac = 1.0 - (dist_2d[in_splash] / POT_SPLASH_RADIUS)
                enemy_health[in_splash] = np.minimum(
                    enemy_health[in_splash] + POT_HEAL_AMOUNT * heal_frac, 20.0
                )
                enemy_splashed[in_splash] = True

            # Consume pot, apply throw lockout
            pot_count[can_throw] -= 1
            pot_lockout[can_throw] = POT_THROW_LOCKOUT

        return enemy_splashed

    def _step_pearl(self, throw_action, backward_action, pearl_count, pearl_cooldown,
                     self_x, self_y, self_z, self_vx, self_vy, self_vz,
                     enemy_x, enemy_y, enemy_z, self_health):
        """Handle ender pearl mechanics.

        Direction depends on movement:
        - Pressing BACK + pearl = escape pearl (teleport 10 blocks AWAY from enemy)
        - Otherwise = aggressive pearl (teleport 2 blocks past enemy)
        Self-damage on landing (5 HP through armor).
        20-tick cooldown.

        Returns: bool array of envs where pearl was used.
        """
        can_throw = throw_action & (pearl_count > 0) & (pearl_cooldown <= 0)
        pearled = np.zeros(len(self_health), dtype=bool)

        if np.any(can_throw):
            dx = enemy_x[can_throw] - self_x[can_throw]
            dz = enemy_z[can_throw] - self_z[can_throw]
            dist = np.sqrt(dx**2 + dz**2)
            safe_dist = np.maximum(dist, 0.1)
            unit_x = dx / safe_dist
            unit_z = dz / safe_dist

            # Direction: BACK = escape (away), else = aggressive (toward/past)
            escaping = backward_action[can_throw]
            # Aggressive: 2 blocks past enemy. Escape: 10 blocks away from enemy.
            land_x = np.where(escaping,
                self_x[can_throw] - unit_x * 10.0,  # away
                enemy_x[can_throw] + unit_x * 2.0    # past enemy
            )
            land_z = np.where(escaping,
                self_z[can_throw] - unit_z * 10.0,
                enemy_z[can_throw] + unit_z * 2.0
            )
            self_x[can_throw] = land_x
            self_z[can_throw] = land_z
            self_y[can_throw] = enemy_y[can_throw]

            # Kill velocity on landing
            self_vx[can_throw] = 0.0
            self_vy[can_throw] = 0.0
            self_vz[can_throw] = 0.0

            # Self-damage (5 HP, bypasses armor)
            self_health[can_throw] -= PEARL_SELF_DAMAGE
            self_health[:] = np.maximum(0.0, self_health)

            # Consume pearl, apply cooldown
            pearl_count[can_throw] -= 1
            pearl_cooldown[can_throw] = PEARL_COOLDOWN_TICKS
            pearled[can_throw] = True

        return pearled

    def _step_movement_batch(self, x, y, z, vx, vy, vz, yaw,
                              on_ground, sprinting, fall_dist, actions,
                              prev_forward):
        """Vectorized movement physics for N players.

        Sprint state mechanic (1.8 combo core):
        - Sprint is cancelled when you HIT someone (done in combat step)
        - To re-sprint: must release W then press W again (W-tap)
          or tap S then W (S-tap) — both create a rising edge on forward
        - With toggle sprint: sprint re-engages on forward rising edge
        """
        n = len(x)
        fwd = actions[:, ACT_FORWARD].astype(np.float32) - actions[:, ACT_BACKWARD].astype(np.float32)
        strafe = actions[:, ACT_RIGHT].astype(np.float32) - actions[:, ACT_LEFT].astype(np.float32)
        jump = actions[:, ACT_JUMP].astype(bool)
        sprint_input = actions[:, ACT_SPRINT].astype(bool)
        sneak = actions[:, ACT_SNEAK].astype(bool)

        was_on_ground = on_ground.copy()
        forward_now = fwd > 0

        # Sprint state: toggle sprint model (auto-sprint)
        # Sprint STARTS on rising edge of forward (was not pressing W, now pressing W)
        sprint_trigger = forward_now & ~prev_forward  # W-tap or S-tap detected
        # Sprint CONTINUES if already sprinting and still pressing forward
        sprint_continue = sprinting & forward_now & ~sneak
        # New sprint state (sprint cancel happens in combat step)
        sprinting[:] = sprint_trigger | sprint_continue

        # Update prev_forward for next tick's W-tap detection
        prev_forward[:] = forward_now

        # Jump
        can_jump = jump & was_on_ground
        vy[can_jump] = JUMP_IMPULSE
        on_ground[can_jump] = False

        # Sprint jump kick
        sprint_jump = can_jump & sprinting
        yaw_rad = np.radians(yaw)
        vx[sprint_jump] -= np.sin(yaw_rad[sprint_jump]) * SPRINT_JUMP_KICK
        vz[sprint_jump] += np.cos(yaw_rad[sprint_jump]) * SPRINT_JUMP_KICK

        # Input acceleration
        move_speed = np.full(n, PLAYER_WALK_SPEED)
        move_speed[sprinting] *= SPRINT_SPEED_MULT
        move_speed[sneak & ~sprinting] *= SNEAK_SPEED_MULT

        friction = GROUND_FRICTION
        slip = np.where(was_on_ground, friction * DRAG_FACTOR, DRAG_FACTOR)
        accel = np.where(was_on_ground, move_speed * (0.6 / slip) ** 3, 0.02)

        # Direction from yaw
        yaw_rad = np.radians(yaw)
        sin_yaw = np.sin(yaw_rad)
        cos_yaw = np.cos(yaw_rad)

        # Normalize diagonal
        dist_sq = fwd * fwd + strafe * strafe
        too_long = dist_sq > 1.0
        inv_len = np.where(too_long, 1.0 / np.sqrt(np.maximum(dist_sq, 1e-8)), 1.0)
        fwd = fwd * inv_len
        strafe = strafe * inv_len

        input_x = -sin_yaw * fwd - cos_yaw * strafe
        input_z = cos_yaw * fwd - sin_yaw * strafe

        vx += input_x * accel
        vz += input_z * accel

        # Position update
        x += vx
        y += vy
        z += vz

        # Ground collision
        below_floor = y <= self.floor_y
        y[below_floor] = self.floor_y
        falling = below_floor & (vy < 0)
        vy[falling] = 0.0
        on_ground[below_floor] = True
        fall_dist[below_floor] = 0.0
        on_ground[~below_floor] = False
        rising_or_flat = ~below_floor & (vy < 0)
        fall_dist[rising_or_flat] -= vy[rising_or_flat]

        # Arena walls
        half = self.arena_half
        wall_x = np.abs(x) > half
        x[wall_x] = np.clip(x[wall_x], -half, half)
        vx[wall_x] = 0.0
        wall_z = np.abs(z) > half
        z[wall_z] = np.clip(z[wall_z], -half, half)
        vz[wall_z] = 0.0

        # Drag + gravity
        vx *= slip
        vz *= slip
        vy[:] = (vy - GRAVITY) * AIR_DRAG

    def _step_combat_batch(self, ax, ay, az, avx, avz, ayaw, a_on_ground, a_sprint, a_fall,
                            tx, ty, tz, tvx, tvy, tvz, tyaw, t_on_ground, t_sprint, t_blocking,
                            t_health, t_absorption, t_hurt_time, attack_mask, dmg_mult=None):
        """Vectorized combat: attacker → target. Returns (damage, sprint_hits)."""
        n = len(ax)
        damage = np.zeros(n)
        sprint_hit = np.zeros(n, dtype=bool)

        # Range check — MC reach is from attacker EYE to target HITBOX
        # Player eye height = 1.62, hitbox = 0.6 wide x 1.8 tall from feet
        EYE_HEIGHT = 1.62
        HITBOX_HEIGHT = 1.8
        dx = tx - ax
        dz = tz - az
        horiz_dist = np.sqrt(dx**2 + dz**2)

        # Vertical: attacker eye to nearest point on target hitbox
        a_eye_y = ay + EYE_HEIGHT
        t_box_bottom = ty
        t_box_top = ty + HITBOX_HEIGHT
        # Clamp attacker eye Y to target hitbox range → distance = 0 if inside
        closest_y = np.clip(a_eye_y, t_box_bottom, t_box_top)
        dy_reach = a_eye_y - closest_y
        dist = np.sqrt(dx**2 + dy_reach**2 + dz**2)
        in_range = dist <= ATTACK_REACH

        # Angle check: 3D facing — include pitch toward target center
        yaw_rad = np.radians(ayaw)
        t_center_y = ty + HITBOX_HEIGHT * 0.5  # aim at hitbox center
        dy_aim = t_center_y - a_eye_y
        aim_dist = np.sqrt(dx**2 + dy_aim**2 + dz**2)
        safe_aim = np.maximum(aim_dist, 0.001)
        # Direction from attacker eye to target center
        dir_x = dx / safe_aim
        dir_y = dy_aim / safe_aim
        dir_z = dz / safe_aim
        # Attacker look vector: yaw + pitch toward target
        pitch = np.arctan2(-dy_aim, np.maximum(horiz_dist, 0.001))
        face_x = -np.sin(yaw_rad) * np.cos(pitch)
        face_y = -np.sin(pitch)  # MC convention: positive pitch = looking down
        face_z = np.cos(yaw_rad) * np.cos(pitch)
        # 3D dot product
        dot = face_x * dir_x + face_y * dir_y + face_z * dir_z
        facing_target = dot >= MELEE_ANGLE_COS

        # I-frame check: server blocks ALL hits during i-frames
        can_hit = t_hurt_time <= 0

        # Active mask: attacking AND in range AND facing AND can hit
        active = attack_mask & in_range & facing_target & can_hit
        if not np.any(active):
            return damage, sprint_hit, dist

        # Critical hit: falling, not on ground
        crit = (a_fall > 0) & ~a_on_ground

        # Damage — Kitara sword blocking: (1 + raw_damage) * 0.5, then armor
        blocked = active & t_blocking
        unblocked = active & ~t_blocking
        dmg = np.zeros(n)
        dmg[unblocked & crit] = self.crit_dmg_after_armor
        dmg[unblocked & ~crit] = self.base_dmg_after_armor
        dmg[blocked & crit] = self.crit_dmg_blocked
        dmg[blocked & ~crit] = self.base_dmg_blocked
        damage[:] = dmg

        if dmg_mult is not None:
            dmg *= dmg_mult
            damage[:] = dmg

        # Apply damage: absorption first, then health
        absorbed = np.minimum(t_absorption, dmg)
        t_absorption -= absorbed * active
        remaining = dmg - absorbed
        t_health -= remaining * active
        t_health[:] = np.maximum(0.0, t_health)
        t_hurt_time[active] = I_FRAME_TICKS

        # ── Kitara Advanced Knockback ──
        # Direction from attacker YAW (not toward target!)
        a_yaw_rad = np.radians(ayaw)
        dir_x = -np.sin(a_yaw_rad)
        dir_z = np.cos(a_yaw_rad)

        # Horizontal KB with velocity inheritance
        # motX = victimVx * friction + dir_x, motZ = victimVz * friction + dir_z
        if KB_HORIZONTAL_INHERIT:
            mot_x = tvx * KB_HORIZONTAL_FRICTION + dir_x
            mot_z = tvz * KB_HORIZONTAL_FRICTION + dir_z
        else:
            mot_x = dir_x.copy()
            mot_z = dir_z.copy()

        # Scale by horizontal base
        mot_x *= KB_HORIZONTAL  # 0.38
        mot_z *= KB_HORIZONTAL

        # Vertical KB
        mot_y = np.full(n, KB_VERTICAL)  # 0.35

        # Ground multiplier
        grounded = t_on_ground
        mot_x[active & grounded] *= KB_HORIZONTAL_ON_GROUND  # 1.1
        mot_z[active & grounded] *= KB_HORIZONTAL_ON_GROUND
        mot_y[active & grounded] *= KB_VERTICAL_ON_GROUND  # 1.0

        # Sprint multiplier (THE combo mechanic — sprint hits = more KB)
        sprint_hit = active & a_sprint
        mot_x[sprint_hit] *= KB_HORIZONTAL_SPRINTING  # 1.35
        mot_z[sprint_hit] *= KB_HORIZONTAL_SPRINTING
        mot_y[sprint_hit] *= KB_VERTICAL_SPRINTING  # 1.0

        # Sword blocking KB reduction
        mot_x[blocked] *= KB_SWORD_BLOCK_HORIZONTAL  # 0.85
        mot_z[blocked] *= KB_SWORD_BLOCK_HORIZONTAL
        mot_y[blocked] *= KB_SWORD_BLOCK_VERTICAL  # 0.95

        # SET target velocity (Kitara replaces, doesn't add)
        tvx[active] = mot_x[active]
        tvy[active] = mot_y[active]
        tvz[active] = mot_z[active]
        t_on_ground[active] = False

        # Attacker slowdown + sprint cancel (THE combo reset trigger)
        avx[active] *= KB_SLOWDOWN  # 0.6
        avz[active] *= KB_SLOWDOWN
        if KB_CANCEL_SPRINT:
            a_sprint[active] = False  # Sprint cancelled! Must W-tap or S-tap to re-sprint

        return damage, sprint_hit, dist

    def _compute_rewards_charger(self, a_actions, b_actions, a_dmg_dealt, b_dmg_dealt,
                                  a_sprint_hits, b_sprint_hits,
                                  a_pot_splash_enemy, b_pot_splash_enemy,
                                  a_pearled, b_pearled,
                                  a_combo_pre=None):
        """Charger reward: aggressive, always closing, always swinging.
        Simple reward that produces a bot that brainlessly charges and attacks."""
        rewards = np.zeros(self.n)

        # ── KILL / DEATH (terminal signal) ──
        rewards[self.b_health <= 0] += 20.0
        rewards[self.a_health <= 0] -= 20.0

        # ── SURVIVAL BONUS on kill — reward staying healthy ──
        killed_enemy = self.b_health <= 0
        hp_bonus = np.where(self.a_health > 14.0, 5.0,
                   np.where(self.a_health > 8.0, 2.5, 0.0))
        rewards += killed_enemy * hp_bonus

        # ── COMBO CHAINS (the core mechanic — escalating reward) ──
        # Each consecutive sprint-hit in a combo is worth more than the last.
        # streak=1: first hit (3.0), streak=2: second (5.0), streak=3: (7.0), etc.
        # Combos are THE way you kill — reward them massively.
        hit_this_tick = a_sprint_hits
        combo_bonus = np.where(hit_this_tick, 3.0 + 2.0 * self.a_combo_streak.astype(np.float64), 0.0)
        rewards += combo_bonus

        # ── CLEAN HIT vs TRADE ──
        # Clean hit: you sprint-hit them, they did NOT hit you = timing was good
        # Trade: both hit same tick = should have staggered timing
        clean_hit = a_sprint_hits & ~b_sprint_hits
        traded_hit = a_sprint_hits & b_sprint_hits
        rewards[clean_hit] += 2.0
        rewards[traded_hit] -= 1.5

        # ── COMBO BROKEN — opponent hit you while you had a combo going ──
        # You had a streak >= 2 and they landed a hit, resetting it.
        # Teaches the bot to maintain spacing so the opponent can't counter-hit.
        if a_combo_pre is not None:
            combo_broken = (a_combo_pre >= 2) & b_sprint_hits
            # Scale penalty by how long the combo was — losing a 5-streak hurts more
            rewards[combo_broken] -= (2.0 + a_combo_pre[combo_broken].astype(np.float64))

        # Non-sprint hit: you landed damage but didn't sprint-hit.
        # Sprint-hitting is essential for KB and combos — penalize lazy hits.
        non_sprint_hit = (a_dmg_dealt > 0) & ~a_sprint_hits
        rewards[non_sprint_hit] -= 1.0

        # ── TRADE WINNING ──
        # Both players hit each other this tick (or within i-frame window).
        # Reward is based on NET damage advantage in the trade.
        # Positive = you dealt more damage, negative = you took more.
        net_dmg = a_dmg_dealt - b_dmg_dealt
        rewards += net_dmg * 0.5

        # ── RAW DAMAGE ──
        # Baseline reward for any damage dealt
        rewards += a_dmg_dealt * 0.3

        # Time pressure
        rewards -= 0.02

        # ── POSITIONING ──
        dx = self.bx - self.ax
        dz = self.bz - self.az
        dy = self.by - self.ay
        horiz_dist = np.sqrt(dx**2 + dz**2)
        dist = np.sqrt(dx**2 + dz**2 + dy**2)

        in_range = dist <= ATTACK_REACH

        # (Whiff and i-frame penalties removed — attack is auto-derived from ENGAGE)

        # ── SPACING — reward sprint-hits at max reach (out of their range) ──
        max_range_hit = a_sprint_hits & (dist >= 2.8) & (dist <= ATTACK_REACH)
        rewards[max_range_hit] += 2.0


        # Distance shaping: reward closing distance
        dist_2d = horiz_dist
        dist_delta = self._prev_dist - dist_2d  # positive = closing, negative = retreating
        retreating = dist_delta < 0
        getting_comboed = self.b_combo_streak >= 2  # enemy has a combo on YOU
        # Retreating is allowed if you're getting comboed — need to escape and heal
        # If you're not getting comboed, you should be fighting, not running
        shaped = np.where(
            retreating & getting_comboed,  # getting comboed = retreat to heal
            0.0,
            np.where(
                retreating,  # retreating when not getting comboed = punished
                dist_delta * 0.5,  # strong penalty for cowardly retreat
                dist_delta * 0.3  # reward for closing (stronger — get in their face)
            )
        )
        rewards += shaped

        # Engagement: reward for being in combat range
        rewards[in_range] += 0.1

        # Passive play penalty: if HP is high and you're far, you're wasting time
        healthy = self.a_health > 12.0
        far_away = dist_2d > 5.0
        rewards[healthy & far_away] -= 0.1

        # ── JUMP SPAM PENALTY ──
        # Jumping while already airborne is useless — penalize holding spacebar
        jumping = a_actions[:, ACT_JUMP].astype(bool)
        jump_while_air = jumping & ~self.a_on_ground
        rewards[jump_while_air] -= 0.1

        # ── GOLDEN APPLE ──
        eat_action = a_actions[:, ACT_EAT_GAP].astype(bool)
        started_eating = eat_action & (self.a_eat_cooldown <= 0) & (self.a_eating_ticks <= 0) & (self.a_gapple_count > 0)
        is_eating = self.a_eating_ticks > 0
        eat_on_cd = eat_action & (self.a_eat_cooldown > 0)
        rewards[eat_on_cd] -= 0.5
        eat_while_eating = eat_action & is_eating
        rewards[eat_while_eating] -= 0.3
        no_gaps = eat_action & (self.a_gapple_count <= 0)
        rewards[no_gaps] -= 0.5
        # Only eat when LOW HP (< 8) — not medium, not high
        good_eat = started_eating & (self.a_health < 16.0)
        rewards[good_eat] += 10.0
        wasteful_eat = started_eating & (self.a_health > 18.0)
        rewards[wasteful_eat] -= 5.0
        # Must be retreating while eating — penalize eating while closing/in range
        retreating_while_eating = is_eating & (dist_delta < -0.05)
        rewards[retreating_while_eating] += 0.5
        approaching_while_eating = is_eating & (dist_delta > 0.05) & (dist_2d < 5.0)
        rewards[approaching_while_eating] -= 1.0

        # ── SPLASH POTS ──
        pot_action = a_actions[:, ACT_THROW_POT].astype(bool)
        rewards[a_pot_splash_enemy] -= 8.0
        pot_while_busy = pot_action & ((self.a_eating_ticks > 0) | (self.a_pot_lockout > 0))
        rewards[pot_while_busy] -= 0.3
        wasteful_pot = pot_action & (self.a_pot_count > 0) & (self.a_pot_lockout <= 0) & (self.a_health > 12.0)
        rewards[wasteful_pot] -= 5.0
        no_pots = pot_action & (self.a_pot_count <= 0)
        rewards[no_pots] -= 0.5
        # Only pot when critically low
        critical_pot = pot_action & (self.a_pot_count > 0) & (self.a_pot_lockout <= 0) & (self.a_health < 8.0)
        rewards[critical_pot] += 10.0

        # ── ENDER PEARLS (strategic repositioning) ──
        pearl_action = a_actions[:, ACT_THROW_PEARL].astype(bool)
        # Pressing pearl while eating or on lockout (spam penalty)
        pearl_while_busy = pearl_action & ((self.a_eating_ticks > 0) | (self.a_pot_lockout > 0))
        rewards[pearl_while_busy] -= 0.3
        # Reward: pearl to close distance when far away
        pearl_far = a_pearled & (dist_2d > 8.0)
        rewards[pearl_far] += 1.5
        # Penalty: pearl when already in range (wastes 5 HP for nothing)
        pearl_close = a_pearled & (dist_2d < ATTACK_REACH)
        rewards[pearl_close] -= 3.0
        # Penalty: pearl on cooldown or no pearls (wasted action)
        no_pearls = pearl_action & ((self.a_pearl_count <= 0) | (self.a_pearl_cooldown > 0))
        rewards[no_pearls] -= 0.1

        # Health advantage change
        health_diff = (self.a_health - self.b_health) / 20.0
        rewards += 0.05 * (health_diff - self._prev_health_diff)

        self._prev_dist[:] = dist_2d
        self._prev_health_diff[:] = health_diff

        return np.clip(rewards, -25.0, 25.0).astype(np.float32)

    def _compute_rewards_counter(self, a_actions, b_actions, a_dmg_dealt, b_dmg_dealt,
                                  a_sprint_hits, b_sprint_hits,
                                  a_pot_splash_enemy, b_pot_splash_enemy,
                                  a_pearled, b_pearled,
                                  a_counter_hits):
        """Counter-hitter reward: get hit first → land first → combo.

        NOTE: 'a' here is the counter-hitter (player B in the sim).
        All state references use b_ arrays since this is called with B's perspective.
        """
        rewards = np.zeros(self.n)

        # ── KILL / DEATH ──
        # Use b_ state since this function is called from B's perspective
        rewards[self.a_health <= 0] += 20.0
        rewards[self.b_health <= 0] -= 20.0

        # ── SURVIVAL BONUS on kill — reward staying healthy ──
        killed_enemy = self.a_health <= 0
        hp_bonus = np.where(self.b_health > 14.0, 5.0,
                   np.where(self.b_health > 8.0, 2.5, 0.0))
        rewards += killed_enemy * hp_bonus

        # ── COUNTER-HIT BONUS (the core mechanic) ──
        # You got hit, landed first, and sprint-hit them back = huge reward
        rewards[a_counter_hits] += 8.0

        # ── COMBO CHAINS (same as charger but even more rewarding) ──
        hit_this_tick = a_sprint_hits
        combo_bonus = np.where(hit_this_tick, 3.0 + 2.0 * self.b_combo_streak.astype(np.float64), 0.0)
        rewards += combo_bonus

        # ── CLEAN HIT vs TRADE ──
        # Counter-hitter should HATE trades even more — the whole point is to NOT trade
        clean_hit = a_sprint_hits & ~b_sprint_hits
        traded_hit = a_sprint_hits & b_sprint_hits
        rewards[clean_hit] += 3.0
        rewards[traded_hit] -= 3.0  # stronger trade penalty

        # ── RAW DAMAGE ──
        rewards += a_dmg_dealt * 0.3

        # Non-sprint hit penalty
        non_sprint_hit = (a_dmg_dealt > 0) & ~a_sprint_hits
        rewards[non_sprint_hit] -= 1.0

        # Time pressure
        rewards -= 0.02

        # ── POSITIONING ──
        dx = self.ax - self.bx
        dz = self.az - self.bz
        dy = self.ay - self.by
        horiz_dist = np.sqrt(dx**2 + dz**2)
        dist = np.sqrt(dx**2 + dz**2 + dy**2)

        in_range = dist <= ATTACK_REACH
        engaging = a_actions[:, ACT_ENGAGE].astype(bool)

        # ── PATIENCE: penalize engaging first, but only before combo starts ──
        # If counter engages in range, hasn't been hit in 5 ticks, AND has no active combo
        # → swinging first → penalty. Once combo streak > 0, free to keep swinging.
        recently_hit = (self.tick - self.b_last_hit_tick) <= 5
        no_combo = self.b_combo_streak == 0
        aggressor_swing = engaging & in_range & ~recently_hit & no_combo
        rewards[aggressor_swing] -= 2.0

        # (Whiff and i-frame penalties removed — attack is auto-derived from ENGAGE)

        # Max range hit bonus
        max_range_hit = a_sprint_hits & (dist >= 2.8) & (dist <= ATTACK_REACH)
        rewards[max_range_hit] += 2.0

        # Distance shaping — counter-hitter holds at medium range, lets charger close
        dist_2d = horiz_dist
        dist_delta = self._prev_dist - dist_2d
        # Far (>4 blocks): close distance normally
        far = dist_2d > 4.0
        rewards[far] += dist_delta[far] * 0.15
        # Medium range (3-4 blocks): reward MAINTAINING distance (hold ground)
        medium = (dist_2d >= 3.0) & (dist_2d <= 4.0)
        rewards[medium] += 0.1  # reward for being in the sweet spot
        rewards[medium & (dist_delta > 0)] -= 0.15  # penalize closing further
        # Close (<3 blocks): no closing reward — only get here after taking a hit

        # ── JUMP SPAM PENALTY ──
        jumping = a_actions[:, ACT_JUMP].astype(bool)
        jump_while_air = jumping & ~self.b_on_ground
        rewards[jump_while_air] -= 0.1

        # ── CONSUMABLES ──
        eat_action = a_actions[:, ACT_EAT_GAP].astype(bool)
        started_eating = eat_action & (self.b_eat_cooldown <= 0) & (self.b_eating_ticks <= 0) & (self.b_gapple_count > 0)
        is_eating = self.b_eating_ticks > 0
        eat_on_cd = eat_action & (self.b_eat_cooldown > 0)
        rewards[eat_on_cd] -= 0.5
        eat_while_eating = eat_action & is_eating
        rewards[eat_while_eating] -= 0.3
        no_gaps = eat_action & (self.b_gapple_count <= 0)
        rewards[no_gaps] -= 0.5
        # Only eat when LOW HP (< 8)
        good_eat = started_eating & (self.b_health < 16.0)
        rewards[good_eat] += 10.0
        wasteful_eat = started_eating & (self.b_health > 18.0)
        rewards[wasteful_eat] -= 5.0
        # Must be retreating while eating
        retreating_while_eating = is_eating & (dist_delta < -0.05)
        rewards[retreating_while_eating] += 0.5
        approaching_while_eating = is_eating & (dist_delta > 0.05) & (dist_2d < 5.0)
        rewards[approaching_while_eating] -= 1.0

        pot_action = a_actions[:, ACT_THROW_POT].astype(bool)
        rewards[a_pot_splash_enemy] -= 8.0
        pot_while_busy = pot_action & ((self.b_eating_ticks > 0) | (self.b_pot_lockout > 0))
        rewards[pot_while_busy] -= 0.3
        wasteful_pot = pot_action & (self.b_pot_count > 0) & (self.b_pot_lockout <= 0) & (self.b_health > 12.0)
        rewards[wasteful_pot] -= 5.0
        no_pots = pot_action & (self.b_pot_count <= 0)
        rewards[no_pots] -= 0.5
        # Only pot when critically low
        critical_pot = pot_action & (self.b_pot_count > 0) & (self.b_pot_lockout <= 0) & (self.b_health < 8.0)
        rewards[critical_pot] += 10.0

        pearl_action = a_actions[:, ACT_THROW_PEARL].astype(bool)
        pearl_while_busy = pearl_action & ((self.b_eating_ticks > 0) | (self.b_pot_lockout > 0))
        rewards[pearl_while_busy] -= 0.3
        pearl_far = a_pearled & (dist_2d > 8.0)
        rewards[pearl_far] += 1.5
        pearl_close = a_pearled & (dist_2d < ATTACK_REACH)
        rewards[pearl_close] -= 3.0
        no_pearls = pearl_action & ((self.b_pearl_count <= 0) | (self.b_pearl_cooldown > 0))
        rewards[no_pearls] -= 0.1

        return np.clip(rewards, -25.0, 25.0).astype(np.float32)

    def _compute_rewards_hybrid(self, a_actions, b_actions, a_dmg_dealt, b_dmg_dealt,
                                 a_sprint_hits, b_sprint_hits,
                                 a_pot_splash_enemy, b_pot_splash_enemy,
                                 a_pearled, b_pearled,
                                 a_combo_pre, a_counter_hits):
        """Hybrid reward: charger aggression + counter-hit bonus.

        Base is _compute_rewards_charger (aggressive, closing, combo-chaining)
        with added counter-hit and first-strike bonuses. Produces a bot that
        fights aggressively AND capitalizes when hit first.

        NOTE: 'a' here is the hybrid fighter (player B in the sim).
        State references use b_ arrays since called from B's perspective.

        Tracks per-component reward totals in self._reward_breakdown for diagnostics.
        """
        # Per-component accumulators (sum across all envs this tick)
        rb = self._reward_breakdown

        rewards = np.zeros(self.n)

        # ── KILL / DEATH ──
        r = np.zeros(self.n)
        r[self.a_health <= 0] += 20.0
        r[self.b_health <= 0] -= 20.0
        rb["kill_death"] += r.sum()
        rewards += r

        # ── COUNTER-HIT BONUS ──
        r = np.zeros(self.n)
        r[a_counter_hits] += 5.0
        rb["counter_hit"] += r.sum()
        rewards += r

        # ── COMBO CHAINS ──
        hit_this_tick = a_sprint_hits
        r = np.where(hit_this_tick, 3.0 + 2.0 * self.b_combo_streak.astype(np.float64), 0.0)
        rb["combo"] += r.sum()
        rewards += r

        # ── CLEAN HIT vs TRADE ──
        r = np.zeros(self.n)
        clean_hit = a_sprint_hits & ~b_sprint_hits
        traded_hit = a_sprint_hits & b_sprint_hits
        r[clean_hit] += 2.0
        r[traded_hit] -= 1.5
        rb["clean_trade"] += r.sum()
        rewards += r

        # ── COMBO BROKEN ──
        r = np.zeros(self.n)
        combo_broken = (a_combo_pre >= 2) & b_sprint_hits
        r[combo_broken] -= (2.0 + a_combo_pre[combo_broken].astype(np.float64))
        rb["combo_broken"] += r.sum()
        rewards += r

        # Non-sprint hit penalty
        r = np.zeros(self.n)
        non_sprint_hit = (a_dmg_dealt > 0) & ~a_sprint_hits
        r[non_sprint_hit] -= 1.0
        rb["non_sprint_hit"] += r.sum()
        rewards += r

        # ── TRADE WINNING ──
        r = (a_dmg_dealt - b_dmg_dealt) * 0.5
        rb["trade_win"] += r.sum()
        rewards += r

        # ── RAW DAMAGE ──
        r = a_dmg_dealt * 0.3
        rb["raw_dmg"] += r.sum()
        rewards += r

        # Time pressure
        r_val = -0.02 * self.n
        rb["time_pressure"] += r_val
        rewards -= 0.02

        # ── POSITIONING ──
        dx = self.ax - self.bx
        dz = self.az - self.bz
        dy = self.ay - self.by
        horiz_dist = np.sqrt(dx**2 + dz**2)
        dist = np.sqrt(dx**2 + dz**2 + dy**2)

        in_range = dist <= ATTACK_REACH

        # (Whiff and i-frame penalties removed — attack is auto-derived from ENGAGE)

        # Max range hit bonus
        r = np.zeros(self.n)
        max_range_hit = a_sprint_hits & (dist >= 2.5) & (dist <= ATTACK_REACH)
        r[max_range_hit] += 3.0
        rb["max_range_hit"] += r.sum()
        rewards += r

        # Distance shaping
        dist_2d = horiz_dist
        dist_delta = self._prev_dist - dist_2d
        getting_comboed = self.a_combo_streak >= 2

        r = np.zeros(self.n)
        far = dist_2d > 4.0
        r[far] += dist_delta[far] * 0.3
        sweet = (dist_2d >= 2.5) & (dist_2d <= 3.5)
        r[sweet] += 0.15
        too_close = dist_2d < 2.0
        r[too_close] -= 0.2
        retreating = dist_delta < 0
        mask = retreating & ~getting_comboed & ~far
        r[mask] += dist_delta[mask] * 0.2
        healthy = self.b_health > 12.0
        far_away = dist_2d > 6.0
        r[healthy & far_away] -= 0.15
        rb["spacing"] += r.sum()
        rewards += r

        # Jump spam penalty
        r = np.zeros(self.n)
        jumping = a_actions[:, ACT_JUMP].astype(bool)
        in_air_jump = jumping & ~self.b_on_ground
        r[in_air_jump] -= 0.1
        rb["jump_spam"] += r.sum()
        rewards += r

        # ── CONSUMABLES ──
        r = np.zeros(self.n)

        # Golden apple
        eat_action = a_actions[:, ACT_EAT_GAP].astype(bool)
        started_eating = eat_action & (self.b_eat_cooldown <= 0) & (self.b_eating_ticks <= 0) & (self.b_gapple_count > 0)
        is_eating = self.b_eating_ticks > 0
        r[eat_action & (self.b_eat_cooldown > 0)] -= 0.5
        r[eat_action & is_eating] -= 0.3
        r[eat_action & (self.b_gapple_count <= 0)] -= 0.5
        # Only eat when LOW HP (< 8)
        good_eat = started_eating & (self.b_health < 16.0)
        r[good_eat] += 10.0
        wasteful_eat = started_eating & (self.b_health > 18.0)
        r[wasteful_eat] -= 5.0
        # Must be retreating while eating
        retreating_while_eating = is_eating & (dist_delta < -0.05)
        r[retreating_while_eating] += 0.5
        approaching_while_eating = is_eating & (dist_delta > 0.05) & (dist_2d < 5.0)
        r[approaching_while_eating] -= 1.0

        # Splash pot
        pot_action = a_actions[:, ACT_THROW_POT].astype(bool)
        r[a_pot_splash_enemy] -= 8.0
        r[pot_action & ((self.b_eating_ticks > 0) | (self.b_pot_lockout > 0))] -= 0.3
        wasteful_pot = pot_action & (self.b_pot_count > 0) & (self.b_pot_lockout <= 0) & (self.b_health > 12.0)
        r[wasteful_pot] -= 5.0
        r[pot_action & (self.b_pot_count <= 0)] -= 0.5
        # Only pot when critically low
        critical_pot = pot_action & (self.b_pot_count > 0) & (self.b_pot_lockout <= 0) & (self.b_health < 8.0)
        r[critical_pot] += 10.0

        # Pearl
        pearl_action = a_actions[:, ACT_THROW_PEARL].astype(bool)
        pearl_far = a_pearled & (dist_2d > 8.0)
        r[pearl_far] += 1.5
        pearl_close = a_pearled & (dist_2d < ATTACK_REACH)
        r[pearl_close] -= 3.0
        no_pearls = pearl_action & ((self.b_pearl_count <= 0) | (self.b_pearl_cooldown > 0))
        r[no_pearls] -= 0.1

        rb["consumables"] += r.sum()
        rewards += r

        return np.clip(rewards, -25.0, 25.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """Build observation for player A. Shape (n, OBS_DIM).
        Layout matches Java ObservationBuilder exactly."""
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)

        # ── self_state[0:38] — matches Java buildSelfState() ──
        obs[:, 0] = self.a_health / 20.0            # [0] health/20
        obs[:, 1] = 1.0                              # [1] maxHealth/20
        obs[:, 2] = self.a_absorption / 20.0         # [2] absorption/20
        obs[:, 3] = self.armor / 20.0                # [3] armor/20
        obs[:, 4] = 0.0                              # [4] armorToughness/20 (not tracked)
        obs[:, 5] = self.avx                          # [5] vel.x
        obs[:, 6] = self.avy                          # [6] vel.y
        obs[:, 7] = self.avz                          # [7] vel.z
        obs[:, 8] = self.ayaw / 180.0                 # [8] yaw/180
        obs[:, 9] = 0.0                              # [9] pitch/90 (not tracked)
        obs[:, 10] = self.a_on_ground.astype(np.float32)  # [10] onGround
        obs[:, 11] = self.a_sprinting.astype(np.float32)  # [11] isSprinting
        obs[:, 12] = 0.0                              # [12] isSneaking
        obs[:, 13] = self.a_blocking.astype(np.float32)   # [13] isBlocking
        obs[:, 14] = (self.a_eating_ticks > 0).astype(np.float32)  # [14] isUsingItem
        obs[:, 15] = self.a_fall_dist / 10.0          # [15] fallDistance/10
        obs[:, 16] = 1.0                              # [16] weapon=sword
        obs[:, 17] = 0.0                              # [17] weapon=axe
        obs[:, 18] = 0.0                              # [18] weapon=other
        obs[:, 19] = 0.0                              # [19] attackDamage/20 (not tracked)
        obs[:, 20] = self.a_gapple_count / 64.0       # [20] goldenApples/64
        obs[:, 21] = self.a_pot_count / 64.0          # [21] splashPots/64
        obs[:, 22] = self.a_pearl_count / 16.0        # [22] enderPearls/16
        # [23] totems = 0
        obs[:, 24] = self.a_eat_cooldown / GAP_COOLDOWN_TICKS   # [24] gapple cooldown (0=ready, 1=full CD)
        obs[:, 25] = self.a_eating_ticks / GAP_EAT_TICKS        # [25] eating progress (0=not eating, 1=just started)
        obs[:, 26] = self.a_pot_lockout / 10.0                  # [26] pot throw lockout
        obs[:, 27] = self.a_pearl_cooldown / PEARL_COOLDOWN_TICKS  # [27] pearl cooldown (0=ready)
        obs[:, 28] = (self.a_regen_ticks > 0).astype(np.float32)  # [28] regen active (matches Java slot 28)
        obs[:, 29] = self.a_regen_ticks / GAP_REGEN_TICKS       # [29] regen ticks remaining
        obs[:, 30] = self.a_combo_streak / 10.0                 # [30] own combo streak
        obs[:, 31] = self.b_combo_streak / 10.0                 # [31] enemy combo streak on us
        # [32] = 0
        obs[:, 33] = self.a_hurt_time / 10.0          # [33] own hurtTime/10 (i-frames)
        # [34-37] sigil combat state
        obs[:, 34] = self.a_brace_timer / 200.0
        obs[:, 35] = self.a_grace_timer / 200.0
        obs[:, 36] = self.a_brace_charges / 30.0
        obs[:, 37] = self.a_brace_cd / 600.0

        # ── entity_features[38:230] — matches Java buildEntityFeatures() ──
        base = 38
        dx = self.bx - self.ax
        dy = self.by - self.ay
        dz = self.bz - self.az
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        obs[:, base + 0] = -1.0                       # [0] alliance (enemy)
        obs[:, base + 1] = dx / 30.0                  # [1] dx/30
        obs[:, base + 2] = dy / 30.0                  # [2] dy/30
        obs[:, base + 3] = dz / 30.0                  # [3] dz/30
        obs[:, base + 4] = dist / 30.0                # [4] dist/30
        obs[:, base + 5] = self.b_health / 20.0       # [5] health/20
        obs[:, base + 6] = 1.0                        # [6] maxHealth/20
        obs[:, base + 7] = self.armor / 20.0          # [7] armor/20
        obs[:, base + 8] = self.bvx                    # [8] vel.x
        obs[:, base + 9] = self.bvy                    # [9] vel.y
        obs[:, base + 10] = self.bvz                   # [10] vel.z
        obs[:, base + 11] = self.byaw / 180.0         # [11] yaw/180
        obs[:, base + 12] = self.b_on_ground.astype(np.float32)  # [12] onGround
        obs[:, base + 13] = self.b_sprinting.astype(np.float32)  # [13] isSprinting
        obs[:, base + 14] = self.b_blocking.astype(np.float32)   # [14] isBlocking
        obs[:, base + 15] = (self.b_eating_ticks > 0).astype(np.float32)  # [15] isUsingItem
        obs[:, base + 16] = 1.0                       # [16] weapon=sword
        # [17-18] weapon=0, [19] attackDmg=0
        obs[:, base + 20] = self.b_hurt_time / 10.0   # [20] hurtTime/10
        obs[:, base + 21] = 1.0                       # [21] isCurrentTarget

        # Entity mask: slot 0 = 1 (enemy present)
        obs[:, 230] = 1.0

        # ── sigil_state[238:286] — 12 slots x 4 dims ──
        sig_base = 238
        # Slot 0: King's Brace
        obs[:, sig_base + 0] = self.a_brace_cd / 600.0
        obs[:, sig_base + 1] = np.minimum(self.a_brace_charges / 30.0, 3.0)
        obs[:, sig_base + 2] = (self.a_brace_timer > 0).astype(np.float32)
        # Slot 1: Cleopatra
        obs[:, sig_base + 4] = self.a_cleo_cd / 180.0
        obs[:, sig_base + 5] = (self.a_cleo_on_b > 0).astype(np.float32)
        obs[:, sig_base + 6] = (self.b_cleo_on_a > 0).astype(np.float32)
        # Slot 2: Quick Sand
        obs[:, sig_base + 8] = self.a_sand_cd / 140.0
        obs[:, sig_base + 9] = (self.a_sand_timer > 0).astype(np.float32)
        # Slot 3: Nile's Grace
        obs[:, sig_base + 12] = self.a_grace_cd / 180.0
        obs[:, sig_base + 13] = (self.a_grace_timer > 0).astype(np.float32)

        return obs

    def get_b_obs(self) -> np.ndarray:
        """Build observation for player B (mirror of A's perspective).
        Layout matches Java ObservationBuilder exactly."""
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)

        # ── self_state[0:38] — matches Java buildSelfState() ──
        obs[:, 0] = self.b_health / 20.0            # [0] health/20
        obs[:, 1] = 1.0                              # [1] maxHealth/20
        obs[:, 2] = self.b_absorption / 20.0         # [2] absorption/20
        obs[:, 3] = self.armor / 20.0                # [3] armor/20
        obs[:, 4] = 0.0                              # [4] armorToughness/20
        obs[:, 5] = self.bvx                          # [5] vel.x
        obs[:, 6] = self.bvy                          # [6] vel.y
        obs[:, 7] = self.bvz                          # [7] vel.z
        obs[:, 8] = self.byaw / 180.0                 # [8] yaw/180
        obs[:, 9] = 0.0                              # [9] pitch/90
        obs[:, 10] = self.b_on_ground.astype(np.float32)  # [10] onGround
        obs[:, 11] = self.b_sprinting.astype(np.float32)  # [11] isSprinting
        obs[:, 12] = 0.0                              # [12] isSneaking
        obs[:, 13] = self.b_blocking.astype(np.float32)   # [13] isBlocking
        obs[:, 14] = (self.b_eating_ticks > 0).astype(np.float32)  # [14] isUsingItem
        obs[:, 15] = self.b_fall_dist / 10.0          # [15] fallDistance/10
        obs[:, 16] = 1.0                              # [16] weapon=sword
        obs[:, 17] = 0.0                              # [17] weapon=axe
        obs[:, 18] = 0.0                              # [18] weapon=other
        obs[:, 19] = 0.0                              # [19] attackDamage/20
        obs[:, 20] = self.b_gapple_count / 64.0       # [20] goldenApples/64
        obs[:, 21] = self.b_pot_count / 64.0          # [21] splashPots/64
        obs[:, 22] = self.b_pearl_count / 16.0        # [22] enderPearls/16
        obs[:, 24] = self.b_eat_cooldown / GAP_COOLDOWN_TICKS   # [24] gapple cooldown
        obs[:, 25] = self.b_eating_ticks / GAP_EAT_TICKS        # [25] eating progress
        obs[:, 26] = self.b_pot_lockout / 10.0                  # [26] pot throw lockout
        obs[:, 27] = self.b_pearl_cooldown / PEARL_COOLDOWN_TICKS  # [27] pearl cooldown
        obs[:, 28] = (self.b_regen_ticks > 0).astype(np.float32)  # [28] regen active
        obs[:, 29] = self.b_regen_ticks / GAP_REGEN_TICKS       # [29] regen ticks remaining
        obs[:, 30] = self.b_combo_streak / 10.0                 # [30] own combo streak
        obs[:, 31] = self.a_combo_streak / 10.0                 # [31] enemy combo streak on us
        obs[:, 33] = self.b_hurt_time / 10.0          # [33] own hurtTime/10 (i-frames)
        # [34-37] sigil combat state
        obs[:, 34] = self.b_brace_timer / 200.0
        obs[:, 35] = self.b_grace_timer / 200.0
        obs[:, 36] = self.b_brace_charges / 30.0
        obs[:, 37] = self.b_brace_cd / 600.0

        # ── entity_features[38:230] — matches Java buildEntityFeatures() ──
        base = 38
        dx = self.ax - self.bx
        dy = self.ay - self.by
        dz = self.az - self.bz
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        obs[:, base + 0] = -1.0                       # [0] alliance
        obs[:, base + 1] = dx / 30.0                  # [1] dx/30
        obs[:, base + 2] = dy / 30.0                  # [2] dy/30
        obs[:, base + 3] = dz / 30.0                  # [3] dz/30
        obs[:, base + 4] = dist / 30.0                # [4] dist/30
        obs[:, base + 5] = self.a_health / 20.0       # [5] health/20
        obs[:, base + 6] = 1.0                        # [6] maxHealth/20
        obs[:, base + 7] = self.armor / 20.0          # [7] armor/20
        obs[:, base + 8] = self.avx                    # [8] vel.x
        obs[:, base + 9] = self.avy                    # [9] vel.y
        obs[:, base + 10] = self.avz                   # [10] vel.z
        obs[:, base + 11] = self.ayaw / 180.0         # [11] yaw/180
        obs[:, base + 12] = self.a_on_ground.astype(np.float32)  # [12] onGround
        obs[:, base + 13] = self.a_sprinting.astype(np.float32)  # [13] isSprinting
        obs[:, base + 14] = self.a_blocking.astype(np.float32)   # [14] isBlocking
        obs[:, base + 15] = (self.a_eating_ticks > 0).astype(np.float32)  # [15] isUsingItem
        obs[:, base + 16] = 1.0                       # [16] weapon=sword
        obs[:, base + 20] = self.a_hurt_time / 10.0   # [20] hurtTime/10
        obs[:, base + 21] = 1.0                       # [21] isCurrentTarget

        obs[:, 230] = 1.0

        # ── sigil_state[238:286] — 12 slots x 4 dims (B's perspective) ──
        sig_base = 238
        # Slot 0: King's Brace
        obs[:, sig_base + 0] = self.b_brace_cd / 600.0
        obs[:, sig_base + 1] = np.minimum(self.b_brace_charges / 30.0, 3.0)
        obs[:, sig_base + 2] = (self.b_brace_timer > 0).astype(np.float32)
        # Slot 1: Cleopatra
        obs[:, sig_base + 4] = self.b_cleo_cd / 180.0
        obs[:, sig_base + 5] = (self.b_cleo_on_a > 0).astype(np.float32)
        obs[:, sig_base + 6] = (self.a_cleo_on_b > 0).astype(np.float32)
        # Slot 2: Quick Sand
        obs[:, sig_base + 8] = self.b_sand_cd / 140.0
        obs[:, sig_base + 9] = (self.b_sand_timer > 0).astype(np.float32)
        # Slot 3: Nile's Grace
        obs[:, sig_base + 12] = self.b_grace_cd / 180.0
        obs[:, sig_base + 13] = (self.b_grace_timer > 0).astype(np.float32)

        return obs
