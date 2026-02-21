"""Numpy-vectorized 1v1 PvP simulator for batch RL training.

Runs N parallel fights simultaneously using pure numpy operations.
10-50x faster than the per-env Python loop in sim_env.py.

All physics match mc_physics.py (Kitara combat model).
"""

import math
import numpy as np

from .mc_physics import (
    GRAVITY, AIR_DRAG, GROUND_FRICTION, DRAG_FACTOR, JUMP_IMPULSE,
    SPRINT_JUMP_KICK, SPRINT_SPEED_MULT, SNEAK_SPEED_MULT, PLAYER_WALK_SPEED,
    BASE_SWORD_DAMAGE, SHARPNESS_PER_LEVEL, CRIT_MULTIPLIER, I_FRAME_TICKS,
    SWORD_BLOCKING_DMG_MULT, ATTACK_REACH,
    PROTECTION_DR_TABLE,
    KB_HORIZONTAL, KB_HORIZONTAL_ON_GROUND, KB_HORIZONTAL_SPRINTING,
    KB_HORIZONTAL_INHERIT, KB_HORIZONTAL_FRICTION, KB_HORIZONTAL_ENCHANT_DEDUCTION,
    KB_VERTICAL, KB_VERTICAL_ON_GROUND, KB_VERTICAL_IN_AIR, KB_VERTICAL_SPRINTING,
    KB_VERTICAL_INHERIT, KB_SLOWDOWN, KB_CANCEL_SPRINT,
    KB_SWORD_BLOCK_HORIZONTAL, KB_SWORD_BLOCK_VERTICAL,
)
from .sim_env import OBS_DIM, NUM_ACTIONS

# Action indices
ACT_FORWARD = 0
ACT_BACKWARD = 1
ACT_LEFT = 2
ACT_RIGHT = 3
ACT_JUMP = 4
ACT_SNEAK = 5
ACT_SPRINT = 6
ACT_ATTACK = 7


class VecPvPSim:
    """Numpy-vectorized N-parallel 1v1 PvP simulator.

    State is stored as parallel numpy arrays of shape (N,).
    All physics computations are batched.
    """

    def __init__(self, n_envs: int, arena_half: float = 30.0,
                 episode_length: int = 1800, seed: int = 0):
        self.n = n_envs
        self.arena_half = arena_half
        self.floor_y = 0.0
        self.episode_length = episode_length
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

        # Episode tracking
        self.tick = np.zeros(n_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(n_envs)

        # Previous state for reward shaping
        self._prev_dist = np.zeros(n_envs)
        self._prev_health_diff = np.zeros(n_envs)

        # Combat config (constant across all envs)
        self.armor = 20
        self.sharpness = 6
        # Server direct damage uses simplified armor: DR = armor * 0.04, capped at 80%
        self.armor_mult = 1.0 - min(self.armor * 0.04, 0.8)

        # Damage formula (pre-computed for standard loadout)
        # No protection enchant DR (server direct damage doesn't apply it)
        self.base_dmg = BASE_SWORD_DAMAGE + SHARPNESS_PER_LEVEL * self.sharpness
        self.base_dmg_after_armor = self.base_dmg * self.armor_mult
        self.crit_dmg_after_armor = self.base_dmg * CRIT_MULTIPLIER * self.armor_mult

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

        self._prev_dist[idx] = dist
        self._prev_health_diff[idx] = 0.0

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

        # Auto-aim: both players face each other (matches BotBrain auto-aim behavior)
        dx_ab = self.bx - self.ax
        dz_ab = self.bz - self.az
        self.ayaw[:] = np.degrees(np.arctan2(-dx_ab, dz_ab))
        self.byaw[:] = np.degrees(np.arctan2(dx_ab, -dz_ab))

        # Decrement i-frames
        self.a_hurt_time = np.maximum(0, self.a_hurt_time - 1)
        self.b_hurt_time = np.maximum(0, self.b_hurt_time - 1)

        # Movement for both players
        self._step_movement_batch(
            self.ax, self.ay, self.az, self.avx, self.avy, self.avz,
            self.ayaw, self.a_on_ground, self.a_sprinting, self.a_fall_dist,
            a_actions
        )
        self._step_movement_batch(
            self.bx, self.by, self.bz, self.bvx, self.bvy, self.bvz,
            self.byaw, self.b_on_ground, self.b_sprinting, self.b_fall_dist,
            b_actions
        )

        # Combat: A attacks B
        a_dmg = self._step_combat_batch(
            self.ax, self.ay, self.az, self.avx, self.avz,
            self.ayaw, self.a_on_ground, self.a_sprinting, self.a_fall_dist,
            self.bx, self.by, self.bz, self.bvx, self.bvy, self.bvz,
            self.byaw, self.b_on_ground, self.b_sprinting,
            self.b_health, self.b_absorption, self.b_hurt_time,
            a_actions[:, ACT_ATTACK].astype(bool),
        )

        # Combat: B attacks A
        b_dmg = self._step_combat_batch(
            self.bx, self.by, self.bz, self.bvx, self.bvz,
            self.byaw, self.b_on_ground, self.b_sprinting, self.b_fall_dist,
            self.ax, self.ay, self.az, self.avx, self.avy, self.avz,
            self.ayaw, self.a_on_ground, self.a_sprinting,
            self.a_health, self.a_absorption, self.a_hurt_time,
            b_actions[:, ACT_ATTACK].astype(bool),
        )

        # Rewards (for player A)
        rewards = self._compute_rewards(a_actions, a_dmg, b_dmg)
        self.episode_rewards += rewards

        # Termination
        a_dead = self.a_health <= 0
        b_dead = self.b_health <= 0
        terminated = a_dead | b_dead
        truncated = self.tick >= self.episode_length
        dones = terminated | truncated

        obs = self._get_obs()

        infos = {
            "a_health": self.a_health.copy(),
            "b_health": self.b_health.copy(),
            "episode_reward": self.episode_rewards.copy(),
            "tick": self.tick.copy(),
        }

        # Auto-reset done envs
        self.reset_envs(dones)

        return obs, rewards, dones, infos

    def _step_movement_batch(self, x, y, z, vx, vy, vz, yaw,
                              on_ground, sprinting, fall_dist, actions):
        """Vectorized movement physics for N players."""
        n = len(x)
        fwd = actions[:, ACT_FORWARD].astype(np.float32) - actions[:, ACT_BACKWARD].astype(np.float32)
        strafe = actions[:, ACT_RIGHT].astype(np.float32) - actions[:, ACT_LEFT].astype(np.float32)
        jump = actions[:, ACT_JUMP].astype(bool)
        sprint_input = actions[:, ACT_SPRINT].astype(bool)
        sneak = actions[:, ACT_SNEAK].astype(bool)

        was_on_ground = on_ground.copy()

        # Update sprint state
        sprinting[:] = sprint_input & ~sneak & (fwd > 0)

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
                            tx, ty, tz, tvx, tvy, tvz, tyaw, t_on_ground, t_sprint,
                            t_health, t_absorption, t_hurt_time, attack_mask):
        """Vectorized combat: attacker → target. Returns damage dealt (n,)."""
        n = len(ax)
        damage = np.zeros(n)

        # Range check
        dx = tx - ax
        dy = ty - ay
        dz = tz - az
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        in_range = dist <= ATTACK_REACH

        # I-frame check: server blocks ALL hits during i-frames
        can_hit = t_hurt_time <= 0

        # Active mask: attacking AND in range AND can hit
        active = attack_mask & in_range & can_hit
        if not np.any(active):
            return damage

        # Critical hit: falling, not on ground
        crit = (a_fall > 0) & ~a_on_ground

        # Damage
        dmg = np.where(crit & active, self.crit_dmg_after_armor, self.base_dmg_after_armor)
        dmg = np.where(active, dmg, 0.0)
        damage[:] = dmg

        # Apply damage
        # Absorption first
        absorbed = np.minimum(t_absorption, dmg)
        t_absorption -= absorbed * active
        remaining = dmg - absorbed
        t_health -= remaining * active
        t_health[:] = np.maximum(0.0, t_health)
        t_hurt_time[active] = I_FRAME_TICKS

        # Knockback (matching server binary sprint model)
        # Server: kbStrength = sprintHit ? 0.9 : 0.4
        # Direction: away from attacker (target - attacker normalized)
        kb_dx = tx - ax
        kb_dz = tz - az
        kb_dist = np.sqrt(kb_dx**2 + kb_dz**2)
        kb_dist = np.maximum(kb_dist, 0.001)  # avoid /0
        kb_nx = kb_dx / kb_dist
        kb_nz = kb_dz / kb_dist

        sprint_mask = active & a_sprint
        kb_strength = np.where(sprint_mask, 0.9, 0.4)
        kb_y = np.full(n, 0.36)

        # Apply KB additively to target velocity (matching server .add())
        tvx[active] += (kb_nx * kb_strength)[active]
        tvy[active] += kb_y[active]
        tvz[active] += (kb_nz * kb_strength)[active]
        t_on_ground[active] = False

        return damage

    def _compute_rewards(self, a_actions, a_dmg_dealt, b_dmg_dealt):
        """Compute reward for player A.

        Reward design:
        - Strong kill/death signals for terminal reward
        - Significant per-hit rewards to learn attacking
        - Distance shaping to learn approach behavior
        - Small engagement bonus for being in combat range
        """
        rewards = np.zeros(self.n)

        # Kill / death (terminal)
        rewards[self.b_health <= 0] += 10.0
        rewards[self.a_health <= 0] -= 5.0

        # Damage dealt (strong signal — this is the core skill)
        rewards += a_dmg_dealt * 0.5

        # Damage taken (penalize, but less than damage dealt to encourage aggression)
        rewards -= b_dmg_dealt * 0.2

        # Whiff penalty (attacking out of range)
        attacking = a_actions[:, ACT_ATTACK].astype(bool)
        dx = self.bx - self.ax
        dz = self.bz - self.az
        dy = self.by - self.ay
        dist = np.sqrt(dx**2 + dz**2 + dy**2)
        whiff = attacking & (dist > ATTACK_REACH)
        rewards[whiff] -= 0.05
        # I-frame waste
        iframe_waste = attacking & (dist <= ATTACK_REACH) & (self.b_hurt_time > 0)
        rewards[iframe_waste] -= 0.03

        # Distance shaping: reward getting closer (potential-based)
        dist_2d = np.sqrt(dx**2 + dz**2)
        rewards += 0.1 * (self._prev_dist - dist_2d)

        # Health advantage shaping
        health_diff = (self.a_health - self.b_health) / 20.0
        rewards += 0.05 * (health_diff - self._prev_health_diff)

        # Small engagement reward: bonus for being in attack range
        in_range = dist <= ATTACK_REACH
        rewards[in_range] += 0.02

        self._prev_dist[:] = dist_2d
        self._prev_health_diff[:] = health_diff

        return np.clip(rewards, -10.0, 10.0).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        """Build observation for player A. Shape (n, OBS_DIM)."""
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)

        # Self state
        obs[:, 0] = self.a_health / 20.0
        obs[:, 1] = 1.0  # hunger
        obs[:, 2] = self.a_absorption / 20.0
        obs[:, 3] = self.armor / 20.0
        speed = np.sqrt(self.avx**2 + self.avz**2)
        obs[:, 4] = np.minimum(1.0, speed / 0.2806)
        obs[:, 5] = self.a_on_ground.astype(np.float32)
        yaw_rad = np.radians(self.ayaw)
        obs[:, 6] = np.cos(yaw_rad)
        obs[:, 7] = np.sin(yaw_rad)
        obs[:, 10] = 1.0  # attack cooldown
        obs[:, 11] = self.a_sprinting.astype(np.float32)

        # Entity features (enemy at slot 0, starting at index 38)
        base = 38
        dx = self.bx - self.ax
        dz = self.bz - self.az
        dy = self.by - self.ay
        dist = np.sqrt(dx**2 + dz**2 + dy**2)
        obs[:, base + 0] = -1.0  # enemy
        obs[:, base + 1] = dx / self.arena_half
        obs[:, base + 2] = dz / self.arena_half
        obs[:, base + 3] = dy / 10.0
        obs[:, base + 4] = dist / self.arena_half
        obs[:, base + 5] = self.b_health / 20.0
        obs[:, base + 6] = self.armor / 20.0
        obs[:, base + 7] = self.b_absorption / 20.0
        e_speed = np.sqrt(self.bvx**2 + self.bvz**2)
        safe_speed = np.where(e_speed > 0, 0.2806, 1.0)
        obs[:, base + 8] = self.bvx / safe_speed
        obs[:, base + 9] = self.bvz / safe_speed
        obs[:, base + 10] = self.bvy / 0.42
        e_yaw_rad = np.radians(self.byaw)
        obs[:, base + 11] = np.cos(e_yaw_rad)
        obs[:, base + 12] = np.sin(e_yaw_rad)
        obs[:, base + 16] = self.b_on_ground.astype(np.float32)

        # Entity mask: slot 0 = 1 (enemy present)
        obs[:, 230] = 1.0

        return obs

    def get_b_obs(self) -> np.ndarray:
        """Build observation for player B (mirror of A's perspective)."""
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)

        obs[:, 0] = self.b_health / 20.0
        obs[:, 1] = 1.0
        obs[:, 2] = self.b_absorption / 20.0
        obs[:, 3] = self.armor / 20.0
        speed = np.sqrt(self.bvx**2 + self.bvz**2)
        obs[:, 4] = np.minimum(1.0, speed / 0.2806)
        obs[:, 5] = self.b_on_ground.astype(np.float32)
        yaw_rad = np.radians(self.byaw)
        obs[:, 6] = np.cos(yaw_rad)
        obs[:, 7] = np.sin(yaw_rad)
        obs[:, 10] = 1.0
        obs[:, 11] = self.b_sprinting.astype(np.float32)

        base = 38
        dx = self.ax - self.bx
        dz = self.az - self.bz
        dy = self.ay - self.by
        dist = np.sqrt(dx**2 + dz**2 + dy**2)
        obs[:, base + 0] = -1.0
        obs[:, base + 1] = dx / self.arena_half
        obs[:, base + 2] = dz / self.arena_half
        obs[:, base + 3] = dy / 10.0
        obs[:, base + 4] = dist / self.arena_half
        obs[:, base + 5] = self.a_health / 20.0
        obs[:, base + 6] = self.armor / 20.0
        obs[:, base + 7] = self.a_absorption / 20.0
        e_speed = np.sqrt(self.avx**2 + self.avz**2)
        safe_speed = np.where(e_speed > 0, 0.2806, 1.0)
        obs[:, base + 8] = self.avx / safe_speed
        obs[:, base + 9] = self.avz / safe_speed
        obs[:, base + 10] = self.avy / 0.42
        e_yaw_rad = np.radians(self.ayaw)
        obs[:, base + 11] = np.cos(e_yaw_rad)
        obs[:, base + 12] = np.sin(e_yaw_rad)
        obs[:, base + 16] = self.a_on_ground.astype(np.float32)

        obs[:, 230] = 1.0
        return obs
