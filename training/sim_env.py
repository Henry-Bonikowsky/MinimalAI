"""Gymnasium-compatible 1v1 PvP environment using vanilla MC physics.

Uses MCSimulator as the backend for fast batch RL training.
Observation and action spaces match the MinimalAI Java plugin exactly.

Usage:
    python -m training.sim_env --test   # quick smoke test
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .mc_physics import MCSimulator, PlayerState, ATTACK_REACH, I_FRAME_TICKS

# ── Action indices (matches Java ActionExecutor / existing combat_sim) ──
ACT_FORWARD = 0
ACT_BACKWARD = 1
ACT_STRAFE_LEFT = 2
ACT_STRAFE_RIGHT = 3
ACT_JUMP = 4
ACT_SNEAK = 5
ACT_SPRINT = 6
ACT_ATTACK = 7
ACT_BLOCK = 8
ACT_EAT_GAP = 9
ACT_THROW_POT = 10
ACT_THROW_PEARL = 11
ACT_SPRINT_RESET = 12
NUM_ACTIONS = 35

# Observation dimensions (flat vector, matches ObservationBuilder.java)
OBS_DIM = 320

# Arena settings
ARENA_HALF = 30.0
SPAWN_DIST_MIN = 4.0
SPAWN_DIST_MAX = 8.0
MAX_EPISODE_TICKS = 1800  # 90 seconds at 20 TPS


class PvPEnv(gym.Env):
    """1v1 PvP environment backed by vanilla MC physics simulator.

    Observation: 320-dim float32 vector (matches MinimalAI Java obs exactly)
    Action: MultiBinary(35)
    """

    metadata = {"render_modes": ["none"], "render_fps": 20}

    def __init__(
        self,
        episode_length: int = MAX_EPISODE_TICKS,
        arena_half: float = ARENA_HALF,
        seed: int = None,
        render_mode: str = "none",
    ):
        super().__init__()
        self.episode_length = episode_length
        self.arena_half = arena_half
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)
        self.sim = MCSimulator(arena_half_size=arena_half)

        self.action_space = spaces.MultiBinary(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.player = PlayerState()
        self.enemy = PlayerState()
        self.tick = 0
        self._prev_dist = 0.0
        self._prev_health_diff = 0.0
        self._episode_reward = 0.0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.tick = 0
        self._episode_reward = 0.0

        # Spawn player at center
        self.player = PlayerState(x=0.0, y=0.0, z=0.0)
        self.player.yaw = self.rng.uniform(-180, 180)

        # Spawn enemy at random distance
        angle = self.rng.uniform(0, 2 * np.pi)
        dist = self.rng.uniform(SPAWN_DIST_MIN, SPAWN_DIST_MAX)
        self.enemy = PlayerState(
            x=np.cos(angle) * dist,
            y=0.0,
            z=np.sin(angle) * dist,
        )
        self.enemy.yaw = np.degrees(angle + np.pi)  # face player

        self._prev_dist = dist
        self._prev_health_diff = 0.0

        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action: np.ndarray):
        self.tick += 1

        # Decode action vector
        player_actions = self._decode_actions(action, self.player)
        enemy_actions = self._enemy_ai()

        # Get friction from terrain (flat arena = 0.6)
        friction = 0.6

        # Run simulation tick
        self.player, self.enemy, events = self.sim.step(
            self.player, self.enemy,
            player_actions, enemy_actions,
            friction=friction,
        )

        # Calculate reward
        reward = self._calculate_reward(action, events)
        self._episode_reward += reward

        # Check termination
        player_dead = self.player.health <= 0
        enemy_dead = self.enemy.health <= 0
        terminated = player_dead or enemy_dead
        truncated = self.tick >= self.episode_length

        obs = self._get_obs()
        return obs, reward, terminated, truncated, self._get_info()

    def _decode_actions(self, action: np.ndarray, state: PlayerState) -> dict:
        """Convert MultiBinary(35) to action dict for MCSimulator."""
        fwd = float(action[ACT_FORWARD]) - float(action[ACT_BACKWARD])
        strafe = float(action[ACT_STRAFE_RIGHT]) - float(action[ACT_STRAFE_LEFT])

        # Sprint reset: toggle sprint off then back on
        sprint = bool(action[ACT_SPRINT])
        if action[ACT_SPRINT_RESET] and state.sprinting:
            sprint = True  # re-sprint after reset

        return {
            "forward": fwd,
            "strafe": strafe,
            "jump": bool(action[ACT_JUMP]),
            "sprint": sprint,
            "sneak": bool(action[ACT_SNEAK]),
            "attack": bool(action[ACT_ATTACK]),
        }

    def _enemy_ai(self) -> dict:
        """Simple rule-based enemy AI."""
        dx = self.player.x - self.enemy.x
        dz = self.player.z - self.enemy.z
        dist = np.sqrt(dx**2 + dz**2)

        # Face player
        self.enemy.yaw = np.degrees(np.arctan2(-dx, dz))

        # Movement
        if dist > 6.0:
            fwd = 0.8 if self.rng.random() < 0.7 else 0.0
            sprint = self.rng.random() > 0.4
        elif dist > 3.0:
            fwd = 0.5
            sprint = self.rng.random() > 0.6
        else:
            fwd = self.rng.choice([-0.5, 0.0, 0.3])
            sprint = False

        strafe = self.rng.choice([-1.0, 0.0, 1.0]) * 0.5
        attack = dist <= ATTACK_REACH and self.rng.random() < 0.5

        return {
            "forward": fwd,
            "strafe": strafe,
            "jump": self.rng.random() > 0.92,
            "sprint": sprint,
            "sneak": False,
            "attack": attack,
        }

    def _calculate_reward(self, action: np.ndarray, events: list) -> float:
        """Calculate per-tick reward."""
        reward = 0.0

        # Sparse: kill / death
        if self.enemy.health <= 0:
            reward += 5.0
        if self.player.health <= 0:
            reward -= 3.0

        # Dense: damage dealt / taken
        for ev in events:
            if ev.damage > 0:
                reward += (ev.damage / 20.0) * 0.5

        # Whiff penalty
        if action[ACT_ATTACK]:
            dx = self.enemy.x - self.player.x
            dz = self.enemy.z - self.player.z
            dy = self.enemy.y - self.player.y
            dist = np.sqrt(dx**2 + dz**2 + dy**2)
            if dist > ATTACK_REACH:
                reward -= 0.05
            elif self.enemy.hurt_time > 0:
                reward -= 0.03  # i-frame waste

        # Potential shaping: distance + health
        dx = self.enemy.x - self.player.x
        dz = self.enemy.z - self.player.z
        dist = np.sqrt(dx**2 + dz**2)
        health_diff = (self.player.health - self.enemy.health) / 20.0

        dist_reward = 0.05 * (self._prev_dist - dist) / 30.0
        health_reward = 0.02 * (health_diff - self._prev_health_diff)

        self._prev_dist = dist
        self._prev_health_diff = health_diff
        reward += dist_reward + health_reward

        return float(np.clip(reward, -5.0, 5.0))

    def _get_obs(self) -> np.ndarray:
        """Build 320-dim observation vector."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        p = self.player
        e = self.enemy

        # Self state (first 38 dims match self_state from entities.py)
        obs[0] = p.health / 20.0
        obs[1] = 1.0  # hunger (always full)
        obs[2] = p.absorption / 20.0
        obs[3] = p.armor / 20.0
        speed = np.sqrt(p.vx**2 + p.vz**2)
        obs[4] = min(1.0, speed / 0.2806)
        obs[5] = 1.0 if p.on_ground else 0.0
        yaw_rad = np.radians(p.yaw)
        obs[6] = np.cos(yaw_rad)
        obs[7] = np.sin(yaw_rad)
        obs[8] = 0.0  # pitch cos
        obs[9] = 0.0  # pitch sin
        obs[10] = 1.0  # attack cooldown (always ready)
        obs[11] = 1.0 if p.sprinting else 0.0
        obs[12] = 1.0 if p.sneaking else 0.0
        obs[13] = 0.0  # blocking

        # Entity features starting at index 38
        # Target relative position
        dx = e.x - p.x
        dz = e.z - p.z
        dy = e.y - p.y
        dist = np.sqrt(dx**2 + dz**2 + dy**2)
        base = 38
        obs[base + 0] = -1.0  # enemy alliance
        obs[base + 1] = dx / self.arena_half
        obs[base + 2] = dz / self.arena_half
        obs[base + 3] = dy / 10.0
        obs[base + 4] = dist / self.arena_half
        obs[base + 5] = e.health / 20.0
        obs[base + 6] = e.armor / 20.0
        obs[base + 7] = e.absorption / 20.0
        e_speed = np.sqrt(e.vx**2 + e.vz**2)
        obs[base + 8] = e.vx / 0.2806 if e_speed > 0 else 0.0
        obs[base + 9] = e.vz / 0.2806 if e_speed > 0 else 0.0
        obs[base + 10] = e.vy / 0.42 if abs(e.vy) > 0 else 0.0
        e_yaw_rad = np.radians(e.yaw)
        obs[base + 11] = np.cos(e_yaw_rad)
        obs[base + 12] = np.sin(e_yaw_rad)
        obs[base + 16] = 1.0 if e.on_ground else 0.0

        return obs

    def _get_info(self) -> dict:
        dx = self.enemy.x - self.player.x
        dz = self.enemy.z - self.player.z
        return {
            "tick": self.tick,
            "player_health": self.player.health,
            "enemy_health": self.enemy.health,
            "distance": np.sqrt(dx**2 + dz**2),
            "episode_reward": self._episode_reward,
        }


class SimVecEnv:
    """Vectorized PvP environment running N parallel fights.

    Not a true gym.VectorEnv — just a simple wrapper for batch training.
    Each sub-env is independent and auto-resets on termination.
    """

    def __init__(self, n_envs: int, episode_length: int = MAX_EPISODE_TICKS, seed: int = 0):
        self.n_envs = n_envs
        self.envs = [
            PvPEnv(episode_length=episode_length, seed=seed + i)
            for i in range(n_envs)
        ]
        self.obs_dim = OBS_DIM
        self.num_actions = NUM_ACTIONS

    def reset(self) -> np.ndarray:
        """Reset all envs. Returns (n_envs, OBS_DIM) observations."""
        obs = np.zeros((self.n_envs, OBS_DIM), dtype=np.float32)
        for i, env in enumerate(self.envs):
            obs[i], _ = env.reset()
        return obs

    def step(self, actions: np.ndarray):
        """Step all envs. actions: (n_envs, NUM_ACTIONS).

        Returns:
            obs: (n_envs, OBS_DIM)
            rewards: (n_envs,)
            dones: (n_envs,) bool
            infos: list of dicts
        """
        obs = np.zeros((self.n_envs, OBS_DIM), dtype=np.float32)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos = []

        for i, env in enumerate(self.envs):
            o, r, terminated, truncated, info = env.step(actions[i])
            done = terminated or truncated
            rewards[i] = r
            dones[i] = done
            infos.append(info)

            if done:
                o, _ = env.reset()
            obs[i] = o

        return obs, rewards, dones, infos


def _test():
    """Quick smoke test."""
    print("Testing PvPEnv...")
    env = PvPEnv(episode_length=1000, seed=42)
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,), f"obs shape: {obs.shape}"
    print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")

    total_reward = 0.0
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            print(f"  Episode ended at tick {step+1}: reward={total_reward:.3f}, "
                  f"player_hp={info['player_health']:.1f}, enemy_hp={info['enemy_health']:.1f}")
            obs, info = env.reset()
            total_reward = 0.0

    print("  Single env OK")

    print("\nTesting SimVecEnv(16)...")
    vec = SimVecEnv(16, episode_length=500, seed=0)
    obs = vec.reset()
    assert obs.shape == (16, OBS_DIM), f"vec obs shape: {obs.shape}"

    for step in range(500):
        actions = np.random.randint(0, 2, size=(16, NUM_ACTIONS))
        obs, rewards, dones, infos = vec.step(actions)

    print(f"  Completed 500 steps x 16 envs = {500*16} total steps")
    print(f"  obs shape: {obs.shape}, rewards range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print("  Vec env OK")

    print("\nAll tests passed!")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _test()
    else:
        print("Usage: python -m training.sim_env --test")
