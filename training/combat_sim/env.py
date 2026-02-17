"""Gymnasium environment for Minecraft PvP combat simulation.

Supports 1vN and NvN scenarios with variable entity counts.
Observation is a dict with self_state, entity_features, combat_ctx, sigil_state, env_state.
Action is a multi-binary vector of 28 discrete actions.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .entities import Agent, Alliance, WeaponType, WEAPON_STATS, MAX_HEALTH, ARENA_SIZE, SPRINT_SPEED
from .physics import CombatPhysics


# Action indices
ACT_FORWARD = 0
ACT_BACKWARD = 1
ACT_LEFT = 2
ACT_RIGHT = 3
ACT_JUMP = 4
ACT_SNEAK = 5
ACT_SPRINT = 6
ACT_ATTACK = 7
ACT_USE = 8
ACT_SWITCH_WEAPON = 9
ACT_SWITCH_CONSUMABLE = 10
ACT_CONSUME = 11
ACT_WTAP = 12
ACT_STRAFE_LEFT_ATTACK = 13
ACT_STRAFE_RIGHT_ATTACK = 14
ACT_BLOCK_HIT = 15
ACT_SIGIL_0 = 16
ACT_SIGIL_1 = 17
ACT_SIGIL_2 = 18
ACT_SIGIL_3 = 19
ACT_LOOK_UP = 20
ACT_LOOK_DOWN = 21
ACT_LOOK_LEFT = 22
ACT_LOOK_RIGHT = 23
ACT_TARGET_NEAREST = 24
ACT_TARGET_LOWEST_HP = 25
ACT_TARGET_HIGHEST_THREAT = 26
ACT_TARGET_CYCLE = 27

NUM_ACTIONS = 28
MAX_ENTITIES = 32
ENTITY_FEATURE_DIM = 20
SELF_STATE_DIM = 30
COMBAT_CTX_DIM = 22
SIGIL_STATE_DIM = 12
ENV_STATE_DIM = 8


class CombatEnv(gym.Env):
    """Minecraft PvP combat environment.

    Args:
        num_enemies: Number of enemy agents.
        num_allies: Number of ally agents (not counting the player).
        episode_length: Max ticks per episode.
        domain_randomization: Whether to randomize physics params.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 20}

    def __init__(
        self,
        num_enemies: int = 1,
        num_allies: int = 0,
        episode_length: int = 1800,
        domain_randomization: bool = True,
        render_mode: str = "none",
        seed: int = None,
    ):
        super().__init__()

        self.num_enemies = num_enemies
        self.num_allies = num_allies
        self.episode_length = episode_length
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed)
        self.physics = CombatPhysics(domain_randomization=domain_randomization, rng=self.rng)

        # Create agents
        self.player = Agent(agent_id=0, alliance=Alliance.ALLY)
        self.enemies: list[Agent] = []
        self.allies: list[Agent] = []
        self._all_entities: list[Agent] = []

        # Action space: multi-binary (each action is independently on/off)
        self.action_space = spaces.MultiBinary(NUM_ACTIONS)

        # Observation space: dict of arrays
        self.observation_space = spaces.Dict({
            "self_state": spaces.Box(-1, 1, shape=(SELF_STATE_DIM,), dtype=np.float32),
            "entity_features": spaces.Box(-1, 1, shape=(MAX_ENTITIES, ENTITY_FEATURE_DIM), dtype=np.float32),
            "entity_mask": spaces.Box(0, 1, shape=(MAX_ENTITIES,), dtype=np.float32),
            "combat_ctx": spaces.Box(-1, 1, shape=(COMBAT_CTX_DIM,), dtype=np.float32),
            "sigil_state": spaces.Box(0, 1, shape=(SIGIL_STATE_DIM,), dtype=np.float32),
            "env_state": spaces.Box(-1, 1, shape=(ENV_STATE_DIM,), dtype=np.float32),
        })

        self.current_tick = 0
        self.episode_reward = 0.0

        # Reward tracking
        self._prev_health_diff = 0.0
        self._prev_dist_to_target = 0.0
        self._prev_facing_target = 0.0

    def _create_entities(self):
        """Create enemy and ally agents."""
        self.enemies = []
        self.allies = []

        for i in range(self.num_enemies):
            enemy = Agent(agent_id=i + 1, alliance=Alliance.ENEMY, weapon=WeaponType.SWORD)
            self.enemies.append(enemy)

        for i in range(self.num_allies):
            ally = Agent(
                agent_id=self.num_enemies + i + 1,
                alliance=Alliance.ALLY,
                weapon=WeaponType.SWORD,
            )
            self.allies.append(ally)

        self._all_entities = self.enemies + self.allies

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.physics.rng = self.rng

        self.physics.randomize_params()
        self._create_entities()

        # Spawn player at center
        self.player.reset(x=0.0, z=0.0, facing=0.0)

        # Spawn enemies in a circle around player
        for i, enemy in enumerate(self.enemies):
            angle = 2 * np.pi * i / max(self.num_enemies, 1) + self.rng.uniform(-0.3, 0.3)
            dist = self.rng.uniform(4.0, 8.0)
            enemy.reset(
                x=np.cos(angle) * dist,
                z=np.sin(angle) * dist,
                facing=angle + np.pi,  # face toward center
            )

        # Spawn allies near player
        for i, ally in enumerate(self.allies):
            angle = 2 * np.pi * i / max(self.num_allies, 1) + self.rng.uniform(-0.3, 0.3)
            dist = self.rng.uniform(2.0, 4.0)
            ally.reset(
                x=np.cos(angle) * dist,
                z=np.sin(angle) * dist,
                facing=angle + np.pi,
            )

        # Default target: nearest enemy
        self.player.target_id = self.enemies[0].agent_id if self.enemies else -1

        self.current_tick = 0
        self.episode_reward = 0.0
        self._prev_health_diff = 0.0
        self._prev_dist_to_target = ARENA_SIZE
        self._prev_facing_target = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        """Execute one tick of the simulation.

        Args:
            action: Binary array of shape (28,).

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.current_tick += 1

        # Apply action masking (resolve conflicts)
        action = self._apply_action_mask(action)

        # Execute player actions
        self._execute_player_actions(action)

        # Execute enemy AI (simple rule-based for training)
        for enemy in self.enemies:
            if enemy.is_alive:
                self._enemy_ai(enemy)

        # Execute ally AI (simple follow + attack)
        for ally in self.allies:
            if ally.is_alive:
                self._ally_ai(ally)

        # Calculate reward BEFORE physics tick (which resets damage counters)
        reward = self._calculate_reward()

        # Tick physics for all agents (resets per-tick damage, applies drag/gravity)
        self.physics.tick_physics(self.player)
        for entity in self._all_entities:
            self.physics.tick_physics(entity)
        self.episode_reward += reward

        # Check termination
        terminated = not self.player.is_alive or all(not e.is_alive for e in self.enemies)
        truncated = self.current_tick >= self.episode_length

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _apply_action_mask(self, action: np.ndarray) -> np.ndarray:
        """Resolve conflicting actions."""
        action = action.copy()

        # Movement conflicts: forward/backward
        if action[ACT_FORWARD] and action[ACT_BACKWARD]:
            action[ACT_BACKWARD] = 0
        # Left/right
        if action[ACT_LEFT] and action[ACT_RIGHT]:
            action[ACT_RIGHT] = 0

        # Combo actions override basic movement
        if action[ACT_WTAP]:
            action[ACT_FORWARD] = 1
            action[ACT_ATTACK] = 1
        if action[ACT_STRAFE_LEFT_ATTACK]:
            action[ACT_LEFT] = 1
            action[ACT_ATTACK] = 1
        if action[ACT_STRAFE_RIGHT_ATTACK]:
            action[ACT_RIGHT] = 1
            action[ACT_ATTACK] = 1
        if action[ACT_BLOCK_HIT]:
            action[ACT_USE] = 1
            action[ACT_ATTACK] = 1

        # Mask sigils that are on cooldown or unavailable
        for i in range(4):
            bind = self.player.sigil_binds[i]
            if not bind.available or bind.cooldown_remaining > 0:
                action[ACT_SIGIL_0 + i] = 0

        return action

    def _execute_player_actions(self, action: np.ndarray):
        """Execute the player's action vector."""
        player = self.player

        # Sprint toggle
        if action[ACT_SPRINT]:
            player.is_sprinting = True
        if action[ACT_SNEAK]:
            player.is_sneaking = True
            player.is_sprinting = False
        if not action[ACT_SPRINT] and not action[ACT_SNEAK]:
            player.is_sneaking = False

        # W-tap: release sprint briefly for knockback reset
        if action[ACT_WTAP]:
            self.physics.apply_wtap(player)
            # Re-sprint next tick
            player.is_sprinting = True

        # Movement
        forward = 0.0
        strafe = 0.0
        if action[ACT_FORWARD]:
            forward += 1.0
        if action[ACT_BACKWARD]:
            forward -= 1.0
        if action[ACT_LEFT]:
            strafe -= 1.0
        if action[ACT_RIGHT]:
            strafe += 1.0

        self.physics.apply_movement(player, forward, strafe, bool(action[ACT_JUMP]))

        # Camera / facing
        turn_speed = np.pi / 18  # 10 degrees per tick
        if action[ACT_LOOK_LEFT]:
            self.physics.turn_agent(player, -turn_speed)
        if action[ACT_LOOK_RIGHT]:
            self.physics.turn_agent(player, turn_speed)
        # Look up/down affects Y aim but we're mostly 2D for v1

        # Auto-face target if no manual camera
        if not any(action[ACT_LOOK_LEFT:ACT_LOOK_RIGHT + 1]):
            target = self._get_target()
            if target is not None:
                self.physics.face_toward(player, target, max_turn=np.pi / 6)

        # Target selection
        if action[ACT_TARGET_NEAREST]:
            self._select_target_nearest()
        elif action[ACT_TARGET_LOWEST_HP]:
            self._select_target_lowest_hp()
        elif action[ACT_TARGET_HIGHEST_THREAT]:
            self._select_target_highest_threat()
        elif action[ACT_TARGET_CYCLE]:
            self._cycle_target()

        # Blocking
        player.is_blocking = bool(action[ACT_USE]) and not action[ACT_ATTACK]

        # Attack
        if action[ACT_ATTACK]:
            target = self._get_target()
            if target is not None:
                self.physics.try_attack(player, target, self.current_tick)

        # Sigil activation
        for i in range(4):
            if action[ACT_SIGIL_0 + i]:
                self.physics.activate_sigil(player, i, self._all_entities, self.current_tick)

    def _get_target(self) -> Agent | None:
        """Get the player's current target."""
        for entity in self._all_entities:
            if entity.agent_id == self.player.target_id and entity.is_alive:
                return entity
        # Fallback: nearest enemy
        return self._nearest_alive_enemy()

    def _nearest_alive_enemy(self) -> Agent | None:
        """Find the nearest alive enemy."""
        best = None
        best_dist = float("inf")
        for enemy in self.enemies:
            if enemy.is_alive:
                dist = self.physics.distance_between(self.player, enemy)
                if dist < best_dist:
                    best_dist = dist
                    best = enemy
        return best

    def _select_target_nearest(self):
        target = self._nearest_alive_enemy()
        if target:
            self.player.target_id = target.agent_id

    def _select_target_lowest_hp(self):
        best = None
        best_hp = float("inf")
        for enemy in self.enemies:
            if enemy.is_alive and enemy.health < best_hp:
                best_hp = enemy.health
                best = enemy
        if best:
            self.player.target_id = best.agent_id

    def _select_target_highest_threat(self):
        """Target the enemy closest and most dangerous."""
        best = None
        best_score = -float("inf")
        for enemy in self.enemies:
            if enemy.is_alive:
                dist = self.physics.distance_between(self.player, enemy)
                # Threat = inverse distance * damage potential
                weapon_dmg = WEAPON_STATS[enemy.weapon]["damage"]
                score = weapon_dmg / max(dist, 0.5)
                if score > best_score:
                    best_score = score
                    best = enemy
        if best:
            self.player.target_id = best.agent_id

    def _cycle_target(self):
        """Cycle to next alive enemy."""
        alive = [e for e in self.enemies if e.is_alive]
        if not alive:
            return
        current_ids = [e.agent_id for e in alive]
        if self.player.target_id in current_ids:
            idx = current_ids.index(self.player.target_id)
            next_idx = (idx + 1) % len(current_ids)
            self.player.target_id = current_ids[next_idx]
        else:
            self.player.target_id = current_ids[0]

    def _enemy_ai(self, enemy: Agent):
        """Simple rule-based enemy AI for training.

        Varies behavior to create diverse training scenarios.
        """
        dist = self.physics.distance_between(enemy, self.player)

        # Face player
        self.physics.face_toward(enemy, self.player, max_turn=np.pi / 8)

        # Move toward player if far, circle if close
        if dist > 4.0:
            # Approach
            forward = 1.0
            strafe = 0.0
            enemy.is_sprinting = True
        elif dist > 2.0:
            # Combat range - mix of approach and strafing
            forward = 0.3
            strafe = self.rng.choice([-1.0, 1.0]) * 0.7
            enemy.is_sprinting = self.rng.random() > 0.5
        else:
            # Very close - back up sometimes
            forward = self.rng.choice([-0.5, 0.3])
            strafe = self.rng.choice([-1.0, 1.0])
            enemy.is_sprinting = False

        self.physics.apply_movement(enemy, forward, strafe, jump=self.rng.random() > 0.85)

        # Attack when in range and cooldown ready
        if dist <= 3.0 and enemy.attack_cooldown == 0:
            self.physics.try_attack(enemy, self.player, self.current_tick)

    def _ally_ai(self, ally: Agent):
        """Simple ally AI - follow player and attack nearest enemy."""
        # Find nearest enemy
        target = None
        best_dist = float("inf")
        for enemy in self.enemies:
            if enemy.is_alive:
                d = self.physics.distance_between(ally, enemy)
                if d < best_dist:
                    best_dist = d
                    target = enemy

        if target is None:
            # Follow player
            self.physics.face_toward(ally, self.player, max_turn=np.pi / 8)
            dist_to_player = self.physics.distance_between(ally, self.player)
            if dist_to_player > 5.0:
                self.physics.apply_movement(ally, 1.0, 0.0, False)
                ally.is_sprinting = True
            return

        self.physics.face_toward(ally, target, max_turn=np.pi / 8)

        if best_dist > 3.0:
            self.physics.apply_movement(ally, 1.0, 0.0, False)
            ally.is_sprinting = True
        else:
            self.physics.apply_movement(ally, 0.3, self.rng.choice([-0.5, 0.5]), False)
            if ally.attack_cooldown == 0:
                self.physics.try_attack(ally, target, self.current_tick)

    def _calculate_reward(self) -> float:
        """Calculate reward for the current tick. Normalized to [-1, 1]."""
        reward = 0.0
        player = self.player

        # === Sparse event rewards ===
        # Kill
        if player.damage_dealt_this_tick > 0:
            for enemy in self.enemies:
                if not enemy.is_alive and enemy.last_hurt_tick == self.current_tick:
                    reward += 1.0  # kill reward

        # Death
        if not player.is_alive:
            reward -= 1.0

        # === Dense combat rewards ===
        # Damage dealt (normalized by max health)
        reward += (player.damage_dealt_this_tick / MAX_HEALTH) * 0.1

        # Damage taken (asymmetric - dealing > taking)
        reward -= (player.damage_taken_this_tick / MAX_HEALTH) * 0.05

        # === Potential-based shaping ===
        target = self._get_target()
        if target is not None and target.is_alive:
            # Health difference
            health_diff = (player.health - target.health) / MAX_HEALTH

            # Distance to target (optimal range: 2-3.5 blocks)
            dist = self.physics.distance_between(player, target)
            in_range = 1.0 if 2.0 <= dist <= 3.5 else 0.0

            # Facing target
            dx = target.x - player.x
            dz = target.z - player.z
            angle_to = np.arctan2(dz, dx)
            angle_diff = abs(player.facing_angle - angle_to)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            facing = 1.0 if angle_diff < np.pi / 6 else 0.0

            # Potential function
            phi_current = 0.3 * health_diff + 0.1 * in_range + 0.05 * facing
            phi_prev = 0.3 * self._prev_health_diff + 0.1 * (1.0 if self._prev_dist_to_target >= 2.0 and self._prev_dist_to_target <= 3.5 else 0.0) + 0.05 * self._prev_facing_target

            # Potential-based shaping: gamma * phi(s') - phi(s)
            reward += 0.99 * phi_current - phi_prev

            # Update previous values
            self._prev_health_diff = health_diff
            self._prev_dist_to_target = dist
            self._prev_facing_target = facing
        else:
            self._prev_health_diff = 0.0
            self._prev_dist_to_target = ARENA_SIZE
            self._prev_facing_target = 0.0

        return float(np.clip(reward, -2.0, 2.0))

    def _get_obs(self) -> dict:
        """Build the observation dictionary."""
        player = self.player

        # Self state (30 dims)
        self_state = player.get_self_state()

        # Entity features (up to MAX_ENTITIES x 20)
        entity_features = np.zeros((MAX_ENTITIES, ENTITY_FEATURE_DIM), dtype=np.float32)
        entity_mask = np.zeros(MAX_ENTITIES, dtype=np.float32)

        idx = 0
        for entity in self._all_entities:
            if idx >= MAX_ENTITIES:
                break
            if entity.is_alive:
                entity_features[idx] = entity.get_entity_features(player)
                entity_mask[idx] = 1.0
                idx += 1

        # Combat context (22 dims)
        combat_ctx = self._get_combat_context()

        # Sigil state (12 dims)
        sigil_state = player.get_sigil_state()

        # Environment state (8 dims)
        env_state = self._get_env_state()

        return {
            "self_state": self_state,
            "entity_features": entity_features,
            "entity_mask": entity_mask,
            "combat_ctx": combat_ctx,
            "sigil_state": sigil_state,
            "env_state": env_state,
        }

    def _get_combat_context(self) -> np.ndarray:
        """Build the 22-dim combat context vector."""
        ctx = np.zeros(COMBAT_CTX_DIM, dtype=np.float32)
        player = self.player

        # Weapon info (3)
        ctx[0] = float(player.weapon) / 2.0
        ctx[1] = 0.0  # consumable count (sim has none)
        ctx[2] = 1.0  # has weapon

        # Special items (5) - all zero in v1 sim
        # ctx[3:8] = 0.0

        # Selected slot (1)
        ctx[8] = 0.0

        # Recent combat (2)
        ctx[9] = player.damage_dealt_this_tick / MAX_HEALTH
        ctx[10] = player.damage_taken_this_tick / MAX_HEALTH

        # Hit accuracy (1) - simplified
        ctx[11] = min(1.0, player.combo_counter / 5.0)

        # Combo/flags (3)
        ctx[12] = min(1.0, player.combo_counter / 10.0)
        ctx[13] = 0.0  # knockback flag
        ctx[14] = 0.0  # critical flag

        # Movement (2)
        speed = np.sqrt(player.vx**2 + player.vz**2)
        ctx[15] = speed / SPRINT_SPEED
        ctx[16] = 0.0  # strafing detection

        # Ground + cover (5)
        ctx[17] = 1.0 if player.on_ground else 0.0
        # Cover in 4 directions - always 0 in flat arena
        # ctx[18:22] = 0.0

        return ctx

    def _get_env_state(self) -> np.ndarray:
        """Build the 8-dim environment state vector."""
        env = np.zeros(ENV_STATE_DIM, dtype=np.float32)

        # Distance from center (1)
        dist_center = np.sqrt(self.player.x**2 + self.player.z**2)
        env[0] = dist_center / ARENA_SIZE

        # Height advantage vs target (1)
        target = self._get_target()
        if target and target.is_alive:
            env[1] = (self.player.y - target.y) / 5.0
            # Can reach target (1)
            env[2] = 1.0 if self.physics.distance_between(self.player, target) <= 3.5 else 0.0
        # Ground quality (1) - always good in flat arena
        env[3] = 1.0

        # Obstacles (4) - none in flat arena
        # env[4:8] = 0.0

        return env

    def _get_info(self) -> dict:
        """Return info dict for logging."""
        alive_enemies = sum(1 for e in self.enemies if e.is_alive)
        target = self._get_target()
        target_dist = self.physics.distance_between(self.player, target) if target else -1.0

        return {
            "tick": self.current_tick,
            "player_health": self.player.health,
            "enemies_alive": alive_enemies,
            "kills": self.player.kills,
            "episode_reward": self.episode_reward,
            "target_distance": target_dist,
        }

    def get_action_mask(self) -> np.ndarray:
        """Return a mask of valid actions (1 = valid, 0 = invalid)."""
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)

        # Mask sigils that are unavailable or on cooldown
        for i in range(4):
            bind = self.player.sigil_binds[i]
            if not bind.available or bind.cooldown_remaining > 0:
                mask[ACT_SIGIL_0 + i] = 0.0

        # Can't consume if nothing to consume (v1: nothing)
        mask[ACT_CONSUME] = 0.0

        # Can't switch weapons (v1: only sword)
        mask[ACT_SWITCH_WEAPON] = 0.0
        mask[ACT_SWITCH_CONSUMABLE] = 0.0

        return mask
