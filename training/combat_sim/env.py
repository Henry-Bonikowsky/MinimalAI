"""Gymnasium environment for 1.8 PvP combat simulation.

Supports 1vN, NvN, and self-play with:
- 35-action multi-binary space
- Kit items (gaps, pots, pearls, totems)
- 12 sigil slots (4 weapon always + 4 armor randomized)
- Direct target selection by entity index
- 1.8 combat mechanics (no cooldown, blockhit, sprint reset)
"""

import logging
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .entities import (
    Agent, Alliance, WeaponType, WEAPON_STATS, MAX_HEALTH, ARENA_SIZE,
    SPRINT_SPEED, NUM_SIGIL_SLOTS, KIT_GOLDEN_APPLES, KIT_HEALTH_POTS,
    KIT_ENDER_PEARLS, KIT_TOTEMS,
)
from .physics import CombatPhysics
from .sigils import SigilEngine, randomize_loadout

logger = logging.getLogger("combat_sim")


# Action indices (35 total)
ACT_FORWARD = 0
ACT_BACKWARD = 1
ACT_STRAFE_LEFT = 2
ACT_STRAFE_RIGHT = 3
ACT_JUMP = 4
ACT_SNEAK = 5
ACT_SPRINT = 6
ACT_ATTACK = 7
ACT_BLOCK = 8          # 1.8 sword block
ACT_EAT_GAP = 9        # golden apple
ACT_THROW_POT = 10     # splash health pot
ACT_THROW_PEARL = 11   # ender pearl
ACT_SPRINT_RESET = 12  # toggle sprint for KB reset
ACT_SWAP_WEAPON = 13   # switch sword <-> axe

# Sigil ability activations (12 slots, only ABILITYs unmasked)
ACT_SIGIL_0 = 14
ACT_SIGIL_11 = 25

# Camera
ACT_LOOK_LEFT = 26
ACT_LOOK_RIGHT = 27
ACT_LOOK_UP = 28
ACT_LOOK_DOWN = 29

# Direct target selection
ACT_TARGET_0 = 30
ACT_TARGET_4 = 34

NUM_ACTIONS = 35
MAX_ENTITIES = 8
SELF_STATE_DIM = 38
ENTITY_FEATURE_DIM = 24
COMBAT_CTX_DIM = 26
SIGIL_STATE_DIM = NUM_SIGIL_SLOTS * 4  # 48
ENV_STATE_DIM = 8


class CombatEnv(gym.Env):
    """1.8 PvP combat environment with kit and sigils.

    Args:
        num_enemies: Number of enemy agents.
        num_allies: Number of ally agents (not counting the player).
        episode_length: Max ticks per episode.
        domain_randomization: Whether to randomize physics params.
        self_play: If True, opponent is controlled by an external network.
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 20}

    def __init__(
        self,
        num_enemies: int = 1,
        num_allies: int = 0,
        episode_length: int = 1800,
        domain_randomization: bool = True,
        self_play: bool = False,
        render_mode: str = "none",
        seed: int = None,
    ):
        super().__init__()

        self.num_enemies = num_enemies
        self.num_allies = num_allies
        self.episode_length = episode_length
        self.render_mode = render_mode
        self.self_play = self_play

        self.rng = np.random.default_rng(seed)
        self.physics = CombatPhysics(domain_randomization=domain_randomization, rng=self.rng)

        # Create agents
        self.player = Agent(agent_id=0, alliance=Alliance.ALLY)
        self.enemies: list[Agent] = []
        self.allies: list[Agent] = []
        self._all_entities: list[Agent] = []

        # Self-play opponent action storage
        self._opponent_actions: dict = {}  # agent_id -> action array

        # Action space: multi-binary
        self.action_space = spaces.MultiBinary(NUM_ACTIONS)

        # Observation space
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
        self._prev_health_diff = 0.0
        self._prev_dist_to_target = 0.0
        self._prev_facing_target = 0.0
        self._last_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._sigils_activated_this_tick = []

        # Sorted entities for target selection (updated each tick)
        self._sorted_enemies: list[Agent] = []

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
            self.physics.sigil_engine.rng = self.rng

        self.physics.randomize_params()
        self._create_entities()

        # Spawn player at center
        self.player.reset(x=0.0, z=0.0, facing=0.0)

        # Spawn enemies in a circle
        for i, enemy in enumerate(self.enemies):
            angle = 2 * np.pi * i / max(self.num_enemies, 1) + self.rng.uniform(-0.3, 0.3)
            dist = self.rng.uniform(4.0, 8.0)
            enemy.reset(
                x=np.cos(angle) * dist,
                z=np.sin(angle) * dist,
                facing=angle + np.pi,
            )
            # Scripted enemy: no healing items (AI must learn to kill first)
            if not self.self_play:
                enemy.kit.golden_apples = 0
                enemy.kit.health_pots = 0
                enemy.kit.ender_pearls = 0
                enemy.kit.totems = 0

        # Spawn allies near player
        for i, ally in enumerate(self.allies):
            angle = 2 * np.pi * i / max(self.num_allies, 1) + self.rng.uniform(-0.3, 0.3)
            dist = self.rng.uniform(2.0, 4.0)
            ally.reset(
                x=np.cos(angle) * dist,
                z=np.sin(angle) * dist,
                facing=angle + np.pi,
            )

        # Randomize sigil loadouts
        randomize_loadout(self.player, self.rng)
        for entity in self._all_entities:
            randomize_loadout(entity, self.rng)

        # Default target: nearest enemy
        self.player.target_id = self.enemies[0].agent_id if self.enemies else -1

        self.current_tick = 0
        self.episode_reward = 0.0
        self._prev_health_diff = 0.0
        self._prev_dist_to_target = ARENA_SIZE
        self._prev_facing_target = 0.0
        self._last_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._sigils_activated_this_tick = []
        self._opponent_actions = {}
        self._update_sorted_enemies()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def set_opponent_action(self, agent_id: int, action: np.ndarray):
        """Set action for a self-play opponent agent (called externally)."""
        self._opponent_actions[agent_id] = action

    def get_opponent_obs(self, agent_id: int) -> dict:
        """Get observation from an opponent's perspective (for self-play)."""
        agent = None
        for e in self.enemies:
            if e.agent_id == agent_id:
                agent = e
                break
        if agent is None:
            return None
        return self._get_obs_for(agent)

    def step(self, action: np.ndarray):
        """Execute one tick of the simulation."""
        self.current_tick += 1

        # Apply action masking
        action = self._apply_action_mask(action)
        self._last_action = action.copy()

        # Execute player actions
        self._execute_actions(self.player, action)

        # Execute enemy actions
        for enemy in self.enemies:
            if enemy.is_alive:
                if self.self_play and enemy.agent_id in self._opponent_actions:
                    opp_action = self._opponent_actions[enemy.agent_id]
                    opp_action = self._apply_action_mask_for(enemy, opp_action)
                    self._execute_actions(enemy, opp_action)
                else:
                    self._enemy_ai(enemy)

        # Execute ally AI
        for ally in self.allies:
            if ally.is_alive:
                self._ally_ai(ally)

        # Tick mummies for all agents
        all_agents = [self.player] + self._all_entities
        for agent in all_agents:
            self.physics.sigil_engine.tick_mummies(agent, all_agents, self.current_tick)

        # Calculate reward BEFORE physics tick
        reward = self._calculate_reward()

        # Log tick details BEFORE physics resets per-tick counters
        if logger.isEnabledFor(logging.DEBUG):
            p = self.player
            e = self.enemies[0] if self.enemies else None
            if p.damage_dealt_this_tick > 0 or p.damage_taken_this_tick > 0 or (e and e.damage_taken_this_tick > 0):
                logger.debug(
                    f"t={self.current_tick:4d} | "
                    f"P hp={p.health:.1f} abs={p.absorption:.1f} immune={p.hurt_immune_ticks} | "
                    f"E hp={e.health:.1f} immune={e.hurt_immune_ticks} | "
                    f"P dealt={p.damage_dealt_this_tick:.2f} took={p.damage_taken_this_tick:.2f} | "
                    f"E took={e.damage_taken_this_tick:.2f} | "
                    f"dist={self.physics.distance_between(p, e):.1f}"
                    if e else ""
                )

        # Tick physics for all agents
        self.physics.tick_physics(self.player)
        for entity in self._all_entities:
            self.physics.tick_physics(entity)
        self.episode_reward += reward

        # Update sorted enemies for targeting
        self._update_sorted_enemies()

        # Check termination
        terminated = not self.player.is_alive or all(not e.is_alive for e in self.enemies)
        truncated = self.current_tick >= self.episode_length

        if terminated or truncated:
            reason = "player_died" if not self.player.is_alive else (
                "all_enemies_dead" if all(not e.is_alive for e in self.enemies) else "truncated"
            )
            logger.info(
                f"Episode end: {reason} | tick={self.current_tick} | "
                f"P hp={self.player.health:.1f} kills={self.player.kills} | "
                f"reward={self.episode_reward:.3f} | "
                f"gaps_used={KIT_GOLDEN_APPLES-self.player.kit.golden_apples} "
                f"pots_used={KIT_HEALTH_POTS-self.player.kit.health_pots} "
                f"pearls_used={KIT_ENDER_PEARLS-self.player.kit.ender_pearls}"
            )

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _update_sorted_enemies(self):
        """Sort alive enemies by distance to player."""
        alive = [e for e in self.enemies if e.is_alive]
        alive.sort(key=lambda e: self.physics.distance_between(self.player, e))
        self._sorted_enemies = alive

    def _apply_action_mask(self, action: np.ndarray) -> np.ndarray:
        """Resolve conflicting actions for the player."""
        return self._apply_action_mask_for(self.player, action)

    def _apply_action_mask_for(self, agent: Agent, action: np.ndarray) -> np.ndarray:
        """Resolve conflicting actions for any agent."""
        action = action.copy()

        # Movement conflicts
        if action[ACT_FORWARD] and action[ACT_BACKWARD]:
            action[ACT_BACKWARD] = 0
        if action[ACT_STRAFE_LEFT] and action[ACT_STRAFE_RIGHT]:
            action[ACT_STRAFE_RIGHT] = 0

        # Weapon swap disabled - all abilities work on sword
        action[ACT_SWAP_WEAPON] = 0

        # Eating lock-in: once eating starts, can't cancel, can't attack/sprint
        if agent.kit.is_eating:
            action[ACT_EAT_GAP] = 0
            action[ACT_ATTACK] = 0
            action[ACT_SPRINT] = 0
            action[ACT_SPRINT_RESET] = 0
            action[ACT_BLOCK] = 0
            action[ACT_THROW_POT] = 0
            action[ACT_THROW_PEARL] = 0
            for i in range(NUM_SIGIL_SLOTS):
                action[ACT_SIGIL_0 + i] = 0

        # Kit masking
        if agent.kit.golden_apples <= 0 or agent.kit.is_eating:
            action[ACT_EAT_GAP] = 0
        if agent.kit.health_pots <= 0 or agent.kit.pot_cooldown > 0:
            action[ACT_THROW_POT] = 0
        if agent.kit.ender_pearls <= 0 or agent.kit.pearl_cooldown > 0:
            action[ACT_THROW_PEARL] = 0

        # Sprint reset only if sprinting
        if not agent.is_sprinting:
            action[ACT_SPRINT_RESET] = 0

        # Sigil masking
        sigil_mask = self.physics.sigil_engine.get_ability_mask(agent)
        for i in range(NUM_SIGIL_SLOTS):
            if sigil_mask[i] == 0.0:
                action[ACT_SIGIL_0 + i] = 0

        # Target masking
        for i in range(5):
            if i >= len(self._sorted_enemies):
                action[ACT_TARGET_0 + i] = 0

        # Stun prevents all actions except passive
        if agent.has_effect("stun"):
            action[:] = 0

        return action

    def _execute_actions(self, agent: Agent, action: np.ndarray):
        """Execute an agent's action vector."""
        # Sprint toggle
        if action[ACT_SPRINT]:
            agent.is_sprinting = True
        if action[ACT_SNEAK]:
            agent.is_sneaking = True
            agent.is_sprinting = False
        if not action[ACT_SPRINT] and not action[ACT_SNEAK]:
            agent.is_sneaking = False

        # Sprint reset (1.8 KB technique)
        if action[ACT_SPRINT_RESET]:
            self.physics.sprint_reset(agent)
            # Re-sprint immediately
            agent.is_sprinting = True

        # Swap weapon disabled - always sword
        # (all abilities work regardless of weapon)

        # Movement
        forward = 0.0
        strafe = 0.0
        if action[ACT_FORWARD]:
            forward += 1.0
        if action[ACT_BACKWARD]:
            forward -= 1.0
        if action[ACT_STRAFE_LEFT]:
            strafe -= 1.0
        if action[ACT_STRAFE_RIGHT]:
            strafe += 1.0

        self.physics.apply_movement(agent, forward, strafe, bool(action[ACT_JUMP]))

        # Camera
        turn_speed = np.pi / 18  # 10 degrees per tick
        if action[ACT_LOOK_LEFT]:
            self.physics.turn_agent(agent, -turn_speed)
        if action[ACT_LOOK_RIGHT]:
            self.physics.turn_agent(agent, turn_speed)

        # Auto-face target if no manual camera
        if not action[ACT_LOOK_LEFT] and not action[ACT_LOOK_RIGHT]:
            target = self._get_target_for(agent)
            if target is not None:
                self.physics.face_toward(agent, target, max_turn=np.pi / 6)

        # Target selection (direct index)
        for i in range(5):
            if action[ACT_TARGET_0 + i]:
                if agent is self.player and i < len(self._sorted_enemies):
                    agent.target_id = self._sorted_enemies[i].agent_id
                break

        # 1.8 blockhit: BLOCK + ATTACK = both active simultaneously
        agent.is_blocking = bool(action[ACT_BLOCK])

        # Eating (committed - once started, must finish the full 32 ticks)
        if action[ACT_EAT_GAP]:
            self.physics.start_eating(agent)
            logger.debug(f"t={self.current_tick} | Agent {agent.agent_id} started eating gap")

        # Health pot
        if action[ACT_THROW_POT]:
            self.physics.throw_pot(agent)
            logger.debug(f"t={self.current_tick} | Agent {agent.agent_id} threw pot (hp={agent.health:.1f})")

        # Ender pearl
        if action[ACT_THROW_PEARL]:
            self.physics.throw_pearl(agent)
            logger.debug(f"t={self.current_tick} | Agent {agent.agent_id} threw pearl")

        # Attack
        if action[ACT_ATTACK]:
            target = self._get_target_for(agent)
            if target is not None:
                self.physics.try_attack(agent, target, self.current_tick)

        # Sigil activations
        all_agents = [self.player] + self._all_entities
        sigils_activated = []
        for i in range(NUM_SIGIL_SLOTS):
            if action[ACT_SIGIL_0 + i]:
                success = self.physics.sigil_engine.activate_ability(
                    agent, i, all_agents, self.current_tick
                )
                if success:
                    sigils_activated.append(i)
        if agent is self.player:
            self._sigils_activated_this_tick = sigils_activated

    def _get_target_for(self, agent: Agent) -> Agent | None:
        """Get an agent's current target."""
        for entity in self._all_entities + [self.player]:
            if entity.agent_id == agent.target_id and entity.is_alive:
                return entity
        # Fallback: nearest alive enemy (relative to agent)
        if agent.alliance == Alliance.ALLY:
            return self._nearest_alive(agent, self.enemies)
        else:
            targets = [self.player] + self.allies
            return self._nearest_alive(agent, targets)

    def _nearest_alive(self, agent: Agent, candidates: list) -> Agent | None:
        best = None
        best_dist = float("inf")
        for c in candidates:
            if c.is_alive and c.agent_id != agent.agent_id:
                dist = self.physics.distance_between(agent, c)
                if dist < best_dist:
                    best_dist = dist
                    best = c
        return best

    def _enemy_ai(self, enemy: Agent):
        """Rule-based enemy AI for training (when not self-play).

        Deliberately mediocre opponent:
        - Low CPS (~6-8 clicks/sec)
        - Slow to engage, sometimes wanders
        - Rarely blocks or uses kit items
        - Doesn't sprint-reset or combo
        """
        dist = self.physics.distance_between(enemy, self.player)

        # Slow aim tracking (worse than perfect)
        self.physics.face_toward(enemy, self.player, max_turn=np.pi / 12)

        # Movement: hesitant, doesn't always rush in
        if dist > 6.0:
            # Sometimes just wanders instead of chasing
            if self.rng.random() < 0.7:
                forward = 0.8
                strafe = self.rng.choice([-0.3, 0.3])
                enemy.is_sprinting = self.rng.random() > 0.4
            else:
                forward = 0.0
                strafe = self.rng.choice([-1.0, 1.0])
                enemy.is_sprinting = False
        elif dist > 3.0:
            forward = 0.5
            strafe = self.rng.choice([-1.0, 1.0]) * 0.5
            enemy.is_sprinting = self.rng.random() > 0.6
        else:
            # In melee range: strafe around, sometimes back up
            forward = self.rng.choice([-0.5, 0.0, 0.3])
            strafe = self.rng.choice([-1.0, 1.0])
            enemy.is_sprinting = False

        self.physics.apply_movement(enemy, forward, strafe, jump=self.rng.random() > 0.92)

        # ~10 CPS but capped by 5-tick hurt immunity anyway
        if dist <= 3.0 and enemy.attack_tick_cooldown == 0 and self.rng.random() < 0.5:
            self.physics.try_attack(enemy, self.player, self.current_tick)

        # Rarely blockhits
        if dist <= 3.0 and enemy.weapon == WeaponType.SWORD:
            enemy.is_blocking = self.rng.random() < 0.05

        # Only use healing if enemy has items (self-play mode)
        if enemy.kit.golden_apples > 0 and enemy.health < 8.0 and self.rng.random() > 0.97:
            self.physics.start_eating(enemy)
        if enemy.kit.health_pots > 0 and enemy.health < 5.0 and self.rng.random() > 0.85:
            self.physics.throw_pot(enemy)

    def _ally_ai(self, ally: Agent):
        """Simple ally AI - follow player, attack nearest enemy."""
        target = self._nearest_alive(ally, self.enemies)
        if target is None:
            self.physics.face_toward(ally, self.player, max_turn=np.pi / 8)
            if self.physics.distance_between(ally, self.player) > 5.0:
                self.physics.apply_movement(ally, 1.0, 0.0, False)
                ally.is_sprinting = True
            return

        self.physics.face_toward(ally, target, max_turn=np.pi / 8)
        dist = self.physics.distance_between(ally, target)
        if dist > 3.0:
            self.physics.apply_movement(ally, 1.0, 0.0, False)
            ally.is_sprinting = True
        else:
            self.physics.apply_movement(ally, 0.3, self.rng.choice([-0.5, 0.5]), False)
            if ally.attack_tick_cooldown == 0:
                self.physics.try_attack(ally, target, self.current_tick)

    def _calculate_reward(self) -> float:
        """Calculate reward for the current tick.

        Reward design:
        - Sparse: kill (+5), death (-3)
        - Dense: damage dealt/taken
        - Action quality: penalize whiffs, wasted pots, reward smart ability use
        - Potential shaping: approach + health advantage
        """
        reward = 0.0
        player = self.player
        action = self._last_action
        target = self._get_target_for(player)

        # ── Sparse events ──
        if player.damage_dealt_this_tick > 0:
            for enemy in self.enemies:
                if not enemy.is_alive and enemy.last_hurt_tick == self.current_tick:
                    reward += 5.0

        if not player.is_alive:
            reward -= 3.0

        # ── Dense: damage dealt/taken ──
        reward += (player.damage_dealt_this_tick / MAX_HEALTH) * 0.5
        reward -= (player.damage_taken_this_tick / MAX_HEALTH) * 0.1

        # ── Action quality: attack timing ──
        if action[ACT_ATTACK] and target is not None and target.is_alive:
            dist = self.physics.distance_between(player, target)
            # Whiff: swinging out of reach
            if dist > 3.5:
                reward -= 0.05
            # Wasted hit: target has hurt immunity (i-frames)
            elif target.hurt_immune_ticks > 0:
                reward -= 0.03

        # ── Action quality: pot timing ──
        if action[ACT_THROW_POT]:
            if player.health > 15.0:  # > 75% HP = wasteful
                reward -= 0.3
            elif player.health < 10.0:  # < 50% HP = smart
                reward += 0.2

        # ── Action quality: sigil usage ──
        for slot_idx in self._sigils_activated_this_tick:
            slot = player.sigil_slots[slot_idx]
            has_nearby_enemy = False
            if target is not None and target.is_alive:
                dist_to_target = self.physics.distance_between(player, target)
                has_nearby_enemy = dist_to_target < 8.0

            # Defensive abilities (bolster, quicksand, niles_grace, kings_brace burst)
            if slot.sigil_type in ("royal_bolster", "niles_grace"):
                if player.health < 10.0:  # used when actually hurt
                    reward += 0.3
                elif player.health > 18.0:  # wasted at full HP
                    reward -= 0.1
            elif slot.sigil_type == "quick_sand":
                if has_nearby_enemy:
                    reward += 0.2  # good: enemies nearby to slow
                else:
                    reward -= 0.1  # wasted: nobody to affect
            # Offensive abilities (cleopatra, royal_guard)
            elif slot.sigil_type in ("cleopatra", "royal_guard"):
                if has_nearby_enemy:
                    reward += 0.2
                else:
                    reward -= 0.1
            elif slot.sigil_type == "kings_brace":
                reward += 0.2  # burst is always situationally good (requires 100 charges)

        # ── Opportunity cost: not using defensive sigils when hurt ──
        if player.health < 6.0:  # < 30% HP
            for slot in player.sigil_slots:
                if (slot.equipped and slot.cooldown_remaining == 0 and
                        slot.sigil_type in ("royal_bolster", "niles_grace") and
                        "ability" in slot.activation_type):
                    reward -= 0.02  # accumulates per tick until they use it

        # ── Potential-based shaping ──
        if target is not None and target.is_alive:
            dist = self.physics.distance_between(player, target)
            dist_potential = max(0.0, 1.0 - dist / ARENA_SIZE)
            in_range = 1.0 if 2.0 <= dist <= 3.5 else 0.0

            dx = target.x - player.x
            dz = target.z - player.z
            angle_to = np.arctan2(dz, dx)
            angle_diff = abs(player.facing_angle - angle_to)
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            facing = 1.0 if angle_diff < np.pi / 6 else 0.0

            health_diff = (player.health - target.health) / MAX_HEALTH

            phi_current = (
                0.2 * health_diff +
                0.15 * dist_potential +
                0.1 * in_range +
                0.05 * facing
            )
            prev_dist_potential = max(0.0, 1.0 - self._prev_dist_to_target / ARENA_SIZE)
            prev_in_range = 1.0 if 2.0 <= self._prev_dist_to_target <= 3.5 else 0.0
            phi_prev = (
                0.2 * self._prev_health_diff +
                0.15 * prev_dist_potential +
                0.1 * prev_in_range +
                0.05 * self._prev_facing_target
            )

            reward += 0.99 * phi_current - phi_prev

            self._prev_health_diff = health_diff
            self._prev_dist_to_target = dist
            self._prev_facing_target = facing
        else:
            self._prev_health_diff = 0.0
            self._prev_dist_to_target = ARENA_SIZE
            self._prev_facing_target = 0.0

        return float(np.clip(reward, -5.0, 5.0))

    def _get_obs(self) -> dict:
        """Build observation from player's perspective."""
        return self._get_obs_for(self.player)

    def _get_obs_for(self, agent: Agent) -> dict:
        """Build observation from any agent's perspective."""
        self_state = agent.get_self_state()

        entity_features = np.zeros((MAX_ENTITIES, ENTITY_FEATURE_DIM), dtype=np.float32)
        entity_mask = np.zeros(MAX_ENTITIES, dtype=np.float32)

        idx = 0
        all_others = [e for e in [self.player] + self._all_entities if e.agent_id != agent.agent_id]
        for entity in all_others:
            if idx >= MAX_ENTITIES:
                break
            if entity.is_alive:
                entity_features[idx] = entity.get_entity_features(agent)
                entity_mask[idx] = 1.0
                idx += 1

        combat_ctx = self._get_combat_context(agent)

        # Update sigil state with current tick info
        sigil_state = agent.get_sigil_state()
        for i, slot in enumerate(agent.sigil_slots):
            # Update last_used normalized time
            if slot.last_used_tick >= 0:
                ticks_since = self.current_tick - slot.last_used_tick
                sigil_state[i * 4 + 3] = min(1.0, ticks_since / 600.0)
            else:
                sigil_state[i * 4 + 3] = 1.0

        env_state = self._get_env_state(agent)

        return {
            "self_state": self_state,
            "entity_features": entity_features,
            "entity_mask": entity_mask,
            "combat_ctx": combat_ctx,
            "sigil_state": sigil_state,
            "env_state": env_state,
        }

    def _get_combat_context(self, agent: Agent) -> np.ndarray:
        """Build 26-dim combat context vector."""
        ctx = np.zeros(COMBAT_CTX_DIM, dtype=np.float32)

        # Weapon info (3)
        ctx[0] = float(agent.weapon) / 2.0
        ctx[1] = 1.0  # has weapon
        ctx[2] = 1.0 if agent.weapon == WeaponType.SWORD else 0.0  # can block

        # Kit counts (4)
        ctx[3] = agent.kit.golden_apples / max(KIT_GOLDEN_APPLES, 1)
        ctx[4] = agent.kit.health_pots / max(KIT_HEALTH_POTS, 1)
        ctx[5] = agent.kit.ender_pearls / max(KIT_ENDER_PEARLS, 1)
        ctx[6] = agent.kit.totems / max(KIT_TOTEMS, 1)

        # Kit state (2)
        ctx[7] = 1.0 if agent.kit.is_eating else 0.0
        ctx[8] = agent.kit.pearl_cooldown / 20.0

        # Recent combat (2)
        ctx[9] = agent.damage_dealt_this_tick / MAX_HEALTH
        ctx[10] = agent.damage_taken_this_tick / MAX_HEALTH

        # Combo (2)
        ctx[11] = min(1.0, agent.combo_counter / 10.0)
        ctx[12] = agent.cps / 20.0

        # Movement (2)
        speed = np.sqrt(agent.vx**2 + agent.vz**2)
        ctx[13] = speed / SPRINT_SPEED if SPRINT_SPEED > 0 else 0.0
        ctx[14] = 1.0 if agent.on_ground else 0.0

        # Block state (2)
        ctx[15] = 1.0 if agent.is_blocking else 0.0
        ctx[16] = 1.0 if agent.sprint_reset_ready else 0.0

        # Status effect flags (6)
        ctx[17] = 1.0 if agent.has_effect("speed") else 0.0
        ctx[18] = 1.0 if agent.has_effect("regen") else 0.0
        ctx[19] = 1.0 if agent.has_effect("resistance") else 0.0
        ctx[20] = 1.0 if agent.has_effect("stun") else 0.0
        ctx[21] = 1.0 if agent.has_effect("wither") else 0.0
        ctx[22] = 1.0 if agent.has_effect("slowness") else 0.0

        # Sigil combat state (3)
        ctx[23] = agent.sigil_damage_amp
        ctx[24] = agent.sigil_damage_reduction
        ctx[25] = min(1.0, agent.invuln_hits / 3.0)

        return ctx

    def _get_env_state(self, agent: Agent = None) -> np.ndarray:
        """Build 8-dim environment state."""
        if agent is None:
            agent = self.player
        env = np.zeros(ENV_STATE_DIM, dtype=np.float32)

        dist_center = np.sqrt(agent.x**2 + agent.z**2)
        env[0] = dist_center / ARENA_SIZE

        target = self._get_target_for(agent)
        if target and target.is_alive:
            env[1] = (agent.y - target.y) / 5.0
            env[2] = 1.0 if self.physics.distance_between(agent, target) <= 3.5 else 0.0
        env[3] = 1.0  # ground quality (flat arena)

        # Alive counts
        alive_enemies = sum(1 for e in self.enemies if e.is_alive)
        alive_allies = sum(1 for a in self.allies if a.is_alive) + (1 if self.player.is_alive else 0)
        env[4] = alive_enemies / max(self.num_enemies, 1)
        env[5] = alive_allies / max(self.num_allies + 1, 1)

        # Episode progress
        env[6] = self.current_tick / self.episode_length

        return env

    def _get_info(self) -> dict:
        alive_enemies = sum(1 for e in self.enemies if e.is_alive)
        target = self._get_target_for(self.player)
        target_dist = self.physics.distance_between(self.player, target) if target else -1.0

        return {
            "tick": self.current_tick,
            "player_health": self.player.health,
            "enemies_alive": alive_enemies,
            "kills": self.player.kills,
            "episode_reward": self.episode_reward,
            "target_distance": target_dist,
            "gaps_remaining": self.player.kit.golden_apples,
            "pots_remaining": self.player.kit.health_pots,
            "pearls_remaining": self.player.kit.ender_pearls,
        }

    def get_action_mask(self, agent: Agent = None) -> np.ndarray:
        """Return mask of valid actions (1 = valid, 0 = invalid)."""
        if agent is None:
            agent = self.player
        mask = np.ones(NUM_ACTIONS, dtype=np.float32)

        # Weapon swap always disabled
        mask[ACT_SWAP_WEAPON] = 0.0

        # Eating lock-in: while eating, most actions masked
        if agent.kit.is_eating:
            mask[ACT_EAT_GAP] = 0.0
            mask[ACT_ATTACK] = 0.0
            mask[ACT_SPRINT] = 0.0
            mask[ACT_SPRINT_RESET] = 0.0
            mask[ACT_BLOCK] = 0.0
            mask[ACT_THROW_POT] = 0.0
            mask[ACT_THROW_PEARL] = 0.0
            for i in range(NUM_SIGIL_SLOTS):
                mask[ACT_SIGIL_0 + i] = 0.0
            # Can still move (slowly) and look, but that's it
            return mask

        # Kit masking
        if agent.kit.golden_apples <= 0:
            mask[ACT_EAT_GAP] = 0.0
        if agent.kit.health_pots <= 0 or agent.kit.pot_cooldown > 0:
            mask[ACT_THROW_POT] = 0.0
        if agent.kit.ender_pearls <= 0 or agent.kit.pearl_cooldown > 0:
            mask[ACT_THROW_PEARL] = 0.0

        # Sprint reset only if sprinting
        if not agent.is_sprinting:
            mask[ACT_SPRINT_RESET] = 0.0

        # Sigil masking
        sigil_mask = self.physics.sigil_engine.get_ability_mask(agent)
        for i in range(NUM_SIGIL_SLOTS):
            mask[ACT_SIGIL_0 + i] = sigil_mask[i]

        # Target masking
        alive_enemies = len(self._sorted_enemies)
        for i in range(5):
            if i >= alive_enemies:
                mask[ACT_TARGET_0 + i] = 0.0

        # Stun: mask everything
        if agent.has_effect("stun"):
            mask[:] = 0.0

        return mask
