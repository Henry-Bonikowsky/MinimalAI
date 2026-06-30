"""Vectorized sigil timing simulator for batch RL training.

Models the exact bot loadout:
  ACTIVES (neural net controlled):
    0. King's Brace   — 100 charges → Resistance III (80% DR) for 10s
    1. Cleopatra       — strip target buffs + 20% dmg amp for 5s
    2. Quick Sand      — slow aura + pull on hit for 6s
    3. Nile's Grace    — regen III + Resistance III for 10s

  PASSIVES (automatic, modeled as background):
    - Ancient Crown: immunity to debuffs (blocks opponent's Cleopatra)
    - Divine Intervention: on-block → 3 invuln hits + regen
    - Lifeforce: +1 HP every 100 ticks
    - Extra Padding: +4 max HP (24 HP total)
    - Iron Fist: +3 base damage
    - Rocket Boots: slight DPS increase from speed
    - Well Fed: on-hit food restore (modeled as slight sustain)
    - Meal Planning: auto-feed (modeled as slight sustain)

Runs N parallel fights using numpy. Combat is stochastic hits
modified by active buffs/debuffs from sigils.
"""

import numpy as np


# Active ability indices (neural net output 0-3)
SIG_KINGS_BRACE = 0    # 100 charges → 80% DR for 200 ticks
SIG_CLEOPATRA = 1      # strip buffs + 20% dmg amp for 100 ticks
SIG_QUICK_SAND = 2     # slow + pull for 120 ticks
SIG_NILES_GRACE = 3    # regen + 40% DR for 200 ticks

NUM_SIGIL_ACTIONS = 4
OBS_DIM = 16  # reduced from 20 to match 4 abilities

# Cooldowns (ticks)
COOLDOWNS = np.array([
    600,   # kings_brace: 30s (powerful)
    180,   # cleopatra: 9s (was 3s per config, but that seems too fast)
    140,   # quick_sand: 7s
    180,   # niles_grace: 9s
], dtype=np.int32)

# King's Brace
KINGS_BRACE_CHARGE_REQ = 30  # lowered from 100 for sim pacing
KINGS_BRACE_DURATION = 200
KINGS_BRACE_DR = 0.80

# Cleopatra
CLEOPATRA_DURATION = 100
CLEOPATRA_DMG_AMP = 1.20
# Note: Ancient Crown gives immunity to Cleopatra debuff — modeled below

# Quick Sand
QUICKSAND_DURATION = 120
QUICKSAND_SLOW_DPS_BONUS = 1.30  # slow = more hits land = ~30% DPS increase
QUICKSAND_PULL_BONUS = 0.15  # per-tick damage from pull keeping target in range

# Nile's Grace
NILES_GRACE_DURATION = 200
NILES_GRACE_REGEN = 0.0   # regen modeled via slower death, not HP restore (burst combos kill through regen on server)
NILES_GRACE_DR = 0.30     # reduced from 0.40 — stacking with Brace was too strong

# Divine Intervention (passive, on-block chance)
DIVINE_INVULN_CHANCE = 0.03  # per-tick chance when blocking
DIVINE_INVULN_HITS = 3
DIVINE_REGEN_TICKS = 100

# Passive stat bonuses (from regulars)
EXTRA_PADDING_HP = 4.0  # +4 max HP → 24 total
IRON_FIST_DMG = 1.0     # +1 damage per hit from Strength II
LIFEFORCE_HEAL = 0.0    # background sustain negligible vs combo burst

# Combat constants
# Server reality: rule engine lands ~2-3 hits/sec with W-tap combos
BASE_HIT_CHANCE = 0.15   # ~3.0 hits/sec at 20 tps (rule engine with combos)
BASE_DAMAGE = 3.0 + IRON_FIST_DMG  # 4.0 per hit (server damage + strength)
MAX_HEALTH = 20.0 + EXTRA_PADDING_HP  # 24 HP
CRIT_CHANCE = 0.15
CRIT_MULTIPLIER = 1.5


class SigilSim:
    """Vectorized sigil timing simulator with 4 active abilities."""

    def __init__(self, n_envs: int, episode_length: int = 1200, seed: int = 0):
        self.n = n_envs
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

        # Player state
        self.a_health = np.full(n_envs, MAX_HEALTH)
        self.a_absorption = np.zeros(n_envs)
        self.b_health = np.full(n_envs, MAX_HEALTH)
        self.b_absorption = np.zeros(n_envs)

        # Sigil cooldowns (0 = ready)
        self.a_cooldowns = np.zeros((n_envs, NUM_SIGIL_ACTIONS), dtype=np.int32)
        self.b_cooldowns = np.zeros((n_envs, NUM_SIGIL_ACTIONS), dtype=np.int32)

        # Active buff/debuff timers (0 = inactive)
        # King's Brace
        self.a_brace_timer = np.zeros(n_envs, dtype=np.int32)
        self.b_brace_timer = np.zeros(n_envs, dtype=np.int32)
        self.a_brace_charges = np.full(n_envs, 15, dtype=np.int32)
        self.b_brace_charges = np.full(n_envs, 15, dtype=np.int32)

        # Cleopatra (applied TO target)
        self.a_cleo_on_target = np.zeros(n_envs, dtype=np.int32)  # A applied cleo to B
        self.b_cleo_on_target = np.zeros(n_envs, dtype=np.int32)  # B applied cleo to A

        # Quick Sand
        self.a_sand_timer = np.zeros(n_envs, dtype=np.int32)
        self.b_sand_timer = np.zeros(n_envs, dtype=np.int32)

        # Nile's Grace
        self.a_grace_timer = np.zeros(n_envs, dtype=np.int32)
        self.b_grace_timer = np.zeros(n_envs, dtype=np.int32)

        # Divine Intervention invuln hits remaining
        self.a_divine_hits = np.zeros(n_envs, dtype=np.int32)
        self.b_divine_hits = np.zeros(n_envs, dtype=np.int32)
        self.a_divine_regen = np.zeros(n_envs, dtype=np.int32)
        self.b_divine_regen = np.zeros(n_envs, dtype=np.int32)

        # Episode tracking
        self.tick = np.zeros(n_envs, dtype=np.int32)
        self.done = np.zeros(n_envs, dtype=bool)

        # Stats
        self.a_kills = 0
        self.b_kills = 0
        self.draws = 0
        self.episodes_completed = 0
        self.a_ability_uses = np.zeros(NUM_SIGIL_ACTIONS, dtype=np.int64)
        self.b_ability_uses = np.zeros(NUM_SIGIL_ACTIONS, dtype=np.int64)
        self.a_total_dmg_dealt = 0.0
        self.b_total_dmg_dealt = 0.0
        self.a_hp_at_win = []
        self.b_hp_at_win = []
        self.episode_lengths = []

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.n, dtype=bool)

        self.a_health[mask] = MAX_HEALTH
        self.a_absorption[mask] = 0.0
        self.b_health[mask] = MAX_HEALTH
        self.b_absorption[mask] = 0.0

        self.a_cooldowns[mask] = 0
        self.b_cooldowns[mask] = 0

        self.a_brace_timer[mask] = 0
        self.b_brace_timer[mask] = 0
        self.a_brace_charges[mask] = 15
        self.b_brace_charges[mask] = 15

        self.a_cleo_on_target[mask] = 0
        self.b_cleo_on_target[mask] = 0
        self.a_sand_timer[mask] = 0
        self.b_sand_timer[mask] = 0
        self.a_grace_timer[mask] = 0
        self.b_grace_timer[mask] = 0
        self.a_divine_hits[mask] = 0
        self.b_divine_hits[mask] = 0
        self.a_divine_regen[mask] = 0
        self.b_divine_regen[mask] = 0

        self.tick[mask] = 0
        self.done[mask] = False

        return self._get_obs(), self._get_mask()

    def step(self, a_actions, b_actions=None):
        if b_actions is None:
            b_actions = self._rule_opponent()

        # 1. Activate abilities
        self._process_activations(a_actions, 'a')
        self._process_activations(b_actions, 'b')

        # 2. Combat tick
        a_dmg, b_dmg = self._combat_tick()

        # 3. Passives (regen, divine regen, lifeforce)
        self._apply_passives()

        # 4. Tick timers
        self._tick_timers()

        # 5. Track damage
        self.a_total_dmg_dealt += a_dmg.sum()
        self.b_total_dmg_dealt += b_dmg.sum()

        # 6. Check deaths
        self.tick += 1
        a_dead = self.a_health <= 0
        b_dead = self.b_health <= 0
        timeout = self.tick >= self.episode_length

        # 7. Rewards
        rewards = self._compute_rewards(a_actions, a_dmg, b_dmg, a_dead, b_dead, timeout)

        self.done = a_dead | b_dead | timeout

        # 8. Record outcomes and reset
        done_mask = self.done.copy()
        if done_mask.any():
            a_won = b_dead & ~a_dead & done_mask
            b_won = a_dead & ~b_dead & done_mask
            both_dead = a_dead & b_dead & done_mask
            timed_out = timeout & ~a_dead & ~b_dead & done_mask

            self.a_kills += a_won.sum()
            self.b_kills += b_won.sum()
            self.draws += (both_dead | timed_out).sum()
            self.episodes_completed += done_mask.sum()

            if a_won.any():
                self.a_hp_at_win.extend((self.a_health[a_won] + self.a_absorption[a_won]).tolist())
            if b_won.any():
                self.b_hp_at_win.extend((self.b_health[b_won] + self.b_absorption[b_won]).tolist())
            self.episode_lengths.extend(self.tick[done_mask].tolist())
            self.reset(done_mask)

        return self._get_obs(), self._get_mask(), rewards, done_mask, {}

    def _process_activations(self, actions, player):
        cooldowns = self.a_cooldowns if player == 'a' else self.b_cooldowns

        for sig_idx in range(NUM_SIGIL_ACTIONS):
            can_activate = (actions[:, sig_idx] == 1) & (cooldowns[:, sig_idx] <= 0)

            if sig_idx == SIG_KINGS_BRACE:
                # Also need charges
                charges = self.a_brace_charges if player == 'a' else self.b_brace_charges
                can_activate = can_activate & (charges >= KINGS_BRACE_CHARGE_REQ)
                if not can_activate.any():
                    continue
                timer = self.a_brace_timer if player == 'a' else self.b_brace_timer
                timer[can_activate] = KINGS_BRACE_DURATION
                charges[can_activate] = 0
            elif not can_activate.any():
                continue
            elif sig_idx == SIG_CLEOPATRA:
                # Strip target buffs + apply mark
                cleo_timer = self.b_cleo_on_target if player == 'a' else self.a_cleo_on_target
                cleo_timer[can_activate] = CLEOPATRA_DURATION
                # Strip target's defensive buffs (but Ancient Crown blocks this!)
                # Both players have Ancient Crown → Cleopatra's debuff is partially resisted
                # Model as: 50% chance the debuff sticks (Crown provides some immunity)
                resisted = self.rng.random(self.n) < 0.5
                cleo_timer[can_activate & resisted] = 0
                # Still strip buffs even if debuff is resisted
                if player == 'a':
                    self.b_brace_timer[can_activate] = 0
                    self.b_grace_timer[can_activate] = 0
                else:
                    self.a_brace_timer[can_activate] = 0
                    self.a_grace_timer[can_activate] = 0
            elif sig_idx == SIG_QUICK_SAND:
                timer = self.a_sand_timer if player == 'a' else self.b_sand_timer
                timer[can_activate] = QUICKSAND_DURATION
            elif sig_idx == SIG_NILES_GRACE:
                timer = self.a_grace_timer if player == 'a' else self.b_grace_timer
                timer[can_activate] = NILES_GRACE_DURATION

            # Track uses and start cooldown
            if player == 'a':
                self.a_ability_uses[sig_idx] += can_activate.sum()
            else:
                self.b_ability_uses[sig_idx] += can_activate.sum()
            cooldowns[can_activate, sig_idx] = COOLDOWNS[sig_idx]

    def _combat_tick(self):
        # Roll hits
        a_hits = self.rng.random(self.n) < BASE_HIT_CHANCE
        b_hits = self.rng.random(self.n) < BASE_HIT_CHANCE

        # Base damage
        a_dmg = np.where(a_hits, BASE_DAMAGE, 0.0)
        b_dmg = np.where(b_hits, BASE_DAMAGE, 0.0)

        # Crits
        a_crit = a_hits & (self.rng.random(self.n) < CRIT_CHANCE)
        b_crit = b_hits & (self.rng.random(self.n) < CRIT_CHANCE)
        a_dmg[a_crit] *= CRIT_MULTIPLIER
        b_dmg[b_crit] *= CRIT_MULTIPLIER

        # Quick Sand: target takes more hits (slow = can't escape)
        a_dmg *= np.where(self.a_sand_timer > 0, QUICKSAND_SLOW_DPS_BONUS, 1.0)
        b_dmg *= np.where(self.b_sand_timer > 0, QUICKSAND_SLOW_DPS_BONUS, 1.0)

        # Quick Sand: pull damage
        a_dmg += np.where(self.a_sand_timer > 0, QUICKSAND_PULL_BONUS, 0.0)
        b_dmg += np.where(self.b_sand_timer > 0, QUICKSAND_PULL_BONUS, 0.0)

        # Cleopatra mark on target: +20% damage taken
        a_dmg *= np.where(self.a_cleo_on_target > 0, CLEOPATRA_DMG_AMP, 1.0)  # B takes more from A
        b_dmg *= np.where(self.b_cleo_on_target > 0, CLEOPATRA_DMG_AMP, 1.0)  # A takes more from B

        # Damage reduction from defender
        # King's Brace: 80% DR
        b_dr = np.where(self.b_brace_timer > 0, 1.0 - KINGS_BRACE_DR, 1.0)
        a_dr = np.where(self.a_brace_timer > 0, 1.0 - KINGS_BRACE_DR, 1.0)

        # Nile's Grace: 40% DR
        b_dr *= np.where(self.b_grace_timer > 0, 1.0 - NILES_GRACE_DR, 1.0)
        a_dr *= np.where(self.a_grace_timer > 0, 1.0 - NILES_GRACE_DR, 1.0)

        # Divine Intervention: invuln hits (negate damage entirely)
        a_blocked = a_hits & (self.b_divine_hits > 0)
        b_blocked = b_hits & (self.a_divine_hits > 0)
        a_dmg[a_blocked] = 0  # A's hit blocked by B's divine
        b_dmg[b_blocked] = 0  # B's hit blocked by A's divine
        self.b_divine_hits[a_blocked] = np.maximum(self.b_divine_hits[a_blocked] - 1, 0)
        self.a_divine_hits[b_blocked] = np.maximum(self.a_divine_hits[b_blocked] - 1, 0)

        # Apply DR
        a_dmg_to_b = a_dmg * b_dr
        b_dmg_to_a = b_dmg * a_dr

        # Apply damage
        self._apply_damage(a_dmg_to_b, 'b')
        self._apply_damage(b_dmg_to_a, 'a')

        # Build brace charges from damage
        self.a_brace_charges += np.where(b_dmg_to_a > 0, 1, 0).astype(np.int32)
        self.b_brace_charges += np.where(a_dmg_to_b > 0, 1, 0).astype(np.int32)
        self.a_brace_charges = np.minimum(self.a_brace_charges, 100)
        self.b_brace_charges = np.minimum(self.b_brace_charges, 100)

        # Divine Intervention: chance to trigger on being hit (simulates blocking)
        a_divine_trigger = (b_dmg_to_a > 0) & (self.a_divine_hits <= 0) & (self.rng.random(self.n) < DIVINE_INVULN_CHANCE)
        b_divine_trigger = (a_dmg_to_b > 0) & (self.b_divine_hits <= 0) & (self.rng.random(self.n) < DIVINE_INVULN_CHANCE)
        self.a_divine_hits[a_divine_trigger] = DIVINE_INVULN_HITS
        self.b_divine_hits[b_divine_trigger] = DIVINE_INVULN_HITS
        self.a_divine_regen[a_divine_trigger] = DIVINE_REGEN_TICKS
        self.b_divine_regen[b_divine_trigger] = DIVINE_REGEN_TICKS

        return a_dmg_to_b, b_dmg_to_a

    def _apply_damage(self, dmg, target):
        health = self.a_health if target == 'a' else self.b_health
        absorb = self.a_absorption if target == 'a' else self.b_absorption

        absorb_dmg = np.minimum(absorb, dmg)
        absorb -= absorb_dmg
        remaining = dmg - absorb_dmg
        health -= remaining
        health[:] = np.maximum(health, 0)

        if target == 'a':
            self.a_health = health
            self.a_absorption = absorb
        else:
            self.b_health = health
            self.b_absorption = absorb

    def _apply_passives(self):
        # Nile's Grace regen
        a_regen = self.a_grace_timer > 0
        self.a_health[a_regen] = np.minimum(self.a_health[a_regen] + NILES_GRACE_REGEN, MAX_HEALTH)
        b_regen = self.b_grace_timer > 0
        self.b_health[b_regen] = np.minimum(self.b_health[b_regen] + NILES_GRACE_REGEN, MAX_HEALTH)

        # Divine Intervention regen
        a_div_regen = self.a_divine_regen > 0
        self.a_health[a_div_regen] = np.minimum(self.a_health[a_div_regen] + 0.15, MAX_HEALTH)
        b_div_regen = self.b_divine_regen > 0
        self.b_health[b_div_regen] = np.minimum(self.b_health[b_div_regen] + 0.15, MAX_HEALTH)

        # Lifeforce passive regen
        self.a_health = np.minimum(self.a_health + LIFEFORCE_HEAL, MAX_HEALTH)
        self.b_health = np.minimum(self.b_health + LIFEFORCE_HEAL, MAX_HEALTH)

    def _tick_timers(self):
        self.a_cooldowns = np.maximum(self.a_cooldowns - 1, 0)
        self.b_cooldowns = np.maximum(self.b_cooldowns - 1, 0)

        self.a_brace_timer = np.maximum(self.a_brace_timer - 1, 0)
        self.b_brace_timer = np.maximum(self.b_brace_timer - 1, 0)
        self.a_cleo_on_target = np.maximum(self.a_cleo_on_target - 1, 0)
        self.b_cleo_on_target = np.maximum(self.b_cleo_on_target - 1, 0)
        self.a_sand_timer = np.maximum(self.a_sand_timer - 1, 0)
        self.b_sand_timer = np.maximum(self.b_sand_timer - 1, 0)
        self.a_grace_timer = np.maximum(self.a_grace_timer - 1, 0)
        self.b_grace_timer = np.maximum(self.b_grace_timer - 1, 0)
        self.a_divine_regen = np.maximum(self.a_divine_regen - 1, 0)
        self.b_divine_regen = np.maximum(self.b_divine_regen - 1, 0)

    def _get_obs(self):
        obs = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        obs[:, 0] = self.a_health / MAX_HEALTH
        obs[:, 1] = self.a_absorption / 20.0
        obs[:, 2] = self.b_health / MAX_HEALTH
        obs[:, 3] = self.b_absorption / 20.0
        obs[:, 4] = np.clip(self.a_brace_charges / float(KINGS_BRACE_CHARGE_REQ), 0, 3.0)

        # Cooldown progress for 4 abilities (0=ready, 1=just used)
        for i in range(NUM_SIGIL_ACTIONS):
            obs[:, 5 + i] = np.clip(self.a_cooldowns[:, i] / max(COOLDOWNS[i], 1), 0, 1)

        # Active buff indicators
        obs[:, 9] = (self.a_brace_timer > 0).astype(np.float32)
        obs[:, 10] = (self.a_grace_timer > 0).astype(np.float32)
        obs[:, 11] = (self.a_sand_timer > 0).astype(np.float32)

        # Debuffs on target
        obs[:, 12] = (self.a_cleo_on_target > 0).astype(np.float32)  # our cleo on them
        obs[:, 13] = (self.b_brace_timer > 0).astype(np.float32)     # target has brace DR
        obs[:, 14] = (self.b_grace_timer > 0).astype(np.float32)     # target has grace

        obs[:, 15] = np.clip(self.tick / self.episode_length, 0, 1)

        return obs

    def _get_mask(self):
        mask = np.zeros((self.n, NUM_SIGIL_ACTIONS), dtype=np.float32)
        for i in range(NUM_SIGIL_ACTIONS):
            mask[:, i] = (self.a_cooldowns[:, i] <= 0).astype(np.float32)
        mask[:, SIG_KINGS_BRACE] *= (self.a_brace_charges >= KINGS_BRACE_CHARGE_REQ).astype(np.float32)
        return mask

    def _get_opponent_mask(self):
        mask = np.zeros((self.n, NUM_SIGIL_ACTIONS), dtype=np.float32)
        for i in range(NUM_SIGIL_ACTIONS):
            mask[:, i] = (self.b_cooldowns[:, i] <= 0).astype(np.float32)
        mask[:, SIG_KINGS_BRACE] *= (self.b_brace_charges >= KINGS_BRACE_CHARGE_REQ).astype(np.float32)
        return mask

    def _rule_opponent(self):
        actions = np.zeros((self.n, NUM_SIGIL_ACTIONS), dtype=np.int32)
        for i in range(NUM_SIGIL_ACTIONS):
            ready = self.b_cooldowns[:, i] <= 0

            if i == SIG_KINGS_BRACE:
                use = ready & (self.b_brace_charges >= KINGS_BRACE_CHARGE_REQ) & (self.b_health < 16) & (self.b_brace_timer <= 0)
            elif i == SIG_CLEOPATRA:
                has_buffs = (self.a_brace_timer > 0) | (self.a_grace_timer > 0)
                use = ready & (has_buffs | (self.a_health > 12))
            elif i == SIG_QUICK_SAND:
                use = ready & (self.a_health > 8) & (self.b_sand_timer <= 0)
            elif i == SIG_NILES_GRACE:
                use = ready & (self.b_health < 16) & (self.b_grace_timer <= 0)
            else:
                use = np.zeros(self.n, dtype=bool)

            random_use = ready & (self.rng.random(self.n) < 0.005)
            actions[:, i] = (use | random_use).astype(np.int32)

        return actions

    ABILITY_NAMES = ["King's Brace", "Cleopatra", "Quick Sand", "Nile's Grace"]

    def _compute_rewards(self, actions, a_dmg, b_dmg, a_dead, b_dead, timeout):
        rewards = np.zeros(self.n, dtype=np.float32)

        # Net damage advantage
        rewards += (a_dmg - b_dmg) * 0.5

        # Kill/death
        rewards[b_dead & ~a_dead] += 5.0
        rewards[a_dead & ~b_dead] -= 5.0

        # Time pressure
        rewards -= 0.005

        # Per-ability context rewards
        for i in range(NUM_SIGIL_ACTIONS):
            activated = actions[:, i] == 1

            if i == SIG_KINGS_BRACE:
                # Good anytime you have charges (it's always worth using)
                good = activated
                great = activated & (self.a_health < 12)
                rewards[good] += 4.0
                rewards[great] += 4.0  # +8.0 when low

            elif i == SIG_CLEOPATRA:
                has_buffs = (self.b_brace_timer > 0) | (self.b_grace_timer > 0)
                good = activated & has_buffs
                ok = activated & ~has_buffs & (self.b_health > 10)
                bad = activated & (self.b_health < 5)
                rewards[good] += 4.0  # stripping buffs is huge
                rewards[ok] += 1.0
                rewards[bad] -= 1.0

            elif i == SIG_QUICK_SAND:
                good = activated & (self.b_health > 10) & (self.a_sand_timer <= 0)
                bad = activated & (self.a_sand_timer > 0)  # wasted, already active
                rewards[good] += 2.0
                rewards[bad] -= 1.0

            elif i == SIG_NILES_GRACE:
                good = activated & (self.a_health < 16) & (self.a_grace_timer <= 0)
                great = activated & (self.a_health < 10)
                bad = activated & (self.a_health >= MAX_HEALTH)
                rewards[good] += 2.0
                rewards[great] += 3.0  # +5.0 when critical
                rewards[bad] -= 1.0

        # Penalty: took damage with brace charges available but not active
        took_damage = b_dmg > 0
        brace_wasted = (took_damage
                       & (self.a_brace_charges >= KINGS_BRACE_CHARGE_REQ)
                       & (self.a_brace_timer <= 0)
                       & (self.a_cooldowns[:, SIG_KINGS_BRACE] <= 0))
        rewards[brace_wasted] -= 1.5

        # Penalty: low HP with defensive available
        low_hp = self.a_health < 8
        if low_hp.any():
            brace_avail = low_hp & (self.a_cooldowns[:, SIG_KINGS_BRACE] <= 0) & (self.a_brace_charges >= KINGS_BRACE_CHARGE_REQ) & (self.a_brace_timer <= 0)
            grace_avail = low_hp & (self.a_cooldowns[:, SIG_NILES_GRACE] <= 0) & (self.a_grace_timer <= 0)
            rewards[brace_avail] -= 1.0
            rewards[grace_avail] -= 0.5

        return np.clip(rewards, -25.0, 25.0)

    def get_stats(self):
        total = self.episodes_completed or 1
        return {
            "episodes": int(self.episodes_completed),
            "a_kills": int(self.a_kills),
            "b_kills": int(self.b_kills),
            "draws": int(self.draws),
            "a_winrate": self.a_kills / total * 100,
            "b_winrate": self.b_kills / total * 100,
            "avg_ep_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "a_avg_hp_at_win": np.mean(self.a_hp_at_win) if self.a_hp_at_win else 0,
            "b_avg_hp_at_win": np.mean(self.b_hp_at_win) if self.b_hp_at_win else 0,
            "a_total_dmg": self.a_total_dmg_dealt,
            "b_total_dmg": self.b_total_dmg_dealt,
            "a_ability_uses": {self.ABILITY_NAMES[i]: int(self.a_ability_uses[i]) for i in range(NUM_SIGIL_ACTIONS)},
            "b_ability_uses": {self.ABILITY_NAMES[i]: int(self.b_ability_uses[i]) for i in range(NUM_SIGIL_ACTIONS)},
        }

    def reset_stats(self):
        self.a_kills = 0
        self.b_kills = 0
        self.draws = 0
        self.episodes_completed = 0
        self.a_ability_uses[:] = 0
        self.b_ability_uses[:] = 0
        self.a_total_dmg_dealt = 0.0
        self.b_total_dmg_dealt = 0.0
        self.a_hp_at_win.clear()
        self.b_hp_at_win.clear()
        self.episode_lengths.clear()

    def format_stats(self):
        s = self.get_stats()
        lines = [
            f"  Episodes: {s['episodes']}",
            f"  Agent wins: {s['a_kills']} ({s['a_winrate']:.1f}%) | "
            f"Opponent wins: {s['b_kills']} ({s['b_winrate']:.1f}%) | "
            f"Draws: {s['draws']}",
            f"  Avg episode length: {s['avg_ep_length']:.0f} ticks ({s['avg_ep_length']/20:.1f}s)",
            f"  Agent avg HP at win: {s['a_avg_hp_at_win']:.1f} | "
            f"Opponent avg HP at win: {s['b_avg_hp_at_win']:.1f}",
            f"  Total dmg — Agent: {s['a_total_dmg']:.0f} | Opponent: {s['b_total_dmg']:.0f}",
            f"  Ability uses:",
        ]
        for name, count in s['a_ability_uses'].items():
            opp_count = s['b_ability_uses'][name]
            lines.append(f"    {name:20s}  agent={count:>6}  opponent={opp_count:>6}")
        return "\n".join(lines)
