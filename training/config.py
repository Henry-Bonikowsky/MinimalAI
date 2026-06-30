"""Centralized training config.

Shared constants (action space, observation dims, consumables, network arch)
are generated from dimensions.json by scripts/gen_config.py.
Run `python scripts/gen_config.py` if dimensions_generated.py is missing.

Reward configs are Python-only and defined below.
"""

from .dimensions_generated import *  # noqa: F401,F403

# ============================================================
#  Reward Configs
#  Each dict defines reward weights for a playstyle.
#  Used by VecPvPSim._compute_rewards() — player A and B
#  can each have their own config.
# ============================================================

# Shared base rewards (used by all styles unless overridden)
REWARDS_BASE = {
    # Terminal
    "kill":                 20.0,
    "death":               -20.0,

    # Combo
    "combo_base":           3.0,   # first hit in combo
    "combo_escalation":     2.0,   # bonus per consecutive hit (streak * this)
    "clean_hit":            2.0,   # sprint-hit without getting hit back
    "trade":               -1.5,   # both hit same tick
    "combo_broken_base":   -2.0,   # lost a combo streak >= 2
    "combo_broken_scale":  -1.0,   # additional per streak length
    "non_sprint_hit":      -1.0,   # landed damage but didn't sprint-hit

    # Damage
    "net_dmg_scale":        0.5,   # reward for net damage advantage
    "raw_dmg_scale":        0.3,   # reward for any damage dealt
    "time_pressure":       -0.02,  # per-tick cost (forces action)

    # Spacing
    "max_range_hit":        2.0,   # sprint-hit at 2.8-3.0 range
    "max_range_min_dist":   2.8,   # min dist to qualify as max-range
    "in_range_bonus":       0.1,   # per-tick reward for being in melee range
    "passive_penalty":     -0.1,   # healthy + far away = wasting time
    "passive_hp_threshold": 12.0,
    "passive_dist_threshold": 5.0,

    # Distance shaping
    "close_reward":         0.3,   # reward per block closed
    "retreat_penalty":     -0.5,   # penalty per block retreated (when not getting comboed)
    "health_diff_scale":    0.05,  # reward for health advantage change

    # Jump
    "jump_spam":           -0.1,   # jumping while airborne

    # Consumables — golden apple
    "eat_on_cd":           -0.5,   # pressed eat while on cooldown
    "eat_while_eating":    -0.3,   # spam eat while already eating
    "wasteful_eat":        -3.0,   # eating at high HP
    "wasteful_eat_hp":     18.0,   # HP threshold for wasteful eat
    "wasteful_eat_abs":     4.0,   # absorption threshold for wasteful eat
    "eat_no_items":        -0.5,   # no gapples left
    "good_eat":             3.0,   # started eating at low HP (smart heal)
    "good_eat_hp":         14.0,   # HP threshold to qualify as good eat
    "safe_eat":             1.5,   # eating while opponent is far (safe window)
    "safe_eat_dist":        5.0,   # dist to qualify as safe eat
    "danger_eat":          -2.0,   # eating while opponent is close (risky)
    "danger_eat_dist":      3.0,   # dist threshold for risky eat

    # Consumables — splash pot
    "pot_splash_enemy":    -8.0,   # healed the enemy
    "pot_while_busy":      -0.3,   # pot during eat/lockout
    "wasteful_pot":        -3.0,   # pot at high HP
    "wasteful_pot_hp":     16.0,   # HP threshold
    "pot_no_items":        -0.5,   # no pots left
    "critical_pot":         4.0,   # potted at critical HP (emergency heal)
    "critical_pot_hp":      8.0,   # HP threshold for critical pot

    # Consumables — ender pearl
    "pearl_while_busy":    -0.3,
    "pearl_far":            1.5,   # pearl to close distance when far
    "pearl_far_threshold":  8.0,   # dist to qualify as "far"
    "pearl_close":         -3.0,   # pearl when already in melee range
    "pearl_no_items":      -0.1,

    # Clip
    "reward_clip":         25.0,   # clip rewards to [-clip, +clip]
}

# Charger: aggressive, always closing, always swinging
REWARDS_CHARGER = {
    **REWARDS_BASE,
}

# Counter: patient, waits for hit, then combos
REWARDS_COUNTER = {
    **REWARDS_BASE,
    "counter_hit":          8.0,   # got hit first → sprint-hit back
    "clean_hit":            3.0,   # counter hates trades even more
    "trade":               -3.0,   # stronger trade penalty
    "aggressor_penalty":   -2.0,   # engaging before being hit (patience)
    # Counter spacing: hold medium range
    "close_reward":         0.15,  # weaker closing (doesn't rush)
    "retreat_penalty":      0.0,   # no retreat penalty (counter can retreat)
    "sweet_spot_bonus":     0.1,   # reward for 3.0-4.0 range
    "sweet_spot_close_pen": -0.15, # penalty for closing past sweet spot
}

# Hybrid: aggressive + counter-hit awareness
REWARDS_HYBRID = {
    **REWARDS_BASE,
    "counter_hit":          5.0,
    "max_range_hit":        3.0,
    "max_range_min_dist":   2.5,   # slightly closer counts
    # Hybrid spacing
    "sweet_spot_bonus":     0.15,  # 2.5-3.5 range
    "too_close_penalty":   -0.2,   # < 2.0 range
    "passive_dist_threshold": 6.0,
    "passive_penalty":     -0.15,
}

# Aggressive: prioritizes damage output, closing distance, finishing kills
REWARDS_AGGRESSIVE = {
    **REWARDS_BASE,
    "kill": 25.0,           # stronger kill reward
    "death": -15.0,         # less death penalty (encourages aggression)
    "combo_base": 4.0,      # big combo rewards
    "combo_escalation": 3.0,
    "clean_hit": 3.0,
    "trade": -0.5,          # doesn't mind trading
    "close_reward": 0.5,    # strong closing incentive
    "retreat_penalty": -1.0, # hates retreating
    "raw_dmg_scale": 0.5,   # loves dealing damage
    "net_dmg_scale": 0.3,
    "good_eat_hp": 10.0,    # only eats when really low
    "wasteful_eat_hp": 14.0, # lower threshold for wasteful
    "passive_penalty": -0.3, # strong idle penalty
}

# Defensive: prioritizes survival, healing, counter-attacking
REWARDS_DEFENSIVE = {
    **REWARDS_BASE,
    "kill": 15.0,           # weaker kill reward
    "death": -25.0,         # hates dying
    "counter_hit": 8.0,     # loves counter-hitting
    "clean_hit": 3.0,
    "trade": -3.0,          # hates trading
    "good_eat": 5.0,        # big reward for smart heals
    "good_eat_hp": 16.0,    # eats proactively
    "critical_pot": 6.0,    # big reward for emergency pots
    "safe_eat": 3.0,        # loves safe eating
    "close_reward": 0.15,   # doesn't rush
    "retreat_penalty": 0.0,  # no retreat penalty
    "sweet_spot_bonus": 0.15, # holds medium range
    "net_dmg_scale": 0.8,    # cares about NET damage (not raw)
    "raw_dmg_scale": 0.1,
    "health_diff_scale": 0.15, # big reward for health advantage
}
