"""Rule-based bot test: can the sim physics produce combos?

Charger: sprint forward, face target, attack when in range.
Counter: approach to ~3.5, stop, wait for hit, then sprint-hit and chase combo.
"""

import numpy as np
from .vec_sim import VecPvPSim
from .config import (
    NUM_ACTIONS, ACT_FORWARD, ACT_BACKWARD, ACT_ATTACK,
    ACT_FACE_TARGET, ACT_ENGAGE, ACT_JUMP, ACT_LEFT, ACT_RIGHT,
    ACT_EAT_GAP, ACT_THROW_POT, DEFAULT_ATTACK_REACH as ATTACK_REACH,
)


def make_actions(n):
    return np.zeros((n, NUM_ACTIONS), dtype=np.int8)


def rule_charger(sim):
    """Brainless charger: sprint forward, engage (auto-attacks when in reach)."""
    a = make_actions(sim.n)
    a[:, ACT_FORWARD] = 1
    a[:, ACT_ENGAGE] = 1  # face + auto-attack handled by sim
    return a


def init_noisy_charger(sim):
    """Initialize per-episode random parameters for noisy charger.
    Call after sim.reset_all() and on episode resets."""
    n = sim.n
    rng = np.random.default_rng()
    sim._nc_approach_delay = rng.integers(0, 41, n)  # 0-40 ticks before moving
    sim._nc_attack_delay = rng.integers(0, 6, n)     # 0-5 tick delay before swinging
    sim._nc_strafe_chance = rng.uniform(0.0, 0.3, n)  # 0-30% strafe per tick
    sim._nc_retreat_chance = rng.uniform(0.0, 0.3, n)  # 0-30% chance to back off after hit
    sim._nc_retreat_ticks = np.zeros(n, dtype=np.int32)  # countdown for retreat
    sim._nc_attack_wait = np.zeros(n, dtype=np.int32)    # countdown for attack delay
    sim._nc_in_range_tick = np.zeros(n, dtype=np.int32)  # tick when first entered range


def reset_noisy_charger(sim, idx):
    """Reset noisy charger params for specific env indices (on episode reset)."""
    n = len(idx)
    rng = np.random.default_rng()
    sim._nc_approach_delay[idx] = rng.integers(0, 41, n)
    sim._nc_attack_delay[idx] = rng.integers(0, 6, n)
    sim._nc_strafe_chance[idx] = rng.uniform(0.0, 0.3, n)
    sim._nc_retreat_chance[idx] = rng.uniform(0.0, 0.3, n)
    sim._nc_retreat_ticks[idx] = 0
    sim._nc_attack_wait[idx] = 0
    sim._nc_in_range_tick[idx] = 0


def rule_charger_noisy(sim):
    """Randomized charger: human-like with variable timing and movement.

    Per-episode randomization makes every fight slightly different, forcing
    the counter to react to the hit itself rather than memorize patterns.
    """
    a = make_actions(sim.n)
    a[:, ACT_ENGAGE] = 1  # always engage (face + auto-attack handled by sim)

    # Approach delay: don't move for first N ticks (sizing up)
    past_delay = sim.tick >= sim._nc_approach_delay

    # Retreat after landing a hit
    just_landed_hit = (sim.b_hurt_time == (10 - 1))  # we hit them last tick
    rng = np.random.default_rng()
    roll = rng.random(sim.n)
    start_retreat = just_landed_hit & (roll < sim._nc_retreat_chance) & (sim._nc_retreat_ticks <= 0)
    sim._nc_retreat_ticks[start_retreat] = rng.integers(5, 16, start_retreat.sum())
    retreating = sim._nc_retreat_ticks > 0
    sim._nc_retreat_ticks[retreating] -= 1

    # Movement
    moving = past_delay & ~retreating
    a[moving, ACT_FORWARD] = 1
    a[retreating, ACT_BACKWARD] = 1

    # Random strafing while approaching
    strafe_roll = rng.random(sim.n)
    strafing = moving & (strafe_roll < sim._nc_strafe_chance)
    strafe_dir = rng.random(sim.n) < 0.5
    a[strafing & strafe_dir, ACT_LEFT] = 1
    a[strafing & ~strafe_dir, ACT_RIGHT] = 1

    # Disengage while retreating (don't auto-attack while backing off)
    a[retreating, ACT_ENGAGE] = 0

    # Consumables: eat gap when low HP (~50% chance per tick when hurt)
    low_hp = sim.a_health < 12.0
    eat_roll = rng.random(sim.n) < 0.03  # ~3% per tick ≈ eats within ~30 ticks of being low
    can_eat = (sim.a_eat_cooldown <= 0) & (sim.a_eating_ticks <= 0) & (sim.a_gapple_count > 0)
    a[low_hp & eat_roll & can_eat, ACT_EAT_GAP] = 1

    # Throw pot when very low HP (~2% per tick)
    very_low = sim.a_health < 8.0
    pot_roll = rng.random(sim.n) < 0.02
    can_pot = (sim.a_pot_count > 0) & (sim.a_pot_lockout <= 0) & (sim.a_eating_ticks <= 0)
    a[very_low & pot_roll & can_pot, ACT_THROW_POT] = 1

    return a


def rule_charger_smart(sim):
    """Smart charger: charges aggressively + heals proactively.

    Forces longer fights where the model must also heal to win.
    - Eats gapple when HP < 16 (proactive regen)
    - Pots instantly when HP < 8 (emergency)
    - Retreats while eating to protect the eat
    - Re-engages immediately after eat finishes
    """
    a = make_actions(sim.n)

    is_eating = sim.a_eating_ticks > 0

    # ── Healing decisions (before movement) ──
    # Start gapple at HP < 16 (deterministic — always eats when available)
    want_eat = (sim.a_health < 16.0) & ~is_eating
    can_eat = (sim.a_eat_cooldown <= 0) & (sim.a_gapple_count > 0)
    a[want_eat & can_eat, ACT_EAT_GAP] = 1

    # Emergency pot at HP < 8 (instant heal, no eat required)
    want_pot = (sim.a_health < 8.0) & ~is_eating
    can_pot = (sim.a_pot_count > 0) & (sim.a_pot_lockout <= 0)
    a[want_pot & can_pot, ACT_THROW_POT] = 1

    # ── Movement ──
    # Retreat while eating (protect the eat)
    a[is_eating, ACT_BACKWARD] = 1
    # Charge forward when not eating
    a[~is_eating, ACT_FORWARD] = 1

    # ── Combat ──
    # Engage (face + auto-attack) when not eating
    a[~is_eating, ACT_ENGAGE] = 1

    return a


def rule_counter(sim):
    """Counter-hitter: hold at ~3.5 blocks, wait for hit, then combo.

    State machine per env:
      0 = APPROACH: sprint toward until dist < 3.8
      1 = WAIT: stop moving, face target, don't attack
      2 = HIT_TAKEN: just got hit, wait to land
      3 = COMBO: on ground after being hit, sprint-hit and chase
    """
    a = make_actions(sim.n)

    dx = sim.ax - sim.bx
    dz = sim.az - sim.bz
    dy = sim.ay - sim.by
    dist = np.sqrt(dx**2 + dz**2 + dy**2)

    # State transitions
    # APPROACH → WAIT when close enough
    approach = sim._counter_state == 0
    close_enough = dist < 3.8
    sim._counter_state[approach & close_enough] = 1

    # WAIT → HIT_TAKEN when we get hit (hurt_time just went to 10)
    waiting = sim._counter_state == 1
    just_hit = sim.b_hurt_time >= 9  # just got hit this tick or last
    sim._counter_state[waiting & just_hit] = 2

    # HIT_TAKEN → COMBO when we land on ground
    hit_taken = sim._counter_state == 2
    on_ground = sim.b_on_ground
    sim._counter_state[hit_taken & on_ground] = 3

    # COMBO → WAIT when opponent lands and is no longer in our i-frames
    # (i.e., the exchange is over, reset to waiting)
    comboing = sim._counter_state == 3
    opp_grounded = sim.a_on_ground
    opp_no_iframes = sim.a_hurt_time <= 0
    our_no_iframes = sim.b_hurt_time <= 0
    exchange_over = opp_grounded & opp_no_iframes & our_no_iframes & (dist > 3.0)
    sim._counter_state[comboing & exchange_over] = 0  # back to approach

    # Also reset to approach if we drift too far
    too_far = dist > 6.0
    sim._counter_state[too_far & (sim._counter_state > 0)] = 0

    # Actions based on state
    # APPROACH: move forward, engage (face + auto-attack)
    approach = sim._counter_state == 0
    a[approach, ACT_FORWARD] = 1
    a[approach, ACT_ENGAGE] = 1

    # WAIT: don't move, face target but don't engage (no auto-attack)
    waiting = sim._counter_state == 1
    a[waiting, ACT_ENGAGE] = 1  # face to track opponent

    # HIT_TAKEN: we're airborne from KB, face target
    hit_taken = sim._counter_state == 2
    a[hit_taken, ACT_ENGAGE] = 1

    # COMBO: sprint forward, engage (auto-attack handles timing)
    comboing = sim._counter_state == 3
    a[comboing, ACT_FORWARD] = 1
    a[comboing, ACT_ENGAGE] = 1

    return a


def run_test(n_envs=1, ticks=600, verbose=True):
    sim = VecPvPSim(n_envs=n_envs, episode_length=ticks)
    sim._counter_state = np.zeros(n_envs, dtype=np.int32)
    obs = sim.reset_all()

    # Track per-env stats
    a_hits = np.zeros(n_envs, dtype=np.int32)
    b_hits = np.zeros(n_envs, dtype=np.int32)
    a_combo = np.zeros(n_envs, dtype=np.int32)
    b_combo = np.zeros(n_envs, dtype=np.int32)
    a_max_combo = np.zeros(n_envs, dtype=np.int32)
    b_max_combo = np.zeros(n_envs, dtype=np.int32)
    trades = np.zeros(n_envs, dtype=np.int32)

    for t in range(ticks):
        a_act = rule_charger(sim)
        b_act = rule_counter(sim)

        obs, rewards, dones, infos = sim.step(a_act, b_act)

        a_hit = infos["a_sprint_hits"]
        b_hit = infos["b_sprint_hits"]
        a_dmg = infos["a_dmg_dealt"]
        b_dmg = infos["b_dmg_dealt"]
        trade = a_hit & b_hit

        a_hits += a_hit.astype(np.int32)
        b_hits += b_hit.astype(np.int32)
        trades += trade.astype(np.int32)

        # Combo tracking
        a_combo[a_hit] += 1
        a_combo[~a_hit & (a_dmg == 0)] = 0  # reset on whiff tick? no, reset when B hits us
        a_combo[b_hit] = 0
        a_max_combo = np.maximum(a_max_combo, a_combo)

        b_combo[b_hit] += 1
        b_combo[a_hit] = 0
        b_max_combo = np.maximum(b_max_combo, b_combo)

        if verbose and n_envs == 1:
            dx = sim.bx[0] - sim.ax[0]
            dz = sim.bz[0] - sim.az[0]
            dist = np.sqrt(dx**2 + dz**2)
            state_names = ["APPROACH", "WAIT", "HIT_TAKEN", "COMBO"]
            state = state_names[sim._counter_state[0]]

            events = []
            if a_hit[0]:
                events.append(f"A->HIT {a_dmg[0]:.1f} @ {infos['a_hit_dist'][0]:.2f}")
            if b_hit[0]:
                events.append(f"B->HIT {b_dmg[0]:.1f} @ {infos['b_hit_dist'][0]:.2f}")
            if trade[0]:
                events.append("TRADE!")
            if b_hit[0] and b_combo[0] > 1:
                events.append(f"COMBO x{b_combo[0]}")
            if a_hit[0] and a_combo[0] > 1:
                events.append(f"A-COMBO x{a_combo[0]}")

            ev_str = " | ".join(events) if events else ""

            # Only print interesting ticks
            if events or t % 20 == 0:
                print(f"  t={t:3d} | d={dist:.1f} | st={state:9s} | "
                      f"Ahp={sim.a_health[0]:5.1f} Bhp={sim.b_health[0]:5.1f} | "
                      f"A_gnd={'Y' if sim.a_on_ground[0] else 'N'} "
                      f"B_gnd={'Y' if sim.b_on_ground[0] else 'N'} | "
                      f"A_iF={sim.a_hurt_time[0]:2d} B_iF={sim.b_hurt_time[0]:2d} | "
                      f"{ev_str}")

        if dones.any():
            break

    print("\n=== RESULTS ===")
    for i in range(min(n_envs, 5)):
        print(f"  Env {i}: A_hits={a_hits[i]} B_hits={b_hits[i]} "
              f"A_max_combo={a_max_combo[i]} B_max_combo={b_max_combo[i]} "
              f"trades={trades[i]}")
    print(f"\n  AVG: A_hits={a_hits.mean():.1f} B_hits={b_hits.mean():.1f} "
          f"A_max_combo={a_max_combo.mean():.1f} B_max_combo={b_max_combo.mean():.1f} "
          f"trades={trades.mean():.1f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--ticks", type=int, default=600)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run_test(n_envs=args.envs, ticks=args.ticks, verbose=not args.quiet)
