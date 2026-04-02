"""Quick sim: engage-only bot vs rule charger. Tune engage mechanics."""

import numpy as np
from .vec_sim import VecPvPSim
from .config import NUM_ACTIONS, ACT_FORWARD, ACT_ENGAGE, ACT_ATTACK

sim = VecPvPSim(1, episode_length=600, seed=42)
obs = sim.reset_all()

for tick in range(600):
    # Engage-only bot (player B) — just W + engage
    b_act = np.zeros((1, NUM_ACTIONS), dtype=np.int8)
    b_act[0, ACT_FORWARD] = 1
    b_act[0, ACT_ENGAGE] = 1

    # Rule charger as opponent (player A)
    from .rule_test import rule_charger
    a_act = rule_charger(sim)

    pre_a_hp = sim.a_health[0]
    pre_b_hp = sim.b_health[0]
    pre_a_hurt = sim.a_hurt_time[0]
    pre_b_hurt = sim.b_hurt_time[0]

    obs, rewards, dones, infos = sim.step(a_act, b_act)

    a_dmg = infos["a_dmg_dealt"][0]
    b_dmg = infos["b_dmg_dealt"][0]
    dist = np.sqrt((sim.ax[0] - sim.bx[0]) ** 2 + (sim.az[0] - sim.bz[0]) ** 2)

    # Debug: print every tick for first 30 ticks, then on events / every 10
    show = tick < 30 or a_dmg > 0 or b_dmg > 0 or tick % 50 == 0
    if show:
        a_combo = sim.a_combo_streak[0]
        b_combo = sim.b_combo_streak[0]
        a_spr = "S" if sim.a_sprinting[0] else " "
        b_spr = "S" if sim.b_sprinting[0] else " "
        a_gnd = "G" if sim.a_on_ground[0] else "A"
        b_gnd = "G" if sim.b_on_ground[0] else "A"
        a_ht = sim.a_hurt_time[0]
        b_ht = sim.b_hurt_time[0]
        a_del = sim.a_engage_delay[0]
        b_del = sim.b_engage_delay[0]
        events = []
        if a_dmg > 0:
            events.append(f"A->{a_dmg:.1f}")
        if b_dmg > 0:
            events.append(f"B->{b_dmg:.1f}")
        if a_dmg > 0 and b_dmg > 0:
            events.append("TRADE")
        ev = " ".join(events)
        print(
            f"t={tick:3d} d={dist:4.1f} | "
            f"A:{pre_a_hp:4.0f}hp {a_spr}{a_gnd} ht={pre_a_hurt:2d} dl={a_del} | "
            f"B:{pre_b_hp:4.0f}hp {b_spr}{b_gnd} ht={pre_b_hurt:2d} dl={b_del} | "
            f"{ev}"
        )

    if dones[0]:
        winner = "A" if sim.b_health[0] <= 0 else "B" if sim.a_health[0] <= 0 else "TIE"
        print(
            f"\nFIGHT OVER t={tick} -- Winner: {winner} | "
            f"A:{sim.a_health[0]:.1f}hp B:{sim.b_health[0]:.1f}hp"
        )
        break
