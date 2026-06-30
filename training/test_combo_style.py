"""Quick test: verify combo styles + healing + calibrated damage."""
import numpy as np
from .vec_sim import VecPvPSim
from .config import ACT_FORWARD, ACT_ENGAGE, ACT_EAT_GAP, ACT_THROW_POT, NUM_ACTIONS


def test_damage():
    """Verify damage is 3.0 per hit, 4.0 per crit."""
    sim = VecPvPSim(1, combo_style="stap", episode_length=200)
    sim.reset_all()
    print(f"  base_dmg_after_armor = {sim.base_dmg_after_armor}")
    print(f"  crit_dmg_after_armor = {sim.crit_dmg_after_armor}")

    # Hit once and check damage
    a = np.zeros((1, NUM_ACTIONS), dtype=np.int8)
    a[:, ACT_FORWARD] = 1
    a[:, ACT_ENGAGE] = 1
    b = np.zeros((1, NUM_ACTIONS), dtype=np.int8)

    pre_hp = sim.b_health[0]
    for t in range(50):
        obs, rewards, dones, infos = sim.step(a, b)
        if infos["a_sprint_hits"][0]:
            dmg = pre_hp - sim.b_health[0]
            print(f"  First hit at tick {t}: {dmg:.1f} HP damage (expected 3.0)")
            break
        pre_hp = sim.b_health[0]


def test_healing():
    """Verify gapple eating works."""
    sim = VecPvPSim(1, combo_style="stap", episode_length=200)
    sim.reset_all()
    # Manually damage player A
    sim.a_health[0] = 8.0

    a = np.zeros((1, NUM_ACTIONS), dtype=np.int8)
    a[:, ACT_EAT_GAP] = 1  # eat gapple
    b = np.zeros((1, NUM_ACTIONS), dtype=np.int8)

    pre_hp = sim.a_health[0]
    pre_gaps = sim.a_gapple_count[0]
    for t in range(50):
        sim.step(a, b)
        if sim.a_gapple_count[0] < pre_gaps:
            print(f"  Gapple consumed at tick {t}, HP: {pre_hp:.0f} -> {sim.a_health[0]:.1f}, "
                  f"absorption: {sim.a_absorption[0]:.1f}, gaps left: {sim.a_gapple_count[0]}")
            break
    # Let regen tick
    a[:, ACT_EAT_GAP] = 0
    for t in range(120):
        sim.step(a, b)
    print(f"  After regen: HP={sim.a_health[0]:.1f}, absorption={sim.a_absorption[0]:.1f}")


def test_combo_styles():
    """Compare combo styles: A charges B (punching bag)."""
    print(f"\n  {'style':5s}  {'a_hits':>6s}  {'combo':>5s}  {'dmg/hit':>7s}")
    for style in ["none", "wtap", "stap"]:
        sim = VecPvPSim(64, combo_style=style, episode_length=300)
        sim.reset_all()
        total_hits = 0
        total_dmg = 0.0
        max_combo = 0
        for t in range(300):
            a = np.zeros((64, NUM_ACTIONS), dtype=np.int8)
            a[:, ACT_FORWARD] = 1
            a[:, ACT_ENGAGE] = 1
            b = np.zeros((64, NUM_ACTIONS), dtype=np.int8)
            obs, rewards, dones, infos = sim.step(a, b)
            h = infos["a_sprint_hits"].sum()
            total_hits += h
            total_dmg += infos["a_dmg_dealt"].sum()
            max_combo = max(max_combo, sim.a_combo_streak.max())
            if dones.any():
                sim.reset_envs(dones)
        avg_dmg = total_dmg / max(total_hits, 1)
        print(f"  {style:5s}  {total_hits:6d}  {max_combo:5d}  {avg_dmg:7.2f}")


print("=== Damage Calibration ===")
test_damage()
print("\n=== Healing (Gapple) ===")
test_healing()
print("\n=== Combo Styles (A charges, B stands) ===")
test_combo_styles()
