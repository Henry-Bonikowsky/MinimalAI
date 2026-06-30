"""Analyze why King's Brace usage is low."""
from .sigil_sim import KINGS_BRACE_CHARGE_REQ, KINGS_BRACE_DR, KINGS_BRACE_DURATION, BASE_HIT_CHANCE, BASE_DAMAGE, COOLDOWNS, SIG_KINGS_BRACE

hits_during_brace = BASE_HIT_CHANCE * KINGS_BRACE_DURATION
dmg_without = hits_during_brace * BASE_DAMAGE
dmg_with = dmg_without * (1 - KINGS_BRACE_DR)
print(f"Hits during brace window: {hits_during_brace:.1f}")
print(f"Damage without brace: {dmg_without:.1f} HP")
print(f"Damage WITH brace: {dmg_with:.1f} HP")
print(f"HP saved per brace use: {dmg_without - dmg_with:.1f}")
print()

ticks_per_charge = 1 / BASE_HIT_CHANCE
ticks_to_full = ticks_per_charge * KINGS_BRACE_CHARGE_REQ
print(f"Ticks to build {KINGS_BRACE_CHARGE_REQ} charges: {ticks_to_full:.0f} ({ticks_to_full/20:.1f}s)")
print(f"Starting charges: 15, need {KINGS_BRACE_CHARGE_REQ - 15} more = {ticks_per_charge * (KINGS_BRACE_CHARGE_REQ - 15):.0f} ticks")
print()

print(f"Opponent brace uses/episode: {9553/3004:.2f}")
print(f"Agent brace uses/episode: {2624/3004:.2f}")
print(f"Brace cooldown: {COOLDOWNS[SIG_KINGS_BRACE]} ticks ({COOLDOWNS[SIG_KINGS_BRACE]/20:.0f}s)")
print(f"Avg episode length: 274 ticks (13.7s)")
print()

if COOLDOWNS[SIG_KINGS_BRACE] > 274:
    print(">>> PROBLEM: brace cooldown (30s) > avg episode (13.7s)!")
    print(">>> Agent can use brace AT MOST once per fight.")
    print(">>> Opponent uses it 3.2x/episode because rule checks every tick.")
    print()
    print("FIX OPTIONS:")
    print("  1. Lower brace cooldown to 200 ticks (10s) — fits 1-2 uses per episode")
    print("  2. Longer episodes (already 3600 ticks = 180s max, but avg is 274)")
    print("  3. The avg episode is 274 because someone DIES at 274. Brace should")
    print("     prevent that death but the agent doesn't use it in time.")
    print()
    print("ROOT CAUSE: Agent waits too long to use Brace. By the time HP drops")
    print("below threshold, it's already too late — dead in a few more hits.")
    print("The OPPONENT uses it preemptively at HP < 10, which keeps it alive longer.")
    print("The AGENT needs to learn to use it BEFORE getting low, not after.")
