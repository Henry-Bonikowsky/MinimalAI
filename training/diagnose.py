"""Diagnostic: watch what the trained AI actually does tick-by-tick."""

import sys
import numpy as np
import torch
from combat_sim.env import (
    CombatEnv, NUM_ACTIONS,
    ACT_FORWARD, ACT_BACKWARD, ACT_STRAFE_LEFT, ACT_STRAFE_RIGHT,
    ACT_JUMP, ACT_SNEAK, ACT_SPRINT, ACT_ATTACK, ACT_BLOCK,
    ACT_EAT_GAP, ACT_THROW_POT, ACT_THROW_PEARL, ACT_SPRINT_RESET,
    ACT_SWAP_WEAPON, ACT_SIGIL_0, ACT_LOOK_LEFT, ACT_LOOK_RIGHT,
    ACT_LOOK_UP, ACT_LOOK_DOWN, ACT_TARGET_0,
)
from models.network import CombatNetwork

ACTION_NAMES = {
    ACT_FORWARD: "FWD", ACT_BACKWARD: "BACK", ACT_STRAFE_LEFT: "STR_L",
    ACT_STRAFE_RIGHT: "STR_R", ACT_JUMP: "JUMP", ACT_SNEAK: "SNEAK",
    ACT_SPRINT: "SPRINT", ACT_ATTACK: "ATK", ACT_BLOCK: "BLOCK",
    ACT_EAT_GAP: "EAT_GAP", ACT_THROW_POT: "POT", ACT_THROW_PEARL: "PEARL",
    ACT_SPRINT_RESET: "SP_RESET", ACT_SWAP_WEAPON: "SWAP",
    ACT_LOOK_LEFT: "LOOK_L", ACT_LOOK_RIGHT: "LOOK_R",
    ACT_LOOK_UP: "LOOK_U", ACT_LOOK_DOWN: "LOOK_D",
}
for i in range(12):
    ACTION_NAMES[ACT_SIGIL_0 + i] = f"SIGIL_{i}"
for i in range(5):
    ACTION_NAMES[ACT_TARGET_0 + i] = f"TGT_{i}"


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"

    # Load model
    network = CombatNetwork()
    cp = torch.load(checkpoint, map_location="cpu", weights_only=False)
    network.load_state_dict(cp["network_state_dict"])
    network.eval()
    print(f"Loaded {checkpoint}")
    if "metadata" in cp:
        print(f"  Metadata: {cp['metadata']}")

    env = CombatEnv(num_enemies=1, episode_length=600, seed=99, domain_randomization=False)

    for ep in range(3):
        obs, info = env.reset()
        hidden = network.init_hidden(1)
        action_mask = env.get_action_mask()

        total_reward = 0.0
        action_counts = {name: 0 for name in ACTION_NAMES.values()}
        pot_ticks = []
        gap_ticks = []
        sigil_ticks = []
        block_ticks = []
        sprint_reset_ticks = []
        retreat_ticks = 0
        approach_ticks = 0
        prev_dist = None

        print(f"\n{'='*80}")
        print(f"EPISODE {ep+1}")
        print(f"Player sigils: {[s.sigil_type for s in env.player.sigil_slots if s.equipped]}")
        print(f"{'='*80}")

        for tick in range(600):
            obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}
            mask_t = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _, hidden = network.get_action_and_value(obs_t, hidden, mask_t)
            action_np = action.squeeze(0).numpy()

            # Track what actions were taken
            active = [ACTION_NAMES.get(i, f"?{i}") for i in range(NUM_ACTIONS) if action_np[i]]
            for name in active:
                if name in action_counts:
                    action_counts[name] += 1

            # Track specific events
            if action_np[ACT_THROW_POT]:
                pot_ticks.append(tick)
            if action_np[ACT_EAT_GAP]:
                gap_ticks.append(tick)
            if action_np[ACT_BLOCK]:
                block_ticks.append(tick)
            if action_np[ACT_SPRINT_RESET]:
                sprint_reset_ticks.append(tick)
            for i in range(12):
                if action_np[ACT_SIGIL_0 + i]:
                    sigil_ticks.append((tick, i))

            # Track distance to enemy
            enemy = env.enemies[0]
            dist = env.physics.distance_between(env.player, enemy)
            if prev_dist is not None:
                if dist < prev_dist - 0.1:
                    approach_ticks += 1
                elif dist > prev_dist + 0.1:
                    retreat_ticks += 1
            prev_dist = dist

            # Print key moments
            p = env.player
            should_print = (
                tick % 50 == 0 or  # every 50 ticks
                action_np[ACT_THROW_POT] or
                action_np[ACT_EAT_GAP] or
                action_np[ACT_THROW_PEARL] or
                action_np[ACT_SPRINT_RESET] or
                action_np[ACT_BLOCK] or
                action_np[ACT_SWAP_WEAPON] or
                any(action_np[ACT_SIGIL_0 + i] for i in range(12)) or
                p.damage_dealt_this_tick > 0 or
                p.damage_taken_this_tick > 0 or
                not enemy.is_alive
            )

            if should_print:
                hp_bar = int(p.health / 20 * 10)
                ehp_bar = int(enemy.health / 20 * 10) if enemy.is_alive else 0
                print(
                    f"t={tick:3d} | "
                    f"P[{'#'*hp_bar}{'.'*(10-hp_bar)}]{p.health:5.1f}hp "
                    f"{'BLOCK ' if p.is_blocking else ''}"
                    f"{'SPRINT ' if p.is_sprinting else ''}"
                    f"{'EATING ' if p.kit.is_eating else ''}"
                    f"| E[{'#'*ehp_bar}{'.'*(10-ehp_bar)}]{enemy.health:5.1f}hp "
                    f"| dist={dist:.1f} "
                    f"| dmg={p.damage_dealt_this_tick:.1f}/{p.damage_taken_this_tick:.1f} "
                    f"| {' '.join(active)}"
                )

            obs, reward, terminated, truncated, info = env.step(action_np)
            action_mask = env.get_action_mask()
            total_reward += reward

            if terminated or truncated:
                break

        print(f"\n--- EPISODE {ep+1} SUMMARY ---")
        print(f"Result: {'KILLED ENEMY' if not enemy.is_alive else 'TRUNCATED'} | "
              f"Ticks: {tick+1} | Reward: {total_reward:.3f}")
        print(f"Player HP: {env.player.health:.1f} | Enemy HP: {enemy.health:.1f}")
        print(f"Pots remaining: {env.player.kit.health_pots} | Gaps remaining: {env.player.kit.golden_apples}")
        print(f"\nAction frequencies (per tick):")
        sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
        for name, count in sorted_actions:
            if count > 0:
                print(f"  {name:12s}: {count:4d} ({count/(tick+1)*100:5.1f}%)")
        print(f"\nMovement: {approach_ticks} approach ticks, {retreat_ticks} retreat ticks")
        print(f"Pot usage at ticks: {pot_ticks}")
        print(f"Gap usage at ticks: {gap_ticks}")
        print(f"Block ticks: {len(block_ticks)} total")
        print(f"Sprint reset ticks: {len(sprint_reset_ticks)} total")
        print(f"Sigil activations: {sigil_ticks}")


if __name__ == "__main__":
    main()
