#!/usr/bin/env python3
"""Evaluate trained models against rule-based and self-play opponents."""

import argparse
import time
import numpy as np
import torch
from .vec_sim import VecPvPSim, OBS_DIM, NUM_ACTIONS
from .models.network import CombatNetwork, GRU_HIDDEN_DIM
from .fast_train import PolicyWrapper, rule_based_opponent
from .rule_test import rule_charger_smart


def evaluate(model_path, n_episodes=500, n_envs=64, episode_length=1800, device="cpu", combo_style="stap", gauntlet=False, smart_opponent=False):
    """Run evaluation episodes and return stats."""
    rng = np.random.default_rng(123)
    sim = VecPvPSim(n_envs, episode_length=episode_length, seed=123, combo_style=combo_style, gauntlet=gauntlet)
    if gauntlet:
        sim.gauntlet_survivor = 'a'  # In evaluate, model is A, rule opponent is B

    network = CombatNetwork()
    policy = PolicyWrapper(network).to(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "network_state_dict" in ckpt:
        network.load_state_dict(ckpt["network_state_dict"])
    else:
        network.load_state_dict(ckpt)
    policy.eval()

    steps_trained = ckpt.get("total_steps", "?")
    episodes_trained = ckpt.get("total_episodes", "?")

    obs = sim.reset_all()
    completed = 0
    kills = 0        # A killed B (b_health <= 0)
    deaths = 0       # B killed A (a_health <= 0)
    timeouts = 0     # Neither died
    total_rewards = []
    ep_lengths = []
    total_damage_dealt = []
    total_damage_taken = []
    total_kill_counts = []
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype=np.int32)
    current_dmg_dealt = np.zeros(n_envs)
    current_dmg_taken = np.zeros(n_envs)

    while completed < n_episodes:
        obs_t = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            action, _, _, _, _ = policy.get_action_and_value(obs_t)
        action_np = action.cpu().numpy().astype(np.int8)

        # Rule-based opponent
        b_actions = rule_charger_smart(sim) if smart_opponent else rule_based_opponent(sim, rng)

        next_obs, rewards, dones, infos = sim.step(action_np, b_actions)

        current_rewards += rewards
        current_lengths += 1
        # Track per-tick damage from sim
        current_dmg_dealt += infos.get("a_dmg_dealt", np.zeros(n_envs))
        current_dmg_taken += infos.get("b_dmg_dealt", np.zeros(n_envs))

        for i in range(n_envs):
            if dones[i] and completed < n_episodes:
                total_rewards.append(float(current_rewards[i]))
                ep_lengths.append(int(current_lengths[i]))
                total_damage_dealt.append(float(current_dmg_dealt[i]))
                total_damage_taken.append(float(current_dmg_taken[i]))
                total_kill_counts.append(int(infos.get("kill_count", np.zeros(n_envs))[i]))

                # Determine win/loss by actual health (infos captured before reset)
                a_hp = infos["a_health"][i]
                b_hp = infos["b_health"][i]
                if gauntlet:
                    # In gauntlet, episode ends on A (model) death or timeout
                    if a_hp <= 0:
                        deaths += 1
                    else:
                        timeouts += 1  # survived to timeout
                else:
                    if b_hp <= 0:
                        kills += 1
                    elif a_hp <= 0:
                        deaths += 1
                    else:
                        timeouts += 1

                current_rewards[i] = 0
                current_lengths[i] = 0
                current_dmg_dealt[i] = 0
                current_dmg_taken[i] = 0
                completed += 1

        obs = next_obs

    results = {
        "model": model_path,
        "steps_trained": steps_trained,
        "episodes_trained": episodes_trained,
        "n_eval_episodes": completed,
        "kill_rate": kills / completed,
        "death_rate": deaths / completed,
        "timeout_rate": timeouts / completed,
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "median_reward": np.median(total_rewards),
        "mean_length": np.mean(ep_lengths),
        "mean_dmg_dealt": np.mean(total_damage_dealt),
        "mean_dmg_taken": np.mean(total_damage_taken),
    }
    if gauntlet:
        results["mean_kills"] = np.mean(total_kill_counts)
        results["max_kills"] = max(total_kill_counts) if total_kill_counts else 0
    return results


def evaluate_selfplay(model_path, n_episodes=500, n_envs=64, episode_length=1800, device="cpu", combo_style="stap"):
    """Evaluate model against itself (check for degenerate strategies)."""
    sim = VecPvPSim(n_envs, episode_length=episode_length, seed=456, combo_style=combo_style)

    network = CombatNetwork()
    policy = PolicyWrapper(network).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if "network_state_dict" in ckpt:
        network.load_state_dict(ckpt["network_state_dict"])
    policy.eval()

    obs = sim.reset_all()
    completed = 0
    ep_lengths = []
    a_kills = 0
    b_kills = 0
    timeouts = 0
    current_lengths = np.zeros(n_envs, dtype=np.int32)

    while completed < n_episodes:
        obs_t = torch.from_numpy(obs).to(device)
        b_obs = sim.get_b_obs()
        b_obs_t = torch.from_numpy(b_obs).to(device)

        with torch.no_grad():
            a_action, _, _, _, _ = policy.get_action_and_value(obs_t)
            b_action = policy.sample_actions(b_obs_t)

        a_np = a_action.cpu().numpy().astype(np.int8)
        b_np = b_action.cpu().numpy().astype(np.int8)

        # Both face each other
        dx = sim.ax - sim.bx
        dz = sim.az - sim.bz
        sim.byaw[:] = np.degrees(np.arctan2(-dx, dz))

        next_obs, rewards, dones, infos = sim.step(a_np, b_np)
        current_lengths += 1

        for i in range(n_envs):
            if dones[i] and completed < n_episodes:
                ep_lengths.append(int(current_lengths[i]))
                # Use infos health (captured BEFORE auto-reset)
                if infos["b_health"][i] <= 0:
                    a_kills += 1
                elif infos["a_health"][i] <= 0:
                    b_kills += 1
                else:
                    timeouts += 1
                current_lengths[i] = 0
                completed += 1

        obs = next_obs

    return {
        "n_episodes": completed,
        "a_kill_rate": a_kills / completed,
        "b_kill_rate": b_kills / completed,
        "timeout_rate": timeouts / completed,
        "mean_length": np.mean(ep_lengths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="Checkpoint paths to evaluate")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--selfplay-test", action="store_true")
    parser.add_argument("--combo-style", choices=["stap", "wtap", "none"], default="stap")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gauntlet", action="store_true", default=False)
    parser.add_argument("--smart-opponent", action="store_true", default=False,
                        help="Evaluate against smart charger that heals")
    args = parser.parse_args()

    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    for path in args.models:
        print(f"\n{'-' * 60}")
        print(f"Model: {path}")
        print(f"{'-' * 60}")

        t0 = time.time()
        results = evaluate(path, n_episodes=args.episodes, n_envs=args.n_envs, device=args.device, combo_style=args.combo_style, gauntlet=args.gauntlet, smart_opponent=args.smart_opponent)
        elapsed = time.time() - t0

        print(f"  Trained:     {results['steps_trained']} steps / {results['episodes_trained']} episodes")
        print(f"  vs Rule-Based ({results['n_eval_episodes']} episodes, {elapsed:.1f}s):")
        print(f"    Kills:       {results['kill_rate']:.1%}  (model killed opponent)")
        print(f"    Deaths:      {results['death_rate']:.1%}  (opponent killed model)")
        print(f"    Timeouts:    {results['timeout_rate']:.1%}")
        print(f"    Mean reward: {results['mean_reward']:.3f} (±{results['std_reward']:.3f})")
        print(f"    Mean length: {results['mean_length']:.0f} ticks")
        print(f"    Avg dmg dealt: {results['mean_dmg_dealt']:.1f} HP")
        print(f"    Avg dmg taken: {results['mean_dmg_taken']:.1f} HP")
        if 'mean_kills' in results:
            print(f"    Gauntlet kills: {results['mean_kills']:.1f} avg, {results['max_kills']} max")

        if args.selfplay_test:
            sp = evaluate_selfplay(path, n_episodes=args.episodes, n_envs=args.n_envs, device=args.device, combo_style=args.combo_style)
            print(f"  Self-play ({sp['n_episodes']} episodes):")
            print(f"    A kills: {sp['a_kill_rate']:.1%}  B kills: {sp['b_kill_rate']:.1%}  Timeouts: {sp['timeout_rate']:.1%}")
            print(f"    Mean length: {sp['mean_length']:.0f} ticks")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
