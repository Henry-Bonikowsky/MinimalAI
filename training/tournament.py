#!/usr/bin/env python3
"""Round-robin tournament between trained PvP models."""

import argparse
import itertools
import numpy as np
import torch
from .vec_sim import VecPvPSim, OBS_DIM, NUM_ACTIONS
from .models.network import CombatNetwork, GRU_HIDDEN_DIM
from .fast_train import PolicyWrapper


def load_model(path, device="cpu"):
    """Load a trained model from checkpoint."""
    net = CombatNetwork()
    policy = PolicyWrapper(net).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "network_state_dict" in ckpt:
        net.load_state_dict(ckpt["network_state_dict"])
    else:
        net.load_state_dict(ckpt)
    policy.eval()
    return policy


def fight(policy_a, policy_b, n_episodes=200, n_envs=64, episode_length=600,
          device="cpu"):
    """Run A vs B. Returns (a_kills, b_kills, timeouts, stats)."""
    sim = VecPvPSim(n_envs, episode_length=episode_length, seed=123)
    obs = sim.reset_all()

    completed = 0
    a_kills = 0
    b_kills = 0
    timeouts = 0
    a_sprint_hits = 0
    b_sprint_hits = 0
    a_total_hits = 0
    b_total_hits = 0
    ep_lengths = []
    current_lengths = np.zeros(n_envs, dtype=np.int32)

    while completed < n_episodes:
        obs_t = torch.from_numpy(obs).to(device)
        b_obs_t = torch.from_numpy(sim.get_b_obs()).to(device)

        with torch.no_grad():
            a_act, _, _, _, _ = policy_a.get_action_and_value(obs_t)
            b_act = policy_b.sample_actions(b_obs_t)

        a_np = a_act.cpu().numpy().astype(np.int8)
        b_np = b_act.cpu().numpy().astype(np.int8)

        next_obs, rewards, dones, infos = sim.step(a_np, b_np)
        current_lengths += 1

        # Track hits
        a_dmg = infos["a_dmg_dealt"]
        b_dmg = infos["b_dmg_dealt"]
        a_total_hits += (a_dmg > 0).sum()
        b_total_hits += (b_dmg > 0).sum()
        a_sprint_hits += infos["a_sprint_hits"].sum()
        b_sprint_hits += infos["b_sprint_hits"].sum()

        for i in range(n_envs):
            if dones[i] and completed < n_episodes:
                ep_lengths.append(int(current_lengths[i]))
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
        "a_kills": a_kills,
        "b_kills": b_kills,
        "timeouts": timeouts,
        "a_win_rate": a_kills / completed,
        "b_win_rate": b_kills / completed,
        "mean_length": np.mean(ep_lengths),
        "a_sprint_hit_rate": a_sprint_hits / max(a_total_hits, 1),
        "b_sprint_hit_rate": b_sprint_hits / max(b_total_hits, 1),
        "a_total_hits": int(a_total_hits),
        "b_total_hits": int(b_total_hits),
    }


def main():
    parser = argparse.ArgumentParser(description="Round-robin PvP tournament")
    parser.add_argument("models", nargs="+",
                        help="name:path pairs, e.g. wtap:checkpoints_wtap/fast_latest.pt")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Parse model names and paths
    entries = {}
    for m in args.models:
        if ":" in m:
            name, path = m.split(":", 1)
        else:
            name = m.rsplit("/", 1)[-1].replace(".pt", "")
            path = m
        entries[name] = path

    names = list(entries.keys())
    print(f"{'='*70}")
    print(f"ROUND-ROBIN TOURNAMENT ({len(names)} models, {args.episodes} episodes each)")
    print(f"{'='*70}")

    # Load all models
    models = {}
    for name, path in entries.items():
        print(f"  Loading {name}: {path}")
        models[name] = load_model(path, args.device)

    # Round-robin: every pair fights
    wins = {n: 0 for n in names}
    losses = {n: 0 for n in names}
    total_games = {n: 0 for n in names}
    results_matrix = {}

    print(f"\n{'─'*70}")
    for a_name, b_name in itertools.combinations(names, 2):
        result = fight(models[a_name], models[b_name],
                       n_episodes=args.episodes, device=args.device)

        a_wr = result["a_win_rate"]
        b_wr = result["b_win_rate"]

        print(f"\n  {a_name} vs {b_name}:")
        print(f"    {a_name}: {result['a_kills']} kills ({a_wr:.0%}) "
              f"| sprint-hit: {result['a_sprint_hit_rate']:.0%} "
              f"| hits: {result['a_total_hits']}")
        print(f"    {b_name}: {result['b_kills']} kills ({b_wr:.0%}) "
              f"| sprint-hit: {result['b_sprint_hit_rate']:.0%} "
              f"| hits: {result['b_total_hits']}")
        print(f"    Timeouts: {result['timeouts']} | Avg length: {result['mean_length']:.0f} ticks")

        wins[a_name] += result["a_kills"]
        wins[b_name] += result["b_kills"]
        losses[a_name] += result["b_kills"]
        losses[b_name] += result["a_kills"]
        total_games[a_name] += args.episodes
        total_games[b_name] += args.episodes
        results_matrix[(a_name, b_name)] = a_wr
        results_matrix[(b_name, a_name)] = b_wr

    # Leaderboard
    print(f"\n{'='*70}")
    print("LEADERBOARD")
    print(f"{'='*70}")

    rankings = sorted(names, key=lambda n: wins[n] / max(total_games[n], 1), reverse=True)
    for rank, name in enumerate(rankings, 1):
        wr = wins[name] / max(total_games[name], 1)
        print(f"  #{rank} {name:12s}  {wins[name]:3d}W / {losses[name]:3d}L  "
              f"({wr:.1%} win rate)")

    print(f"\n{'='*70}")
    print(f"CHAMPION: {rankings[0]}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
