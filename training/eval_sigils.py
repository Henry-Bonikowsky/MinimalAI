#!/usr/bin/env python3
"""Evaluate a trained sigil model.

Runs the model against:
1. Rule-based opponent (same as training)
2. Copy of itself (self-play)
3. Random opponent (baseline)

Usage:
    python -m training.eval_sigils checkpoints_sigils_v3/sigil_model.pt
    python -m training.eval_sigils checkpoints_sigils_v3/sigil_model.pt --episodes 1000
"""

import argparse
import time

import numpy as np
import torch

from .sigil_sim import SigilSim, NUM_SIGIL_ACTIONS, OBS_DIM
from .models.sigil_network import SigilNet


def load_model(path, device):
    model = SigilNet().to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_eval(model, sim, n_episodes, device, opponent_fn=None, opponent_model=None):
    """Run evaluation episodes and return stats.

    opponent_fn: callable(sim) -> actions for opponent, or None for sim's rule opponent
    opponent_model: SigilNet for self-play, or None
    """
    sim.reset()
    sim.reset_stats()

    steps = 0
    while sim.episodes_completed < n_episodes:
        obs, mask = sim._get_obs(), sim._get_mask()

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            action, _, _, _ = model.get_action_and_value(obs_t, mask_t)
        a_actions = action.cpu().numpy().astype(np.int32)

        if opponent_model is not None:
            # Self-play: build opponent obs (swap A/B perspective)
            b_obs = _swap_perspective_obs(obs)
            b_mask = sim._get_opponent_mask()
            with torch.no_grad():
                b_obs_t = torch.from_numpy(b_obs).to(device)
                b_mask_t = torch.from_numpy(b_mask).to(device)
                b_action, _, _, _ = opponent_model.get_action_and_value(b_obs_t, b_mask_t)
            b_actions = b_action.cpu().numpy().astype(np.int32)
        elif opponent_fn is not None:
            b_actions = opponent_fn(sim)
        else:
            b_actions = None  # use sim's built-in rule opponent

        sim.step(a_actions, b_actions)
        steps += sim.n

    return sim.get_stats()


def _swap_perspective_obs(obs):
    """Swap A/B in observation so opponent sees from their own perspective."""
    swapped = obs.copy()
    # Swap health/absorption (indices 0-1 ↔ 2-3)
    swapped[:, 0], swapped[:, 2] = obs[:, 2].copy(), obs[:, 0].copy()
    swapped[:, 1], swapped[:, 3] = obs[:, 3].copy(), obs[:, 1].copy()
    # Swap target effects (indices 16-17 ↔ 18)
    # This is approximate — good enough for eval
    return swapped


def random_opponent(sim):
    """Random opponent: activate each ability with 1% chance when ready."""
    actions = np.zeros((sim.n, NUM_SIGIL_ACTIONS), dtype=np.int32)
    for i in range(NUM_SIGIL_ACTIONS):
        ready = sim.b_cooldowns[:, i] <= 0
        actions[:, i] = (ready & (sim.rng.random(sim.n) < 0.01)).astype(np.int32)
    return actions


def main():
    p = argparse.ArgumentParser(description="Evaluate sigil model")
    p.add_argument("model_path", type=str)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--n-envs", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    print(f"Loaded model: {args.model_path}")
    print(f"Evaluating with {args.episodes} episodes per matchup...\n")

    # 1. vs Rule opponent
    print("=" * 60)
    print("  vs RULE OPPONENT (same as training)")
    print("=" * 60)
    sim = SigilSim(n_envs=args.n_envs, seed=123)
    stats = run_eval(model, sim, args.episodes, device)
    print(sim.format_stats())
    print()

    # 2. vs Random opponent
    print("=" * 60)
    print("  vs RANDOM OPPONENT (baseline)")
    print("=" * 60)
    sim = SigilSim(n_envs=args.n_envs, seed=456)
    stats = run_eval(model, sim, args.episodes, device, opponent_fn=random_opponent)
    print(sim.format_stats())
    print()

    # 3. vs Self (self-play)
    print("=" * 60)
    print("  vs SELF (mirror match)")
    print("=" * 60)
    sim = SigilSim(n_envs=args.n_envs, seed=789)
    stats = run_eval(model, sim, args.episodes, device, opponent_model=model)
    print(sim.format_stats())
    print()

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)


if __name__ == "__main__":
    main()
