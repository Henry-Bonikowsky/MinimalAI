#!/usr/bin/env python3
"""PPO training for sigil ability timing.

Uses SigilSim (vectorized numpy) with rule-based opponent.
Trains SigilNet (~5K params) to learn when to activate abilities.

Usage:
    python -m training.train_sigils --steps 2000000 --n-envs 512 --device cuda
    python -m training.train_sigils --steps 500000 --device cuda --lr 3e-4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli

from .sigil_sim import SigilSim, NUM_SIGIL_ACTIONS, OBS_DIM
from .models.sigil_network import SigilNet, SigilInferenceWrapper


def parse_args():
    p = argparse.ArgumentParser(description="PPO for sigil timing")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=512)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_sigils")
    p.add_argument("--checkpoint-interval", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint .pt file")
    return p.parse_args()


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize
    sim = SigilSim(n_envs=args.n_envs, seed=args.seed)
    model = SigilNet().to(device)

    if args.resume:
        state_dict = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Resumed from: {args.resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    print(f"SigilNet parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Rollout storage
    n = args.n_envs
    T = args.rollout_steps
    obs_buf = np.zeros((T, n, OBS_DIM), dtype=np.float32)
    mask_buf = np.zeros((T, n, NUM_SIGIL_ACTIONS), dtype=np.float32)
    act_buf = np.zeros((T, n, NUM_SIGIL_ACTIONS), dtype=np.float32)
    logp_buf = np.zeros((T, n), dtype=np.float32)
    rew_buf = np.zeros((T, n), dtype=np.float32)
    done_buf = np.zeros((T, n), dtype=np.float32)
    val_buf = np.zeros((T, n), dtype=np.float32)

    # Initial reset
    obs, action_mask = sim.reset()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    total_steps = 0
    num_rollouts = args.steps // (n * T)
    start_time = time.time()

    ep_rewards = []
    ep_lengths = []
    running_reward = 0.0

    for rollout in range(num_rollouts):
        # Collect rollout
        model.eval()
        for t in range(T):
            obs_t = torch.from_numpy(obs).to(device)
            mask_t = torch.from_numpy(action_mask).to(device)

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t, mask_t)

            actions_np = action.cpu().numpy().astype(np.int32)

            obs_buf[t] = obs
            mask_buf[t] = action_mask
            act_buf[t] = actions_np
            logp_buf[t] = log_prob.cpu().numpy()
            val_buf[t] = value.cpu().numpy()

            obs, action_mask, rewards, dones, infos = sim.step(actions_np)

            rew_buf[t] = rewards
            done_buf[t] = dones

            running_reward += rewards.sum()

        total_steps += n * T

        # Compute GAE
        with torch.no_grad():
            next_value = model.get_value(
                torch.from_numpy(obs).to(device),
                torch.from_numpy(action_mask).to(device)
            ).cpu().numpy()

        advantages = np.zeros((T, n), dtype=np.float32)
        lastgae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = val_buf[t + 1]
            nextnonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + args.gamma * next_val * nextnonterminal - val_buf[t]
            lastgae = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgae
            advantages[t] = lastgae

        returns = advantages + val_buf

        # Flatten
        b_obs = torch.from_numpy(obs_buf.reshape(-1, OBS_DIM)).to(device)
        b_mask = torch.from_numpy(mask_buf.reshape(-1, NUM_SIGIL_ACTIONS)).to(device)
        b_act = torch.from_numpy(act_buf.reshape(-1, NUM_SIGIL_ACTIONS)).to(device)
        b_logp = torch.from_numpy(logp_buf.reshape(-1)).to(device)
        b_adv = torch.from_numpy(advantages.reshape(-1)).to(device)
        b_ret = torch.from_numpy(returns.reshape(-1)).to(device)

        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # PPO update
        model.train()
        batch_size = n * T
        assert batch_size >= args.minibatch_size

        for epoch in range(args.epochs):
            indices = np.random.permutation(batch_size)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                if end > batch_size:
                    break
                mb_idx = indices[start:end]

                _, new_logp, entropy, new_val = model.get_action_and_value(
                    b_obs[mb_idx], b_mask[mb_idx], b_act[mb_idx])

                # Policy loss (clipped)
                logratio = new_logp - b_logp[mb_idx]
                ratio = logratio.exp()
                pg_loss1 = -b_adv[mb_idx] * ratio
                pg_loss2 = -b_adv[mb_idx] * torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_val - b_ret[mb_idx]) ** 2).mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        if (rollout + 1) % 5 == 0:
            elapsed = time.time() - start_time
            sps = total_steps / elapsed
            avg_reward = running_reward / (5 * n * T)
            s = sim.get_stats()
            winrate = s['a_winrate']
            ep_len = s['avg_ep_length']
            a_hp = s['a_avg_hp_at_win']
            print(f"[{total_steps:>8,}] sps={sps:.0f} | "
                  f"reward={avg_reward:.3f} | "
                  f"winrate={winrate:.1f}% | ep_len={ep_len:.0f} | hp@win={a_hp:.1f} | "
                  f"pg={pg_loss.item():.4f} | vf={v_loss.item():.4f} | "
                  f"ent={ent_loss.item():.4f}")
            running_reward = 0.0

        # Detailed stats every 20 rollouts
        if (rollout + 1) % 20 == 0:
            print(f"\n--- Stats at {total_steps:,} steps ---")
            print(sim.format_stats())
            print()
            sim.reset_stats()

        # Checkpoint
        if total_steps % args.checkpoint_interval < n * T:
            save_path = checkpoint_dir / f"sigil_model_{total_steps}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  Saved checkpoint: {save_path}")

            # Also export TorchScript
            export_torchscript(model, checkpoint_dir / "sigil_model.pt", device)

    # Final stats + export
    print(f"\n{'='*60}")
    print(f"  FINAL TRAINING REPORT")
    print(f"{'='*60}")
    print(sim.format_stats())
    print()

    final_path = checkpoint_dir / "sigil_model.pt"
    export_torchscript(model, final_path, device)
    print(f"\nTraining complete! Final model: {final_path}")
    print(f"Total steps: {total_steps:,} in {time.time() - start_time:.1f}s")


def export_torchscript(model, path, device):
    """Export SigilNet as TorchScript for Java-side inference.

    Uses a deep copy to avoid disturbing the training model's device placement.
    """
    import copy
    model.eval()
    model_copy = copy.deepcopy(model).cpu()
    wrapper = SigilInferenceWrapper(model_copy)
    dummy_obs = torch.zeros(1, OBS_DIM)
    dummy_mask = torch.ones(1, NUM_SIGIL_ACTIONS)
    traced = torch.jit.trace(wrapper, (dummy_obs, dummy_mask), check_trace=False)
    traced.save(str(path))
    model.to(device)  # ensure original stays on device
    print(f"  Exported TorchScript: {path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
