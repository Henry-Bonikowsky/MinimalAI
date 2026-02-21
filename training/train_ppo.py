#!/usr/bin/env python3
"""PPO training loop using the vanilla MC physics simulator.

Uses SimVecEnv for high-throughput parallel training (thousands of episodes/sec)
without a live MC server. Compatible with the existing CombatNetwork architecture.

Usage:
    python -m training.train_ppo --steps 100000
    python -m training.train_ppo --steps 1000000 --n-envs 1024 --device cuda
    python -m training.train_ppo --resume checkpoints/sim_latest.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from .sim_env import SimVecEnv, OBS_DIM, NUM_ACTIONS
from .models.network import CombatNetwork, GRU_HIDDEN_DIM


def parse_args():
    p = argparse.ArgumentParser(description="PPO training on sim environment")
    p.add_argument("--steps", type=int, default=100_000, help="Total training steps")
    p.add_argument("--n-envs", type=int, default=64, help="Parallel environments")
    p.add_argument("--rollout-steps", type=int, default=256, help="Steps per rollout")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    p.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus")
    p.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    p.add_argument("--ppo-epochs", type=int, default=4, help="PPO optimization epochs")
    p.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size")
    p.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    p.add_argument("--checkpoint-interval", type=int, default=50_000, help="Steps between checkpoints")
    p.add_argument("--log-interval", type=int, default=5, help="Rollouts between log lines")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    p.add_argument("--export", type=str, default=None, help="Export TorchScript model path")
    p.add_argument("--episode-length", type=int, default=1800, help="Max episode ticks")
    return p.parse_args()


class SimRolloutBuffer:
    """Pre-allocated rollout buffer for vectorized sim training."""

    def __init__(self, n_envs: int, rollout_steps: int):
        self.n_envs = n_envs
        self.rollout_steps = rollout_steps
        total = n_envs * rollout_steps

        self.obs = np.zeros((rollout_steps, n_envs, OBS_DIM), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, n_envs, NUM_ACTIONS), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((rollout_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, n_envs), dtype=np.float32)

    def compute_gae(self, last_values: np.ndarray, gamma: float, gae_lambda: float):
        """Compute GAE across all environments."""
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def flatten(self):
        """Flatten (steps, envs, ...) to (steps*envs, ...)."""
        n = self.rollout_steps * self.n_envs
        return {
            "obs": self.obs.reshape(n, OBS_DIM),
            "actions": self.actions.reshape(n, NUM_ACTIONS),
            "log_probs": self.log_probs.reshape(n),
            "advantages": self.advantages.reshape(n),
            "returns": self.returns.reshape(n),
        }


class SimPolicyWrapper(nn.Module):
    """Wraps CombatNetwork for flat obs input (no dict obs).

    The CombatNetwork expects dict observations with separate entity features.
    For the sim env, we pass a flat 320-dim vector and split it internally.
    """

    def __init__(self, network: CombatNetwork):
        super().__init__()
        self.network = network

    def forward(self, flat_obs: torch.Tensor, hidden: torch.Tensor = None):
        """Convert flat obs to dict format and forward through network."""
        obs_dict = self._split_obs(flat_obs)
        logits, value, hidden = self.network(obs_dict, hidden)
        return logits, value, hidden

    def get_action_and_value(self, flat_obs: torch.Tensor, hidden=None, action=None):
        """Get action, log_prob, entropy, value from flat obs."""
        logits, value, hidden = self.forward(flat_obs, hidden)
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs=probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value.squeeze(-1), hidden

    def get_value(self, flat_obs: torch.Tensor, hidden=None):
        """Get value from flat obs."""
        _, value, hidden = self.forward(flat_obs, hidden)
        return value.squeeze(-1), hidden

    def _split_obs(self, flat_obs: torch.Tensor) -> dict:
        """Split flat 320-dim obs into dict format for CombatNetwork.

        Layout (matches ObservationBuilder.java):
        - self_state: [0:38]
        - entity_features: [38:230] -> reshaped to (batch, 8, 24)
        - entity_mask: [230:238] -> (batch, 8)
        - combat_ctx: [238:264] -> (batch, 26)
        - sigil_state: [264:312] -> (batch, 48)
        - env_state: [312:320] -> (batch, 8)
        """
        batch = flat_obs.shape[0] if flat_obs.dim() > 1 else 1
        if flat_obs.dim() == 1:
            flat_obs = flat_obs.unsqueeze(0)

        return {
            "self_state": flat_obs[:, 0:38],
            "entity_features": flat_obs[:, 38:230].reshape(batch, 8, 24),
            "entity_mask": flat_obs[:, 230:238],
            "combat_ctx": flat_obs[:, 238:264],
            "sigil_state": flat_obs[:, 264:312],
            "env_state": flat_obs[:, 312:320],
        }


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Create environments
    print(f"Creating {args.n_envs} parallel environments...")
    vec_env = SimVecEnv(args.n_envs, episode_length=args.episode_length, seed=args.seed)

    # Create network
    network = CombatNetwork()
    wrapper = SimPolicyWrapper(network).to(device)
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=args.lr, eps=1e-5)

    total_steps = 0
    total_episodes = 0
    update_count = 0
    best_reward = -float("inf")

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        network.load_state_dict(ckpt["network_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        total_episodes = ckpt.get("total_episodes", 0)
        update_count = ckpt.get("update_count", 0)

    # Create checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Rollout buffer
    buffer = SimRolloutBuffer(args.n_envs, args.rollout_steps)
    obs = vec_env.reset()

    print(f"Training for {args.steps} steps (rollout={args.rollout_steps}, "
          f"envs={args.n_envs}, batch={args.minibatch_size})")
    print(f"  Effective batch per update: {args.rollout_steps * args.n_envs}")

    start_time = time.time()
    rollout_rewards = []
    rollout_lengths = []

    while total_steps < args.steps:
        # ── Collect rollout ──
        wrapper.eval()
        ep_rewards = []
        ep_lengths = []
        current_ep_rewards = np.zeros(args.n_envs, dtype=np.float32)
        current_ep_lengths = np.zeros(args.n_envs, dtype=np.int32)

        for step in range(args.rollout_steps):
            obs_tensor = torch.from_numpy(obs).to(device)

            with torch.no_grad():
                action, log_prob, _, value, _ = wrapper.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()

            # Step environments
            next_obs, rewards, dones, infos = vec_env.step(action_np.astype(np.int8))

            buffer.obs[step] = obs
            buffer.actions[step] = action_np
            buffer.log_probs[step] = log_prob_np
            buffer.rewards[step] = rewards
            buffer.values[step] = value_np
            buffer.dones[step] = dones

            current_ep_rewards += rewards
            current_ep_lengths += 1

            for i in range(args.n_envs):
                if dones[i]:
                    ep_rewards.append(current_ep_rewards[i])
                    ep_lengths.append(current_ep_lengths[i])
                    current_ep_rewards[i] = 0.0
                    current_ep_lengths[i] = 0
                    total_episodes += 1

            obs = next_obs
            total_steps += args.n_envs

        # Compute last values for GAE
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(device)
            last_values, _ = wrapper.get_value(obs_tensor)
            last_values = last_values.cpu().numpy()

        buffer.compute_gae(last_values, args.gamma, args.gae_lambda)

        # ── PPO update ──
        wrapper.train()
        flat = buffer.flatten()
        n_samples = len(flat["obs"])
        indices = np.arange(n_samples)

        all_pg_losses = []
        all_vf_losses = []
        all_entropy = []
        all_kl = []

        for epoch in range(args.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, args.minibatch_size):
                end = min(start + args.minibatch_size, n_samples)
                mb_idx = indices[start:end]

                mb_obs = torch.from_numpy(flat["obs"][mb_idx]).to(device)
                mb_actions = torch.from_numpy(flat["actions"][mb_idx]).to(device)
                mb_old_log_probs = torch.from_numpy(flat["log_probs"][mb_idx]).to(device)
                mb_advantages = torch.from_numpy(flat["advantages"][mb_idx]).to(device)
                mb_returns = torch.from_numpy(flat["returns"][mb_idx]).to(device)

                # Normalize advantages
                if len(mb_advantages) > 1:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                _, new_log_prob, entropy, new_value, _ = wrapper.get_action_and_value(
                    mb_obs, action=mb_actions
                )

                # Policy loss
                log_ratio = new_log_prob - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                vf_loss = F.mse_loss(new_value, mb_returns)

                # Entropy
                entropy_loss = -entropy.mean()

                loss = pg_loss + args.value_coef * vf_loss + args.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(wrapper.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    all_pg_losses.append(pg_loss.item())
                    all_vf_losses.append(vf_loss.item())
                    all_entropy.append(-entropy_loss.item())
                    all_kl.append(approx_kl)

            # Early stop on high KL
            if all_kl and np.mean(all_kl[-n_samples // args.minibatch_size:]) > 0.02:
                break

        update_count += 1

        # Track episode stats
        rollout_rewards.extend(ep_rewards)
        rollout_lengths.extend(ep_lengths)

        # ── Logging ──
        if update_count % args.log_interval == 0 and rollout_rewards:
            elapsed = time.time() - start_time
            sps = total_steps / max(elapsed, 1)
            mean_r = np.mean(rollout_rewards[-100:])
            mean_l = np.mean(rollout_lengths[-100:]) if rollout_lengths else 0

            print(f"step={total_steps:>8d} | ep={total_episodes:>6d} | "
                  f"reward={mean_r:>7.3f} | len={mean_l:>6.1f} | "
                  f"pg={np.mean(all_pg_losses):.4f} | vf={np.mean(all_vf_losses):.4f} | "
                  f"ent={np.mean(all_entropy):.4f} | kl={np.mean(all_kl):.4f} | "
                  f"sps={sps:.0f}")

            if mean_r > best_reward:
                best_reward = mean_r

        # ── Checkpointing ──
        if total_steps % args.checkpoint_interval < args.rollout_steps * args.n_envs:
            ckpt_path = ckpt_dir / f"sim_{total_steps}.pt"
            _save_checkpoint(wrapper, optimizer, total_steps, total_episodes, update_count, ckpt_path)
            # Also save as latest
            _save_checkpoint(wrapper, optimizer, total_steps, total_episodes, update_count,
                             ckpt_dir / "sim_latest.pt")

    # Final save
    final_path = ckpt_dir / "sim_final.pt"
    _save_checkpoint(wrapper, optimizer, total_steps, total_episodes, update_count, final_path)
    print(f"\nTraining complete. {total_steps} steps, {total_episodes} episodes.")
    print(f"Best mean reward: {best_reward:.3f}")
    print(f"Final checkpoint: {final_path}")

    # Export TorchScript if requested
    if args.export:
        _export_torchscript(wrapper, args.export, device)


def _save_checkpoint(wrapper, optimizer, steps, episodes, updates, path):
    torch.save({
        "network_state_dict": wrapper.network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_steps": steps,
        "total_episodes": episodes,
        "update_count": updates,
    }, path)


def _export_torchscript(wrapper, path, device):
    """Export model as TorchScript for DJL inference."""
    wrapper.eval()

    class InferenceWrapper(nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, self_state, entity_features, entity_mask,
                    combat_ctx, sigil_state, env_state, hidden):
            obs = {
                "self_state": self_state,
                "entity_features": entity_features,
                "entity_mask": entity_mask,
                "combat_ctx": combat_ctx,
                "sigil_state": sigil_state,
                "env_state": env_state,
            }
            logits, value, new_hidden = self.network(obs, hidden)
            action_probs = torch.sigmoid(logits)
            return action_probs, value, new_hidden

    inf_wrapper = InferenceWrapper(wrapper.network).to(device)
    inf_wrapper.eval()

    example_inputs = (
        torch.zeros(1, 38, device=device),
        torch.zeros(1, 8, 24, device=device),
        torch.zeros(1, 8, device=device),
        torch.zeros(1, 26, device=device),
        torch.zeros(1, 48, device=device),
        torch.zeros(1, 8, device=device),
        torch.zeros(1, 1, GRU_HIDDEN_DIM, device=device),
    )

    scripted = torch.jit.trace(inf_wrapper, example_inputs)
    scripted.save(path)
    print(f"Exported TorchScript model to {path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
