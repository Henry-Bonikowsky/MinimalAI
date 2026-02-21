#!/usr/bin/env python3
"""Fast PPO training with self-play using vectorized simulator.

Key improvements over train_ppo.py:
- Numpy-vectorized sim (all N envs run simultaneously)
- Self-play: agent trains against a frozen copy of itself
- Opponent updated every K rollouts for stable training
- Larger effective batches for stable gradients

Usage:
    python -m training.fast_train --steps 500000
    python -m training.fast_train --steps 5000000 --n-envs 512 --device cuda
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from .vec_sim import VecPvPSim, OBS_DIM, NUM_ACTIONS
from .models.network import CombatNetwork, GRU_HIDDEN_DIM


def parse_args():
    p = argparse.ArgumentParser(description="Fast PPO with self-play")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint-interval", type=int, default=100_000)
    p.add_argument("--log-interval", type=int, default=2)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--export", type=str, default=None)
    p.add_argument("--episode-length", type=int, default=1800)
    # Self-play
    p.add_argument("--opponent-update-interval", type=int, default=10,
                    help="Rollouts between opponent policy updates")
    p.add_argument("--self-play", action="store_true", default=True,
                    help="Use self-play (agent vs frozen copy)")
    p.add_argument("--no-self-play", dest="self_play", action="store_false",
                    help="Use rule-based opponent instead")
    return p.parse_args()


class PolicyWrapper(nn.Module):
    """Wraps CombatNetwork for flat obs input."""

    def __init__(self, network: CombatNetwork):
        super().__init__()
        self.network = network

    def forward(self, flat_obs, hidden=None):
        obs_dict = self._split_obs(flat_obs)
        logits, value, hidden = self.network(obs_dict, hidden)
        return logits, value, hidden

    def get_action_and_value(self, flat_obs, hidden=None, action=None):
        logits, value, hidden = self.forward(flat_obs, hidden)
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs=probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value.squeeze(-1), hidden

    def get_value(self, flat_obs, hidden=None):
        _, value, hidden = self.forward(flat_obs, hidden)
        return value.squeeze(-1), hidden

    def sample_actions(self, flat_obs):
        """Fast action sampling (no grad, no value)."""
        with torch.no_grad():
            logits, _, _ = self.forward(flat_obs)
            probs = torch.sigmoid(logits)
            actions = torch.bernoulli(probs)
        return actions

    def _split_obs(self, flat_obs):
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


def rule_based_opponent(sim: VecPvPSim, rng: np.random.Generator) -> np.ndarray:
    """Simple rule-based opponent actions."""
    n = sim.n
    actions = np.zeros((n, NUM_ACTIONS), dtype=np.int8)

    # Face player A
    dx = sim.ax - sim.bx
    dz = sim.az - sim.bz
    sim.byaw[:] = np.degrees(np.arctan2(-dx, dz))

    dist = np.sqrt(dx**2 + dz**2)

    # Movement
    far = dist > 5.0
    mid = (dist > 2.5) & ~far
    close = ~far & ~mid

    # Forward when far/mid
    actions[far, 0] = 1  # forward
    actions[mid, 0] = (rng.random(n) < 0.6)[mid].astype(np.int8)

    # Sprint when far
    actions[far, 6] = 1  # sprint

    # Strafe when close
    strafe_dir = rng.choice([2, 3], n)  # left or right
    actions[close, strafe_dir[close]] = 1

    # Attack when in range
    can_attack = dist <= 3.0
    actions[can_attack, 7] = (rng.random(n) < 0.5)[can_attack].astype(np.int8)

    # Occasional jump
    actions[:, 4] = (rng.random(n) < 0.08).astype(np.int8)

    return actions


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # Vectorized sim
    print(f"Creating vectorized sim with {args.n_envs} parallel environments...")
    sim = VecPvPSim(args.n_envs, episode_length=args.episode_length, seed=args.seed)

    # Networks
    network = CombatNetwork()
    policy = PolicyWrapper(network).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    # Self-play opponent (frozen copy)
    opponent_net = CombatNetwork()
    opponent = PolicyWrapper(opponent_net).to(device)
    opponent.load_state_dict(policy.state_dict())
    opponent.eval()

    total_steps = 0
    total_episodes = 0
    update_count = 0
    best_reward = -float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        network.load_state_dict(ckpt["network_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        total_episodes = ckpt.get("total_episodes", 0)
        update_count = ckpt.get("update_count", 0)
        # Sync opponent
        opponent.load_state_dict(policy.state_dict())

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate rollout buffer
    R = args.rollout_steps
    N = args.n_envs
    buf_obs = np.zeros((R, N, OBS_DIM), dtype=np.float32)
    buf_actions = np.zeros((R, N, NUM_ACTIONS), dtype=np.float32)
    buf_log_probs = np.zeros((R, N), dtype=np.float32)
    buf_rewards = np.zeros((R, N), dtype=np.float32)
    buf_values = np.zeros((R, N), dtype=np.float32)
    buf_dones = np.zeros((R, N), dtype=np.float32)
    buf_advantages = np.zeros((R, N), dtype=np.float32)
    buf_returns = np.zeros((R, N), dtype=np.float32)

    obs = sim.reset_all()
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(N)
    current_ep_lengths = np.zeros(N, dtype=np.int32)

    mode = "self-play" if args.self_play else "rule-based"
    print(f"Training for {args.steps} steps ({mode} opponent)")
    print(f"  rollout={R}, envs={N}, batch_per_update={R*N}")
    print(f"  opponent update every {args.opponent_update_interval} rollouts")
    start_time = time.time()

    while total_steps < args.steps:
        # ── Collect rollout ──
        policy.eval()

        for step in range(R):
            obs_tensor = torch.from_numpy(obs).to(device)

            with torch.no_grad():
                action, log_prob, _, value, _ = policy.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy().astype(np.int8)
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()

            # Opponent actions
            if args.self_play:
                b_obs = sim.get_b_obs()
                b_obs_tensor = torch.from_numpy(b_obs).to(device)
                b_actions = opponent.sample_actions(b_obs_tensor).cpu().numpy().astype(np.int8)
                # Opponent faces player A
                dx = sim.ax - sim.bx
                dz = sim.az - sim.bz
                sim.byaw[:] = np.degrees(np.arctan2(-dx, dz))
            else:
                b_actions = rule_based_opponent(sim, rng)

            # Step sim
            next_obs, rewards, dones, infos = sim.step(action_np, b_actions)

            buf_obs[step] = obs
            buf_actions[step] = action_np.astype(np.float32)
            buf_log_probs[step] = log_prob_np
            buf_rewards[step] = rewards
            buf_values[step] = value_np
            buf_dones[step] = dones.astype(np.float32)

            current_ep_rewards += rewards
            current_ep_lengths += 1

            for i in range(N):
                if dones[i]:
                    ep_rewards.append(float(current_ep_rewards[i]))
                    ep_lengths.append(int(current_ep_lengths[i]))
                    current_ep_rewards[i] = 0.0
                    current_ep_lengths[i] = 0
                    total_episodes += 1

            obs = next_obs
            total_steps += N

        # GAE
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).to(device)
            last_values, _ = policy.get_value(obs_tensor)
            last_values = last_values.cpu().numpy()

        last_gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(R)):
            next_val = last_values if t == R - 1 else buf_values[t + 1]
            non_terminal = 1.0 - buf_dones[t]
            delta = buf_rewards[t] + args.gamma * next_val * non_terminal - buf_values[t]
            last_gae = delta + args.gamma * args.gae_lambda * non_terminal * last_gae
            buf_advantages[t] = last_gae
        buf_returns[:] = buf_advantages + buf_values

        # ── PPO update ──
        policy.train()
        n_samples = R * N
        flat_obs = buf_obs.reshape(n_samples, OBS_DIM)
        flat_actions = buf_actions.reshape(n_samples, NUM_ACTIONS)
        flat_log_probs = buf_log_probs.reshape(n_samples)
        flat_advantages = buf_advantages.reshape(n_samples)
        flat_returns = buf_returns.reshape(n_samples)
        indices = np.arange(n_samples)

        all_pg = []
        all_vf = []
        all_ent = []
        all_kl = []

        for epoch in range(args.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, args.minibatch_size):
                end = min(start + args.minibatch_size, n_samples)
                mb = indices[start:end]

                mb_obs = torch.from_numpy(flat_obs[mb]).to(device)
                mb_act = torch.from_numpy(flat_actions[mb]).to(device)
                mb_old_lp = torch.from_numpy(flat_log_probs[mb]).to(device)
                mb_adv = torch.from_numpy(flat_advantages[mb]).to(device)
                mb_ret = torch.from_numpy(flat_returns[mb]).to(device)

                if len(mb_adv) > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                _, new_lp, entropy, new_val, _ = policy.get_action_and_value(
                    mb_obs, action=mb_act
                )

                log_ratio = new_lp - mb_old_lp
                ratio = torch.exp(log_ratio)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()
                vf_loss = F.mse_loss(new_val, mb_ret)
                ent_loss = -entropy.mean()

                loss = pg_loss + args.value_coef * vf_loss + args.entropy_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    all_pg.append(pg_loss.item())
                    all_vf.append(vf_loss.item())
                    all_ent.append(-ent_loss.item())
                    all_kl.append(approx_kl)

            if all_kl and np.mean(all_kl[-max(1, n_samples // args.minibatch_size):]) > 0.02:
                break

        update_count += 1

        # Update opponent periodically
        if args.self_play and update_count % args.opponent_update_interval == 0:
            opponent.load_state_dict(policy.state_dict())
            opponent.eval()

        # ── Logging ──
        if update_count % args.log_interval == 0 and ep_rewards:
            elapsed = time.time() - start_time
            sps = total_steps / max(elapsed, 1)
            recent = ep_rewards[-200:]
            mean_r = np.mean(recent)
            mean_l = np.mean(ep_lengths[-200:]) if ep_lengths else 0
            win_rate = sum(1 for r in recent if r > 0) / len(recent)

            print(f"step={total_steps:>8d} | ep={total_episodes:>6d} | "
                  f"r={mean_r:>7.3f} | len={mean_l:>6.1f} | "
                  f"win={win_rate:>5.1%} | "
                  f"pg={np.mean(all_pg):.4f} | vf={np.mean(all_vf):.4f} | "
                  f"ent={np.mean(all_ent):.3f} | kl={np.mean(all_kl):.4f} | "
                  f"sps={sps:.0f}")

            if mean_r > best_reward:
                best_reward = mean_r

        # ── Checkpoint ──
        if total_steps % args.checkpoint_interval < R * N:
            path = ckpt_dir / f"fast_{total_steps}.pt"
            _save(policy, optimizer, total_steps, total_episodes, update_count, path)
            _save(policy, optimizer, total_steps, total_episodes, update_count,
                  ckpt_dir / "fast_latest.pt")

    # Final save
    final_path = ckpt_dir / "fast_final.pt"
    _save(policy, optimizer, total_steps, total_episodes, update_count, final_path)
    print(f"\nTraining complete. {total_steps} steps, {total_episodes} episodes.")
    print(f"Best mean reward: {best_reward:.3f}")

    if args.export:
        _export(policy, args.export, device)


def _save(wrapper, optimizer, steps, episodes, updates, path):
    torch.save({
        "network_state_dict": wrapper.network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_steps": steps,
        "total_episodes": episodes,
        "update_count": updates,
    }, path)


def _export(wrapper, path, device):
    wrapper.eval()

    class Inf(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.network = net

        def forward(self, ss, ef, em, cc, sig, env, h):
            obs = {"self_state": ss, "entity_features": ef, "entity_mask": em,
                   "combat_ctx": cc, "sigil_state": sig, "env_state": env}
            logits, value, nh = self.network(obs, h)
            return torch.sigmoid(logits), value, nh

    inf = Inf(wrapper.network).to(device).eval()
    inputs = (
        torch.zeros(1, 38, device=device),
        torch.zeros(1, 8, 24, device=device),
        torch.zeros(1, 8, device=device),
        torch.zeros(1, 26, device=device),
        torch.zeros(1, 48, device=device),
        torch.zeros(1, 8, device=device),
        torch.zeros(1, 1, GRU_HIDDEN_DIM, device=device),
    )
    scripted = torch.jit.trace(inf, inputs)
    scripted.save(path)
    print(f"Exported TorchScript model to {path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
