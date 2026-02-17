"""Main training script for MinimalAI combat AI (v2 - 1.8 PvP + self-play).

Usage:
    python train.py                          # Train with defaults
    python train.py --total-steps 5000000    # Train longer
    python train.py --self-play              # Enable self-play training
    python train.py --resume checkpoints/latest.pt
"""

import os
import sys
import time
import copy
import argparse
import json
import logging
import numpy as np
import torch

from config import TrainingConfig
from combat_sim.env import CombatEnv
from models.network import CombatNetwork, NUM_ACTIONS
from models.ppo import PPO


def make_env(config: TrainingConfig, seed: int = None) -> CombatEnv:
    """Create the combat environment."""
    return CombatEnv(
        num_enemies=config.num_enemies,
        num_allies=config.num_allies,
        episode_length=config.episode_length,
        domain_randomization=config.domain_randomization,
        self_play=config.self_play,
        seed=seed or config.seed,
    )


def linear_schedule(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(1.0, progress)


class OpponentPool:
    """Historical opponent pool for self-play training."""

    def __init__(self, max_size: int = 5, device: str = "cpu"):
        self.max_size = max_size
        self.device = torch.device(device)
        self.opponents: list[CombatNetwork] = []

    def add_snapshot(self, network: CombatNetwork):
        """Add a copy of the current network to the pool."""
        snapshot = CombatNetwork()
        snapshot.load_state_dict(copy.deepcopy(network.state_dict()))
        snapshot.eval()
        snapshot.to(self.device)
        self.opponents.append(snapshot)
        if len(self.opponents) > self.max_size:
            self.opponents.pop(0)

    def sample_opponent(self, rng: np.random.Generator) -> CombatNetwork:
        """Sample a random opponent from the pool."""
        if not self.opponents:
            return None
        idx = rng.integers(0, len(self.opponents))
        return self.opponents[idx]

    def __len__(self):
        return len(self.opponents)


def self_play_step(env: CombatEnv, opponent: CombatNetwork, device: torch.device,
                   opp_hiddens: dict) -> dict:
    """Get opponent actions for all enemy agents in self-play mode."""
    actions = {}
    for enemy in env.enemies:
        if not enemy.is_alive:
            continue
        obs = env.get_opponent_obs(enemy.agent_id)
        if obs is None:
            continue

        # Initialize hidden state for new enemies
        if enemy.agent_id not in opp_hiddens:
            opp_hiddens[enemy.agent_id] = opponent.init_hidden(1).to(device)

        obs_tensor = {
            k: torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
            for k, v in obs.items()
        }
        mask_tensor = torch.tensor(
            env.get_action_mask(enemy), dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _, hidden = opponent.get_action_and_value(
                obs_tensor, opp_hiddens[enemy.agent_id], mask_tensor
            )
        opp_hiddens[enemy.agent_id] = hidden
        actions[enemy.agent_id] = action.squeeze(0).cpu().numpy()

    return actions


def train(config: TrainingConfig, resume_path: str = None, log_episodes: int = 0):
    """Run PPO training with optional self-play."""
    print("=" * 60)
    print("MinimalAI Combat AI Training (v2 - 1.8 PvP)")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Total steps: {config.total_timesteps:,}")
    print(f"Enemies: {config.num_enemies}, Allies: {config.num_allies}")
    print(f"Self-play: {config.self_play}")
    print(f"Rollout steps: {config.rollout_steps}")
    print(f"Learning rate: {config.lr}")
    print("=" * 60)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = make_env(config)
    network = CombatNetwork()
    total_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {total_params:,}")

    ppo = PPO(
        network=network,
        lr=config.lr,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        entropy_coef=config.entropy_coef_start,
        value_coef=config.value_coef,
        max_grad_norm=config.max_grad_norm,
        ppo_epochs=config.ppo_epochs,
        minibatch_size=config.minibatch_size,
        target_kl=config.target_kl,
        device=config.device,
    )

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ppo.load_checkpoint(resume_path)

    # Self-play opponent pool
    opponent_pool = OpponentPool(max_size=5, device=config.device)
    if config.self_play:
        opponent_pool.add_snapshot(network)
        print(f"Self-play enabled: opponent pool initialized with {len(opponent_pool)} snapshot(s)")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    log_path = os.path.join(config.log_dir, "training_log.jsonl")
    log_file = open(log_path, "a")

    saved_skills = set()
    num_updates = config.total_timesteps // config.rollout_steps
    start_time = time.time()
    best_mean_reward = -float("inf")

    rng = np.random.default_rng(config.seed)

    print(f"\nStarting training ({num_updates} updates)...\n")

    for update in range(1, num_updates + 1):
        progress = ppo.total_steps / config.total_timesteps
        ppo.entropy_coef = linear_schedule(
            config.entropy_coef_start, config.entropy_coef_end, progress
        )
        lr = linear_schedule(config.lr, config.lr * 0.1, progress)
        for param_group in ppo.optimizer.param_groups:
            param_group["lr"] = lr

        # Self-play: select opponent for this rollout
        opponent = None
        if config.self_play and len(opponent_pool) > 0:
            opponent = opponent_pool.sample_opponent(rng)

        # Collect rollout (with self-play if enabled)
        rollout_stats = collect_rollout_selfplay(
            ppo, env, config.rollout_steps, opponent, config.device,
            log_episodes=log_episodes if update == 1 else 0,
        )

        train_stats = ppo.train_on_buffer()

        # Self-play: update opponent pool every N updates
        if config.self_play and update % config.opponent_update_interval == 0:
            opponent_pool.add_snapshot(network)

        elapsed = time.time() - start_time
        fps = ppo.total_steps / elapsed if elapsed > 0 else 0

        if update % config.log_interval == 0 or update == 1:
            mean_reward = rollout_stats["mean_reward"]
            mean_length = rollout_stats["mean_length"]

            print(
                f"Update {update:>5}/{num_updates} | "
                f"Steps: {ppo.total_steps:>8,} | "
                f"Ep: {ppo.total_episodes:>5} | "
                f"Reward: {mean_reward:>8.3f} | "
                f"Len: {mean_length:>6.0f} | "
                f"PLoss: {train_stats['policy_loss']:>8.4f} | "
                f"VLoss: {train_stats['value_loss']:>8.4f} | "
                f"Ent: {train_stats['entropy']:>6.3f} | "
                f"Clip: {train_stats['clip_fraction']:>5.3f} | "
                f"FPS: {fps:>6.0f}" +
                (f" | Pool: {len(opponent_pool)}" if config.self_play else "")
            )

            log_entry = {
                "update": update,
                "total_steps": ppo.total_steps,
                "total_episodes": ppo.total_episodes,
                "mean_reward": mean_reward,
                "mean_length": mean_length,
                "fps": fps,
                "lr": lr,
                "entropy_coef": ppo.entropy_coef,
                "opponent_pool_size": len(opponent_pool),
                **train_stats,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                ppo.save_checkpoint(
                    os.path.join(config.checkpoint_dir, "best.pt"),
                    metadata={"mean_reward": mean_reward, "steps": ppo.total_steps},
                )

        if ppo.total_steps % config.checkpoint_interval < config.rollout_steps:
            ppo.save_checkpoint(
                os.path.join(config.checkpoint_dir, "latest.pt"),
                metadata={"steps": ppo.total_steps},
            )

        for skill_name, skill_step in config.skill_checkpoints.items():
            if skill_name not in saved_skills and ppo.total_steps >= skill_step:
                path = os.path.join(config.checkpoint_dir, f"skill_{skill_name}.pt")
                ppo.save_checkpoint(path, metadata={
                    "skill_level": skill_name,
                    "steps": ppo.total_steps,
                    "mean_reward": rollout_stats["mean_reward"],
                })
                saved_skills.add(skill_name)
                print(f"  >> Saved skill checkpoint: {skill_name} at step {ppo.total_steps:,}")

    # Save remaining skill checkpoints
    for skill_name in config.skill_checkpoints:
        if skill_name not in saved_skills:
            path = os.path.join(config.checkpoint_dir, f"skill_{skill_name}.pt")
            ppo.save_checkpoint(path, metadata={
                "skill_level": skill_name,
                "steps": ppo.total_steps,
                "mean_reward": best_mean_reward,
            })
            saved_skills.add(skill_name)
            print(f"  >> Saved skill checkpoint: {skill_name} at step {ppo.total_steps:,}")

    ppo.save_checkpoint(
        os.path.join(config.checkpoint_dir, "final.pt"),
        metadata={"steps": ppo.total_steps, "mean_reward": best_mean_reward},
    )

    log_file.close()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Total steps: {ppo.total_steps:,}")
    print(f"Total episodes: {ppo.total_episodes:,}")
    print(f"Best mean reward: {best_mean_reward:.3f}")
    print(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Average FPS: {ppo.total_steps / elapsed:.0f}")
    print(f"{'=' * 60}")


def collect_rollout_selfplay(ppo: PPO, env: CombatEnv, num_steps: int,
                             opponent: CombatNetwork = None,
                             device: str = "cpu",
                             log_episodes: int = 0) -> dict:
    """Collect rollout with self-play opponent support."""
    ppo.network.eval()
    device_t = torch.device(device)

    obs, info = env.reset()

    # Lazily create buffer with correct observation shapes
    if ppo.buffer is None:
        from models.ppo import RolloutBuffer
        obs_shapes = {k: v.shape for k, v in obs.items()}
        ppo.buffer = RolloutBuffer(num_steps, obs_shapes)
    ppo.buffer.clear()
    hidden = ppo.network.init_hidden(batch_size=1).to(device_t)
    action_mask = env.get_action_mask()

    opp_hiddens = {}
    episode_rewards = []
    episode_lengths = []
    current_ep_reward = 0.0
    current_ep_length = 0
    logged_episodes = 0
    combat_logger = logging.getLogger("combat_sim")

    for step in range(num_steps):
        # Self-play: get opponent actions
        if opponent is not None and env.self_play:
            opp_actions = self_play_step(env, opponent, device_t, opp_hiddens)
            for agent_id, act in opp_actions.items():
                env.set_opponent_action(agent_id, act)

        obs_tensor = {
            k: torch.tensor(v, dtype=torch.float32, device=device_t).unsqueeze(0)
            for k, v in obs.items()
        }
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device_t).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, entropy, value, hidden = ppo.network.get_action_and_value(
                obs_tensor, hidden, mask_tensor
            )

        action_np = action.squeeze(0).cpu().numpy()
        log_prob_val = log_prob.item()
        value_val = value.item()

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        ppo.buffer.add(
            obs=obs,
            action=action_np,
            action_mask=action_mask,
            log_prob=log_prob_val,
            reward=reward,
            value=value_val,
            done=done,
            hidden=hidden.detach().cpu().numpy(),
        )

        current_ep_reward += reward
        current_ep_length += 1
        ppo.total_steps += 1

        if done:
            episode_rewards.append(current_ep_reward)
            episode_lengths.append(current_ep_length)
            current_ep_reward = 0.0
            current_ep_length = 0
            ppo.total_episodes += 1
            logged_episodes += 1

            # Disable debug logging after N episodes
            if log_episodes > 0 and logged_episodes >= log_episodes:
                combat_logger.setLevel(logging.INFO)

            obs, info = env.reset()
            hidden = ppo.network.init_hidden(batch_size=1).to(device_t)
            action_mask = env.get_action_mask()
            opp_hiddens = {}
        else:
            obs = next_obs
            action_mask = env.get_action_mask()

    with torch.no_grad():
        obs_tensor = {
            k: torch.tensor(v, dtype=torch.float32, device=device_t).unsqueeze(0)
            for k, v in obs.items()
        }
        last_value, _ = ppo.network.get_value(obs_tensor, hidden)
        last_value = last_value.item()

    ppo.buffer.compute_gae(last_value, ppo.gamma, ppo.gae_lambda)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
        "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train MinimalAI Combat AI (v2)")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-enemies", type=int, default=1)
    parser.add_argument("--num-allies", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--no-domain-randomization", action="store_true")
    parser.add_argument("--log-episodes", type=int, default=0,
                        help="Log N detailed episodes at start (DEBUG level)")
    args = parser.parse_args()

    # Set up episode logging
    if args.log_episodes > 0:
        combat_logger = logging.getLogger("combat_sim")
        combat_logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        combat_logger.addHandler(handler)

    config = TrainingConfig(
        total_timesteps=args.total_steps,
        num_enemies=args.num_enemies,
        num_allies=args.num_allies,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        domain_randomization=not args.no_domain_randomization,
        self_play=args.self_play,
        device="cpu",
    )

    train(config, resume_path=args.resume, log_episodes=args.log_episodes)


if __name__ == "__main__":
    main()
