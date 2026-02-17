"""Main training script for MinimalAI combat AI.

Usage:
    python train.py                          # Train with defaults
    python train.py --total-steps 5000000    # Train longer
    python train.py --num-enemies 3          # Train against 3 enemies
    python train.py --resume checkpoints/latest.pt  # Resume from checkpoint
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch

from config import TrainingConfig
from combat_sim.env import CombatEnv
from models.network import CombatNetwork
from models.ppo import PPO


def make_env(config: TrainingConfig, seed: int = None) -> CombatEnv:
    """Create the combat environment."""
    return CombatEnv(
        num_enemies=config.num_enemies,
        num_allies=config.num_allies,
        episode_length=config.episode_length,
        domain_randomization=config.domain_randomization,
        seed=seed or config.seed,
    )


def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linear interpolation from start to end based on progress [0, 1]."""
    return start + (end - start) * min(1.0, progress)


def train(config: TrainingConfig, resume_path: str = None):
    """Run PPO training."""
    print("=" * 60)
    print("MinimalAI Combat AI Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Total steps: {config.total_timesteps:,}")
    print(f"Enemies: {config.num_enemies}, Allies: {config.num_allies}")
    print(f"Rollout steps: {config.rollout_steps}")
    print(f"Minibatch size: {config.minibatch_size}")
    print(f"PPO epochs: {config.ppo_epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Gamma: {config.gamma}")
    print("=" * 60)

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create environment
    env = make_env(config)

    # Create network and PPO trainer
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

    # Resume from checkpoint if provided
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ppo.load_checkpoint(resume_path)

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Training log
    log_path = os.path.join(config.log_dir, "training_log.jsonl")
    log_file = open(log_path, "a")

    # Track skill checkpoint saves
    saved_skills = set()

    # Training loop
    num_updates = config.total_timesteps // config.rollout_steps
    start_time = time.time()
    best_mean_reward = -float("inf")

    print(f"\nStarting training ({num_updates} updates)...\n")

    for update in range(1, num_updates + 1):
        # Update entropy coefficient (decay schedule)
        progress = ppo.total_steps / config.total_timesteps
        ppo.entropy_coef = linear_schedule(
            config.entropy_coef_start, config.entropy_coef_end, progress
        )

        # Update learning rate (linear decay)
        lr = linear_schedule(config.lr, config.lr * 0.1, progress)
        for param_group in ppo.optimizer.param_groups:
            param_group["lr"] = lr

        # Collect rollout
        rollout_stats = ppo.collect_rollout(env, config.rollout_steps)

        # Train on buffer
        train_stats = ppo.train_on_buffer()

        # Logging
        elapsed = time.time() - start_time
        fps = ppo.total_steps / elapsed if elapsed > 0 else 0

        if update % config.log_interval == 0 or update == 1:
            mean_reward = rollout_stats["mean_reward"]
            mean_length = rollout_stats["mean_length"]

            print(
                f"Update {update:>5}/{num_updates} | "
                f"Steps: {ppo.total_steps:>8,} | "
                f"Episodes: {ppo.total_episodes:>5} | "
                f"Reward: {mean_reward:>8.3f} | "
                f"Length: {mean_length:>6.0f} | "
                f"Policy Loss: {train_stats['policy_loss']:>8.4f} | "
                f"Value Loss: {train_stats['value_loss']:>8.4f} | "
                f"Entropy: {train_stats['entropy']:>6.3f} | "
                f"Clip: {train_stats['clip_fraction']:>5.3f} | "
                f"KL: {train_stats['approx_kl']:>6.4f} | "
                f"FPS: {fps:>6.0f} | "
                f"LR: {lr:.2e}"
            )

            # JSON log
            log_entry = {
                "update": update,
                "total_steps": ppo.total_steps,
                "total_episodes": ppo.total_episodes,
                "mean_reward": mean_reward,
                "mean_length": mean_length,
                "fps": fps,
                "lr": lr,
                "entropy_coef": ppo.entropy_coef,
                **train_stats,
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            # Track best reward
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                ppo.save_checkpoint(
                    os.path.join(config.checkpoint_dir, "best.pt"),
                    metadata={"mean_reward": mean_reward, "steps": ppo.total_steps},
                )

        # Regular checkpoint
        if ppo.total_steps % config.checkpoint_interval < config.rollout_steps:
            ppo.save_checkpoint(
                os.path.join(config.checkpoint_dir, "latest.pt"),
                metadata={"steps": ppo.total_steps},
            )

        # Skill level checkpoints
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

    # Save any remaining skill checkpoints (handles rounding at end of training)
    for skill_name, skill_step in config.skill_checkpoints.items():
        if skill_name not in saved_skills:
            path = os.path.join(config.checkpoint_dir, f"skill_{skill_name}.pt")
            ppo.save_checkpoint(path, metadata={
                "skill_level": skill_name,
                "steps": ppo.total_steps,
                "mean_reward": best_mean_reward,
            })
            saved_skills.add(skill_name)
            print(f"  >> Saved skill checkpoint: {skill_name} at step {ppo.total_steps:,}")

    # Final save
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
    print(f"Checkpoints saved in: {config.checkpoint_dir}/")
    print(f"Log saved to: {log_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Train MinimalAI Combat AI")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-enemies", type=int, default=1)
    parser.add_argument("--num-allies", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-domain-randomization", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig(
        total_timesteps=args.total_steps,
        num_enemies=args.num_enemies,
        num_allies=args.num_allies,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        domain_randomization=not args.no_domain_randomization,
        device="cpu",  # GPU is occupied
    )

    train(config, resume_path=args.resume)


if __name__ == "__main__":
    main()
