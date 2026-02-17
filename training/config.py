"""Training configuration for MinimalAI combat AI (v2)."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Environment
    num_enemies: int = 1
    num_allies: int = 0
    episode_length: int = 600  # 30 seconds at 20 TPS - forces engagement
    domain_randomization: bool = True
    self_play: bool = False

    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.01  # higher floor prevents policy collapse
    entropy_decay_steps: int = 500_000
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 128
    target_kl: float = 0.02

    # Rollout
    rollout_steps: int = 2048
    total_timesteps: int = 2_000_000

    # Self-play
    opponent_update_interval: int = 50  # copy weights every N updates

    # Checkpoints
    checkpoint_interval: int = 50_000
    checkpoint_dir: str = "checkpoints"
    skill_checkpoints: dict = None

    # Logging
    log_interval: int = 5
    log_dir: str = "logs"

    # Device
    device: str = "cpu"

    # Seed
    seed: int = 42

    def __post_init__(self):
        if self.skill_checkpoints is None:
            total = self.total_timesteps
            self.skill_checkpoints = {
                "novice": int(total * 0.10),
                "apprentice": int(total * 0.25),
                "fighter": int(total * 0.50),
                "warrior": int(total * 0.75),
                "master": int(total * 1.00),
            }


@dataclass
class EvalConfig:
    num_eval_episodes: int = 20
    num_enemies: int = 1
    render: bool = False
    model_path: str = ""
