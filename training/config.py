"""Training configuration for MinimalAI combat AI."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Environment
    num_enemies: int = 1
    num_allies: int = 0
    episode_length: int = 1800  # 90 seconds at 20 TPS
    domain_randomization: bool = True

    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.005
    entropy_decay_steps: int = 500_000
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 128
    target_kl: float = 0.02

    # Rollout
    rollout_steps: int = 2048  # steps per rollout
    total_timesteps: int = 2_000_000  # total training steps

    # Checkpoints (5 difficulty levels)
    checkpoint_interval: int = 50_000  # save every N steps
    checkpoint_dir: str = "checkpoints"
    skill_checkpoints: dict = None  # populated during training

    # Logging
    log_interval: int = 5  # log every N updates
    log_dir: str = "logs"

    # Device - CPU only (GPU is occupied)
    device: str = "cpu"

    # Seed
    seed: int = 42

    def __post_init__(self):
        if self.skill_checkpoints is None:
            # 5 skill levels at percentage milestones
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
    """Config for evaluating trained models."""
    num_eval_episodes: int = 20
    num_enemies: int = 1
    render: bool = False
    model_path: str = ""
