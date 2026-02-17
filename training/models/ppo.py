"""PPO (Proximal Policy Optimization) with GAE.

Implements the clipped surrogate objective with:
- Generalized Advantage Estimation (GAE)
- Entropy bonus (decaying)
- Value function clipping
- Gradient clipping
- Rollout buffer for batch training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from .network import CombatNetwork, NUM_ACTIONS, GRU_HIDDEN_DIM


@dataclass
class RolloutBuffer:
    """Stores a batch of rollout data for PPO training."""
    obs_self_state: list = field(default_factory=list)
    obs_entity_features: list = field(default_factory=list)
    obs_entity_mask: list = field(default_factory=list)
    obs_combat_ctx: list = field(default_factory=list)
    obs_sigil_state: list = field(default_factory=list)
    obs_env_state: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    action_masks: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    hiddens: list = field(default_factory=list)

    # Computed after rollout
    advantages: np.ndarray = None
    returns: np.ndarray = None

    def add(self, obs: dict, action, action_mask, log_prob, reward, value, done, hidden):
        self.obs_self_state.append(obs["self_state"])
        self.obs_entity_features.append(obs["entity_features"])
        self.obs_entity_mask.append(obs["entity_mask"])
        self.obs_combat_ctx.append(obs["combat_ctx"])
        self.obs_sigil_state.append(obs["sigil_state"])
        self.obs_env_state.append(obs["env_state"])
        self.actions.append(action)
        self.action_masks.append(action_mask)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.hiddens.append(hidden)

    def clear(self):
        for attr in [
            'obs_self_state', 'obs_entity_features', 'obs_entity_mask',
            'obs_combat_ctx', 'obs_sigil_state', 'obs_env_state',
            'actions', 'action_masks', 'log_probs', 'rewards', 'values',
            'dones', 'hiddens',
        ]:
            getattr(self, attr).clear()
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.advantages = advantages
        self.returns = advantages + values

    def get_batches(self, batch_size: int, device: torch.device):
        """Yield minibatches for training."""
        n = len(self.rewards)
        indices = np.random.permutation(n)

        # Convert all data to tensors
        all_obs = {
            "self_state": torch.tensor(np.array(self.obs_self_state), dtype=torch.float32, device=device),
            "entity_features": torch.tensor(np.array(self.obs_entity_features), dtype=torch.float32, device=device),
            "entity_mask": torch.tensor(np.array(self.obs_entity_mask), dtype=torch.float32, device=device),
            "combat_ctx": torch.tensor(np.array(self.obs_combat_ctx), dtype=torch.float32, device=device),
            "sigil_state": torch.tensor(np.array(self.obs_sigil_state), dtype=torch.float32, device=device),
            "env_state": torch.tensor(np.array(self.obs_env_state), dtype=torch.float32, device=device),
        }
        all_actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        all_action_masks = torch.tensor(np.array(self.action_masks), dtype=torch.float32, device=device)
        all_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        all_advantages = torch.tensor(self.advantages, dtype=torch.float32, device=device)
        all_returns = torch.tensor(self.returns, dtype=torch.float32, device=device)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            batch_obs = {k: v[idx] for k, v in all_obs.items()}
            yield (
                batch_obs,
                all_actions[idx],
                all_action_masks[idx],
                all_log_probs[idx],
                all_advantages[idx],
                all_returns[idx],
            )


class PPO:
    """PPO trainer for the combat network."""

    def __init__(
        self,
        network: CombatNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 128,
        target_kl: float = 0.02,
        device: str = "cpu",
    ):
        self.network = network.to(device)
        self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl

        self.optimizer = optim.Adam(network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

        # Training stats
        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0

    def collect_rollout(self, env, num_steps: int) -> dict:
        """Collect num_steps of experience from the environment.

        Returns:
            dict with episode stats (rewards, lengths, etc.)
        """
        self.buffer.clear()
        self.network.eval()

        obs, info = env.reset()
        hidden = self.network.init_hidden(batch_size=1).to(self.device)
        action_mask = env.get_action_mask()

        episode_rewards = []
        episode_lengths = []
        current_ep_reward = 0.0
        current_ep_length = 0

        for step in range(num_steps):
            # Convert obs to tensors
            obs_tensor = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                for k, v in obs.items()
            }
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, entropy, value, hidden = self.network.get_action_and_value(
                    obs_tensor, hidden, mask_tensor
                )

            action_np = action.squeeze(0).cpu().numpy()
            log_prob_val = log_prob.item()
            value_val = value.item()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Store experience
            self.buffer.add(
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
            self.total_steps += 1

            if done:
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                current_ep_reward = 0.0
                current_ep_length = 0
                self.total_episodes += 1

                obs, info = env.reset()
                hidden = self.network.init_hidden(batch_size=1).to(self.device)
                action_mask = env.get_action_mask()
            else:
                obs = next_obs
                action_mask = env.get_action_mask()

        # Compute last value for GAE
        with torch.no_grad():
            obs_tensor = {
                k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0)
                for k, v in obs.items()
            }
            last_value, _ = self.network.get_value(obs_tensor, hidden)
            last_value = last_value.item()

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        return {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        }

    def train_on_buffer(self) -> dict:
        """Train PPO on the collected rollout buffer.

        Returns:
            dict with training stats.
        """
        self.network.train()

        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_clip_fractions = []
        all_approx_kl = []

        for epoch in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.minibatch_size, self.device):
                (batch_obs, batch_actions, batch_action_masks,
                 batch_old_log_probs, batch_advantages, batch_returns) = batch

                # Forward pass (no hidden state for batched training - we break temporal deps)
                _, new_log_prob, entropy, new_value, _ = self.network.get_action_and_value(
                    batch_obs, hidden=None, action_mask=batch_action_masks, action=batch_actions
                )

                # Normalize advantages
                adv = batch_advantages
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Policy loss (clipped surrogate)
                log_ratio = new_log_prob - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_value, batch_returns)

                # Entropy loss (bonus for exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Stats
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(-entropy_loss.item())
                all_clip_fractions.append(clip_fraction)
                all_approx_kl.append(approx_kl)

            # Early stopping based on KL divergence
            if np.mean(all_approx_kl[-len(self.buffer) // self.minibatch_size:]) > self.target_kl:
                break

        self.update_count += 1

        return {
            "policy_loss": np.mean(all_policy_losses),
            "value_loss": np.mean(all_value_losses),
            "entropy": np.mean(all_entropy_losses),
            "clip_fraction": np.mean(all_clip_fractions),
            "approx_kl": np.mean(all_approx_kl),
            "epochs_run": epoch + 1,
        }

    def save_checkpoint(self, path: str, metadata: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "update_count": self.update_count,
        }
        if metadata:
            checkpoint["metadata"] = metadata
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_episodes = checkpoint.get("total_episodes", 0)
        self.update_count = checkpoint.get("update_count", 0)

    def export_torchscript(self, path: str, obs_example: dict):
        """Export model as TorchScript for DJL inference."""
        self.network.eval()

        # Create a wrapper that takes flat tensor inputs for TorchScript compatibility
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

        wrapper = InferenceWrapper(self.network)
        wrapper.eval()

        # Create example inputs
        example_inputs = (
            torch.zeros(1, 30),
            torch.zeros(1, 32, 20),
            torch.zeros(1, 32),
            torch.zeros(1, 22),
            torch.zeros(1, 12),
            torch.zeros(1, 8),
            torch.zeros(1, 1, 128),
        )

        scripted = torch.jit.trace(wrapper, example_inputs)
        scripted.save(path)
        print(f"Exported TorchScript model to {path}")


# Need F for value loss
import torch.nn.functional as F
