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


class RolloutBuffer:
    """Pre-allocated rollout buffer for PPO training.

    Uses fixed numpy arrays instead of Python lists to avoid memory
    fragmentation and the triple-copy (list → np.array → torch.tensor).
    """

    def __init__(self, capacity: int, obs_shapes: dict):
        self.capacity = capacity
        self.pos = 0

        # Pre-allocate observation arrays
        self.obs_self_state = np.zeros((capacity, *obs_shapes["self_state"]), dtype=np.float32)
        self.obs_entity_features = np.zeros((capacity, *obs_shapes["entity_features"]), dtype=np.float32)
        self.obs_entity_mask = np.zeros((capacity, *obs_shapes["entity_mask"]), dtype=np.float32)
        self.obs_combat_ctx = np.zeros((capacity, *obs_shapes["combat_ctx"]), dtype=np.float32)
        self.obs_sigil_state = np.zeros((capacity, *obs_shapes["sigil_state"]), dtype=np.float32)
        self.obs_env_state = np.zeros((capacity, *obs_shapes["env_state"]), dtype=np.float32)

        self.actions = np.zeros((capacity, NUM_ACTIONS), dtype=np.float32)
        self.action_masks = np.zeros((capacity, NUM_ACTIONS), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

    def add(self, obs: dict, action, action_mask, log_prob, reward, value, done, hidden):
        i = self.pos
        self.obs_self_state[i] = obs["self_state"]
        self.obs_entity_features[i] = obs["entity_features"]
        self.obs_entity_mask[i] = obs["entity_mask"]
        self.obs_combat_ctx[i] = obs["combat_ctx"]
        self.obs_sigil_state[i] = obs["sigil_state"]
        self.obs_env_state[i] = obs["env_state"]
        self.actions[i] = action
        self.action_masks[i] = action_mask
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.pos += 1

    def clear(self):
        self.pos = 0

    def __len__(self):
        return self.pos

    def compute_gae(self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation."""
        n = self.pos
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int, device: torch.device):
        """Yield minibatches for training. Zero-copy from numpy to torch."""
        n = self.pos
        indices = np.random.permutation(n)

        # Convert pre-allocated arrays directly to tensors (no intermediate copy)
        all_obs = {
            "self_state": torch.from_numpy(self.obs_self_state[:n]).to(device),
            "entity_features": torch.from_numpy(self.obs_entity_features[:n]).to(device),
            "entity_mask": torch.from_numpy(self.obs_entity_mask[:n]).to(device),
            "combat_ctx": torch.from_numpy(self.obs_combat_ctx[:n]).to(device),
            "sigil_state": torch.from_numpy(self.obs_sigil_state[:n]).to(device),
            "env_state": torch.from_numpy(self.obs_env_state[:n]).to(device),
        }
        all_actions = torch.from_numpy(self.actions[:n]).to(device)
        all_action_masks = torch.from_numpy(self.action_masks[:n]).to(device)
        all_log_probs = torch.from_numpy(self.log_probs[:n]).to(device)
        all_advantages = torch.from_numpy(self.advantages[:n]).to(device)
        all_returns = torch.from_numpy(self.returns[:n]).to(device)

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
        self.buffer = None  # created lazily in collect_rollout

        # Training stats
        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0

    def collect_rollout(self, env, num_steps: int) -> dict:
        """Collect num_steps of experience from the environment.

        Returns:
            dict with episode stats (rewards, lengths, etc.)
        """
        self.network.eval()

        obs, info = env.reset()

        # Lazily create buffer with correct observation shapes
        if self.buffer is None:
            obs_shapes = {k: v.shape for k, v in obs.items()}
            self.buffer = RolloutBuffer(num_steps, obs_shapes)
        self.buffer.clear()
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
                hidden=None,
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
