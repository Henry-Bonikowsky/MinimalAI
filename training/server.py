"""TCP training server for MinimalAI live RL training.

Receives experience batches from Java Paper plugin,
runs PPO training, exports updated TorchScript models.

Usage:
    python server.py [--port 9876] [--model-dir ../server/plugins/MinimalAI/models]
    python server.py --port 9876 --model-dir models --checkpoint checkpoints/latest.pt
"""

import socket
import struct
import threading
import argparse
import logging
import json
import time
import os
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path

# Import from existing training code
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.network import (
    CombatNetwork,
    SELF_STATE_DIM,
    ENTITY_FEATURE_DIM,
    COMBAT_CTX_DIM,
    SIGIL_STATE_DIM,
    ENV_STATE_DIM,
    NUM_ACTIONS,
    MAX_ENTITIES,
    GRU_HIDDEN_DIM,
)
from models.ppo import PPO, RolloutBuffer
from config import TrainingConfig

# Protocol constants (must match TrainingClient.java)
MSG_EXPERIENCE = 0x01
MSG_MODEL_UPDATED = 0x02
MSG_TRAINING_STATS = 0x03

# Observation dimensions (must match Java ObservationSpace.java)
# SELF_STATE_DIM = 38
# ENTITY_FEATURE_DIM = 24
# MAX_ENTITIES = 8
# COMBAT_CTX_DIM = 26
# SIGIL_STATE_DIM = 48
# ENV_STATE_DIM = 8
# NUM_ACTIONS = 35
# GRU_HIDDEN_DIM = 128

# Per-experience binary layout sizes (all big-endian)
# float32 = 4 bytes, int32 = 4 bytes, byte = 1 byte
#
# Layout per experience:
#   float[38]  selfState          = 38 * 4 = 152
#   float[192] entityFeatures     = 192 * 4 = 768  (8 * 24 flattened)
#   float[8]   entityMask         = 8 * 4 = 32
#   float[26]  combatCtx          = 26 * 4 = 104
#   float[48]  sigilState         = 48 * 4 = 192
#   float[8]   envState           = 8 * 4 = 32
#   int[35]    actions            = 35 * 4 = 140   (0 or 1 as int32)
#   float[35]  actionProbs        = 35 * 4 = 140
#   float      value              = 4
#   float      reward             = 4
#   byte       done               = 1
#   float[128] hidden             = 128 * 4 = 512
#
# Total per experience: 152 + 768 + 32 + 104 + 192 + 32 + 140 + 140 + 4 + 4 + 1 + 512 = 2081 bytes

EXPERIENCE_SIZE = (
    SELF_STATE_DIM * 4
    + MAX_ENTITIES * ENTITY_FEATURE_DIM * 4
    + MAX_ENTITIES * 4
    + COMBAT_CTX_DIM * 4
    + SIGIL_STATE_DIM * 4
    + ENV_STATE_DIM * 4
    + NUM_ACTIONS * 4       # actions (int32)
    + NUM_ACTIONS * 4       # action probs (float32)
    + 4                     # value (float32)
    + 4                     # reward (float32)
    + 1                     # done (byte)
    + GRU_HIDDEN_DIM * 4    # hidden (float32)
)


class InferenceWrapper(nn.Module):
    """TorchScript-compatible wrapper matching export.py's InferenceWrapper."""

    def __init__(self, net: CombatNetwork):
        super().__init__()
        self.net = net

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
        logits, value, new_hidden = self.net(obs, hidden)
        action_probs = torch.sigmoid(logits)
        return action_probs, value, new_hidden


class TrainingServer:
    def __init__(self, port: int, model_dir: str, config: TrainingConfig,
                 checkpoint_path: str = None):
        self.port = port
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Initialize network and PPO
        self.network = CombatNetwork()
        self.ppo = PPO(
            network=self.network,
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

        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            self.ppo.load_checkpoint(checkpoint_path)

        total_params = sum(p.numel() for p in self.network.parameters())
        logging.info(f"Network parameters: {total_params:,}")

        # Experience buffer (shared across client connections)
        self.experience_buffer = []
        self.lock = threading.Lock()

        # Stats
        self.total_experiences = 0
        self.training_epochs = 0

        # Observation shapes for RolloutBuffer initialization
        self.obs_shapes = {
            "self_state": (SELF_STATE_DIM,),
            "entity_features": (MAX_ENTITIES, ENTITY_FEATURE_DIM),
            "entity_mask": (MAX_ENTITIES,),
            "combat_ctx": (COMBAT_CTX_DIM,),
            "sigil_state": (SIGIL_STATE_DIM,),
            "env_state": (ENV_STATE_DIM,),
        }

        self.server_socket = None
        self.running = False
        self.clients = []
        self.clients_lock = threading.Lock()

    def start(self):
        """Start the TCP server and accept connections."""
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("0.0.0.0", self.port))
        self.server_socket.listen(4)
        self.server_socket.settimeout(1.0)  # Allow periodic shutdown checks
        logging.info(f"Training server listening on port {self.port}")

        while self.running:
            try:
                client, addr = self.server_socket.accept()
                logging.info(f"Client connected: {addr}")
                with self.clients_lock:
                    self.clients.append(client)
                thread = threading.Thread(
                    target=self.handle_client, args=(client, addr), daemon=True
                )
                thread.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def handle_client(self, client: socket.socket, addr):
        """Handle incoming messages from a Java plugin client."""
        try:
            while self.running:
                # Read message type (1 byte)
                raw_type = self._recv_exact(client, 1)
                msg_type = struct.unpack("b", raw_type)[0] & 0xFF

                if msg_type == MSG_EXPERIENCE:
                    # Read data length (4 bytes, big-endian int)
                    data_len = struct.unpack(">i", self._recv_exact(client, 4))[0]
                    data = self._recv_exact(client, data_len)
                    experiences = self.deserialize_experiences(data)

                    with self.lock:
                        self.experience_buffer.extend(experiences)
                        self.total_experiences += len(experiences)

                    logging.info(
                        f"Received {len(experiences)} experiences "
                        f"(buffer: {len(self.experience_buffer)}, "
                        f"total: {self.total_experiences})"
                    )

                    # Train if buffer has enough experiences
                    if len(self.experience_buffer) >= self.config.rollout_steps:
                        self.train_and_export(client)

                else:
                    logging.warning(f"Unknown message type from {addr}: 0x{msg_type:02x}")
                    # Try to skip: read length + payload
                    try:
                        skip_len = struct.unpack(">i", self._recv_exact(client, 4))[0]
                        self._recv_exact(client, skip_len)
                    except Exception:
                        break

        except (ConnectionResetError, BrokenPipeError, struct.error) as e:
            logging.info(f"Client {addr} disconnected: {e}")
        except Exception as e:
            logging.error(f"Error handling client {addr}: {e}", exc_info=True)
        finally:
            with self.clients_lock:
                if client in self.clients:
                    self.clients.remove(client)
            try:
                client.close()
            except Exception:
                pass

    def deserialize_experiences(self, data: bytes) -> list:
        """Parse binary experience data from Java.

        Format per experience (big-endian):
            float[38]  selfState
            float[192] entityFeatures  (8 * 24 flattened)
            float[8]   entityMask
            float[26]  combatCtx
            float[48]  sigilState
            float[8]   envState
            int[35]    actions         (0 or 1 as int32)
            float[35]  actionProbs
            float      value
            float      reward
            byte       done
            float[128] hidden

        Returns:
            List of dicts, each with numpy arrays for the experience fields.
        """
        if len(data) < EXPERIENCE_SIZE:
            logging.warning(
                f"Experience data too short: {len(data)} bytes "
                f"(expected at least {EXPERIENCE_SIZE})"
            )
            return []

        num_experiences = len(data) // EXPERIENCE_SIZE
        if len(data) % EXPERIENCE_SIZE != 0:
            logging.warning(
                f"Experience data length {len(data)} is not a multiple of "
                f"experience size {EXPERIENCE_SIZE}; truncating"
            )

        experiences = []
        offset = 0

        for _ in range(num_experiences):
            # self_state: float[38]
            n = SELF_STATE_DIM
            self_state = np.array(
                struct.unpack_from(f">{n}f", data, offset), dtype=np.float32
            )
            offset += n * 4

            # entity_features: float[192] -> reshape (8, 24)
            n = MAX_ENTITIES * ENTITY_FEATURE_DIM
            entity_features = np.array(
                struct.unpack_from(f">{n}f", data, offset), dtype=np.float32
            ).reshape(MAX_ENTITIES, ENTITY_FEATURE_DIM)
            offset += n * 4

            # entity_mask: float[8]
            n = MAX_ENTITIES
            entity_mask = np.array(
                struct.unpack_from(f">{n}f", data, offset), dtype=np.float32
            )
            offset += n * 4

            # combat_ctx: float[26]
            n = COMBAT_CTX_DIM
            combat_ctx = np.array(
                struct.unpack_from(f">{n}f", data, offset), dtype=np.float32
            )
            offset += n * 4

            # sigil_state: float[48]
            n = SIGIL_STATE_DIM
            sigil_state = np.array(
                struct.unpack_from(f">{n}f", data, offset), dtype=np.float32
            )
            offset += n * 4

            # env_state: float[8]
            n = ENV_STATE_DIM
            env_state = np.array(
                struct.unpack_from(f">{n}f", data, offset), dtype=np.float32
            )
            offset += n * 4

            # actions: int[35] (0 or 1)
            n = NUM_ACTIONS
            actions = np.array(
                struct.unpack_from(f">{n}i", data, offset), dtype=np.float32
            )
            offset += n * 4

            # action_probs: float[35]
            action_probs = np.array(
                struct.unpack_from(f">{n}f", data, offset), dtype=np.float32
            )
            offset += n * 4

            # value: float
            value = struct.unpack_from(">f", data, offset)[0]
            offset += 4

            # reward: float
            reward = struct.unpack_from(">f", data, offset)[0]
            offset += 4

            # done: byte (0 or 1)
            done = struct.unpack_from("b", data, offset)[0]
            offset += 1

            # hidden: float[128]
            n = GRU_HIDDEN_DIM
            hidden = np.array(
                struct.unpack_from(f">{n}f", data, offset), dtype=np.float32
            )
            offset += n * 4

            # Compute log_prob from action_probs and actions (Bernoulli log prob)
            # log_prob = sum(action * log(p) + (1 - action) * log(1 - p))
            eps = 1e-7
            probs_clamped = np.clip(action_probs, eps, 1.0 - eps)
            log_prob = np.sum(
                actions * np.log(probs_clamped)
                + (1.0 - actions) * np.log(1.0 - probs_clamped)
            )

            # Build action mask: all ones (Java should provide proper mask,
            # but for now assume all actions valid from received data)
            action_mask = np.ones(NUM_ACTIONS, dtype=np.float32)

            experiences.append({
                "obs": {
                    "self_state": self_state,
                    "entity_features": entity_features,
                    "entity_mask": entity_mask,
                    "combat_ctx": combat_ctx,
                    "sigil_state": sigil_state,
                    "env_state": env_state,
                },
                "actions": actions,
                "action_mask": action_mask,
                "log_prob": float(log_prob),
                "value": float(value),
                "reward": float(reward),
                "done": bool(done),
                "hidden": hidden,
            })

        return experiences

    def train_and_export(self, client: socket.socket):
        """Run PPO training on buffered experiences, export model, notify client."""
        with self.lock:
            experiences = list(self.experience_buffer)
            self.experience_buffer.clear()

        n = len(experiences)
        if n == 0:
            return

        logging.info(f"Starting training on {n} experiences...")
        start_time = time.time()

        # Build a RolloutBuffer from the received experiences
        buffer = RolloutBuffer(n, self.obs_shapes)
        for exp in experiences:
            buffer.add(
                obs=exp["obs"],
                action=exp["actions"],
                action_mask=exp["action_mask"],
                log_prob=exp["log_prob"],
                reward=exp["reward"],
                value=exp["value"],
                done=exp["done"],
                hidden=None,
            )

        # Compute GAE with the last experience's value as bootstrap
        # (If the last experience is done=True, value is irrelevant)
        last_value = experiences[-1]["value"] if not experiences[-1]["done"] else 0.0
        buffer.compute_gae(last_value, self.config.gamma, self.config.gae_lambda)

        # Update entropy coefficient based on progress
        progress = self.ppo.total_steps / max(self.config.total_timesteps, 1)
        self.ppo.entropy_coef = (
            self.config.entropy_coef_start
            + (self.config.entropy_coef_end - self.config.entropy_coef_start)
            * min(1.0, progress)
        )

        # Assign buffer to PPO and train
        self.ppo.buffer = buffer
        self.ppo.total_steps += n
        train_stats = self.ppo.train_on_buffer()
        self.training_epochs += 1

        elapsed = time.time() - start_time
        logging.info(
            f"Training epoch {self.training_epochs} complete in {elapsed:.2f}s: "
            f"policy_loss={train_stats['policy_loss']:.4f}, "
            f"value_loss={train_stats['value_loss']:.4f}, "
            f"entropy={train_stats['entropy']:.4f}, "
            f"approx_kl={train_stats['approx_kl']:.4f}"
        )

        # Export TorchScript model
        model_name = f"live_epoch{self.training_epochs}"
        model_path = self.model_dir / f"{model_name}.pt"
        self._export_torchscript(model_path)

        # Also save as "latest.pt" for easy hot-reload
        latest_path = self.model_dir / "latest.pt"
        self._export_torchscript(latest_path)

        # Save PPO checkpoint for resuming
        checkpoint_path = self.model_dir / "live_checkpoint.pt"
        self.ppo.save_checkpoint(
            str(checkpoint_path),
            metadata={
                "training_epochs": self.training_epochs,
                "total_experiences": self.total_experiences,
            },
        )

        # Send MSG_MODEL_UPDATED to the client
        self._send_model_updated(client, str(latest_path))

        # Send MSG_TRAINING_STATS to the client
        stats = {
            "epoch": self.training_epochs,
            "total_steps": self.ppo.total_steps,
            "total_experiences": self.total_experiences,
            "elapsed_s": round(elapsed, 3),
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in train_stats.items()},
        }
        self._send_training_stats(client, stats)

        # Also broadcast to all connected clients
        self._broadcast_model_updated(str(latest_path), exclude=client)

    def _export_torchscript(self, path: Path):
        """Export current network as TorchScript for DJL inference."""
        self.network.eval()
        wrapper = InferenceWrapper(self.network)
        wrapper.eval()

        entity_mask = torch.zeros(1, MAX_ENTITIES)
        entity_mask[0, 0] = 1.0

        example = (
            torch.randn(1, SELF_STATE_DIM),
            torch.randn(1, MAX_ENTITIES, ENTITY_FEATURE_DIM),
            entity_mask,
            torch.randn(1, COMBAT_CTX_DIM),
            torch.randn(1, SIGIL_STATE_DIM),
            torch.randn(1, ENV_STATE_DIM),
            torch.zeros(1, 1, GRU_HIDDEN_DIM),
        )

        with torch.no_grad():
            scripted = torch.jit.trace(wrapper, example, check_trace=False)

        path.parent.mkdir(parents=True, exist_ok=True)
        scripted.save(str(path))
        logging.info(f"Exported TorchScript model: {path}")

    def _send_model_updated(self, client: socket.socket, model_path: str):
        """Send MSG_MODEL_UPDATED to a client."""
        try:
            path_bytes = model_path.encode("utf-8")
            msg = struct.pack(">bi", MSG_MODEL_UPDATED, len(path_bytes)) + path_bytes
            client.sendall(msg)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logging.warning(f"Failed to send model_updated: {e}")

    def _send_training_stats(self, client: socket.socket, stats: dict):
        """Send MSG_TRAINING_STATS to a client."""
        try:
            stats_bytes = json.dumps(stats).encode("utf-8")
            msg = struct.pack(">bi", MSG_TRAINING_STATS, len(stats_bytes)) + stats_bytes
            client.sendall(msg)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logging.warning(f"Failed to send training_stats: {e}")

    def _broadcast_model_updated(self, model_path: str, exclude: socket.socket = None):
        """Broadcast model update to all connected clients except the sender."""
        with self.clients_lock:
            for c in list(self.clients):
                if c is not exclude:
                    self._send_model_updated(c, model_path)

    def _recv_exact(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes from a socket."""
        data = b""
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionResetError("Connection closed while reading")
            data += chunk
        return data

    def stop(self):
        """Stop the server and close all connections."""
        self.running = False
        with self.clients_lock:
            for c in self.clients:
                try:
                    c.close()
                except Exception:
                    pass
            self.clients.clear()
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        logging.info("Training server stopped")


def main():
    parser = argparse.ArgumentParser(
        description="MinimalAI TCP Training Server - receives live experience from "
                    "Paper plugin, runs PPO, exports updated TorchScript models."
    )
    parser.add_argument(
        "--port", type=int, default=9876,
        help="TCP port to listen on (default: 9876)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="../server/plugins/MinimalAI/models",
        help="Directory to export TorchScript models to (default: ../server/plugins/MinimalAI/models)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a PPO checkpoint to resume training from"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=2048,
        help="Number of experiences before running a training update (default: 2048)"
    )
    parser.add_argument(
        "--total-steps", type=int, default=10_000_000,
        help="Total training steps budget for LR/entropy scheduling (default: 10M)"
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=4,
        help="PPO epochs per training update (default: 4)"
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=128,
        help="Minibatch size for PPO (default: 128)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TrainingConfig(
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        total_timesteps=args.total_steps,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        device="cpu",
    )

    logging.info("=" * 60)
    logging.info("MinimalAI Training Server")
    logging.info("=" * 60)
    logging.info(f"Port: {args.port}")
    logging.info(f"Model dir: {args.model_dir}")
    logging.info(f"Checkpoint: {args.checkpoint or 'None (fresh start)'}")
    logging.info(f"Rollout steps: {config.rollout_steps}")
    logging.info(f"LR: {config.lr}")
    logging.info(f"PPO epochs: {config.ppo_epochs}")
    logging.info(f"Minibatch size: {config.minibatch_size}")
    logging.info("=" * 60)

    server = TrainingServer(
        port=args.port,
        model_dir=args.model_dir,
        config=config,
        checkpoint_path=args.checkpoint,
    )

    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
