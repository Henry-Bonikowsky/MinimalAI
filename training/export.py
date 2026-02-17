"""Export trained models to TorchScript for DJL inference in Java.

Usage:
    python export.py checkpoints/skill_master.pt exported/master.pt
    python export.py --all  # Export all skill checkpoints
"""

import os
import sys
import argparse
import torch
from models.network import CombatNetwork
from models.ppo import PPO


def export_model(checkpoint_path: str, output_path: str):
    """Export a checkpoint to TorchScript."""
    print(f"Loading checkpoint: {checkpoint_path}")

    network = CombatNetwork()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["network_state_dict"])
    network.eval()

    metadata = checkpoint.get("metadata", {})
    print(f"  Steps: {checkpoint.get('total_steps', 'unknown'):,}")
    print(f"  Skill: {metadata.get('skill_level', 'unknown')}")
    print(f"  Reward: {metadata.get('mean_reward', 'unknown')}")

    # Create inference wrapper for TorchScript
    class InferenceWrapper(torch.nn.Module):
        def __init__(self, net):
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

    wrapper = InferenceWrapper(network)
    wrapper.eval()

    # Use realistic example inputs (at least one entity visible) to trace the main code path
    entity_mask = torch.zeros(1, 32)
    entity_mask[0, 0] = 1.0  # one visible entity
    example = (
        torch.randn(1, 30),
        torch.randn(1, 32, 20),
        entity_mask,
        torch.randn(1, 22),
        torch.randn(1, 12),
        torch.randn(1, 8),
        torch.zeros(1, 1, 128),
    )

    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, example, check_trace=False)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scripted.save(output_path)
    print(f"  Exported to: {output_path}")
    print()


def export_all(checkpoint_dir: str = "checkpoints", output_dir: str = "exported"):
    """Export all skill checkpoints."""
    os.makedirs(output_dir, exist_ok=True)

    skill_names = ["novice", "apprentice", "fighter", "warrior", "master"]
    exported = 0

    for name in skill_names:
        cp_path = os.path.join(checkpoint_dir, f"skill_{name}.pt")
        if os.path.exists(cp_path):
            out_path = os.path.join(output_dir, f"{name}.pt")
            export_model(cp_path, out_path)
            exported += 1

    # Also export best and final if they exist
    for label in ["best", "final"]:
        cp_path = os.path.join(checkpoint_dir, f"{label}.pt")
        if os.path.exists(cp_path):
            out_path = os.path.join(output_dir, f"{label}.pt")
            export_model(cp_path, out_path)
            exported += 1

    print(f"Exported {exported} models to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Export MinimalAI models to TorchScript")
    parser.add_argument("input", nargs="?", help="Checkpoint path to export")
    parser.add_argument("output", nargs="?", help="Output TorchScript path")
    parser.add_argument("--all", action="store_true", help="Export all skill checkpoints")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="exported")
    args = parser.parse_args()

    if args.all:
        export_all(args.checkpoint_dir, args.output_dir)
    elif args.input and args.output:
        export_model(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
