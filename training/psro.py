#!/usr/bin/env python3
"""Policy-Space Response Oracles (PSRO) — iterative self-play training.

Train generation 0 vs rule-based opponent, freeze it, train generation 1
against frozen gen 0, freeze gen 1, train gen 2 against frozen gen 1, etc.
Each generation learns to beat the previous one.

Usage:
    python -m training.psro --generations 5 --steps-per-gen 20000000 --device cuda
    python -m training.psro --generations 3 --resume-from-gen 2  # resume after gen 1 done
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Lazy imports for evaluation (avoid loading CUDA in orchestrator process)
_eval_imports_loaded = False


def _ensure_eval_imports():
    global _eval_imports_loaded
    if _eval_imports_loaded:
        return
    global VecPvPSim, CombatNetwork, PolicyWrapper
    from .vec_sim import VecPvPSim
    from .models.network import CombatNetwork
    from .fast_train import PolicyWrapper
    _eval_imports_loaded = True


def parse_args():
    p = argparse.ArgumentParser(description="PSRO iterative self-play training")
    p.add_argument("--generations", type=int, default=5,
                    help="Number of generations to train (default: 5)")
    p.add_argument("--steps-per-gen", type=int, default=20_000_000,
                    help="Training steps per generation (default: 20M)")
    p.add_argument("--base-dir", type=str, default="checkpoints_psro",
                    help="Base directory for all generation checkpoints")
    p.add_argument("--device", type=str, default="cuda",
                    help="Training device (default: cuda)")
    p.add_argument("--n-envs", type=int, default=512,
                    help="Number of parallel environments (default: 512)")
    p.add_argument("--combo-style", type=str, default="stap",
                    choices=["stap", "wtap", "none"],
                    help="Combo style (default: stap)")
    p.add_argument("--eval-episodes", type=int, default=500,
                    help="Evaluation episodes per matchup (default: 500)")
    p.add_argument("--resume-from-gen", type=int, default=0,
                    help="Resume from this generation (skip completed gens)")
    return p.parse_args()


def gen_dir(base_dir: str, gen: int) -> Path:
    return Path(base_dir) / f"gen{gen}"


def gen_model_path(base_dir: str, gen: int) -> Path:
    return gen_dir(base_dir, gen) / "fast_final.pt"


def gen_is_complete(base_dir: str, gen: int) -> bool:
    return gen_model_path(base_dir, gen).exists()


def run_training(gen: int, args, prev_model: str = None) -> bool:
    """Run fast_train.py as a subprocess for one generation.

    Returns True if training succeeded.
    """
    cmd = [
        sys.executable, "-m", "training.fast_train",
        "--steps", str(args.steps_per_gen),
        "--n-envs", str(args.n_envs),
        "--device", args.device,
        "--combo-style", args.combo_style,
        "--checkpoint-dir", str(gen_dir(args.base_dir, gen)),
    ]

    if gen == 0:
        # Gen 0: train counter vs rule-based charger
        cmd.append("--rule-opponent")
    else:
        # Gen 1+: train counter vs frozen previous generation
        cmd.extend(["--frozen-charger", prev_model])
        cmd.append("--no-rule-opponent")

    print(f"\n{'=' * 70}")
    print(f"PSRO Generation {gen} — Training")
    print(f"{'=' * 70}")
    if gen == 0:
        print(f"  Opponent: rule-based charger")
    else:
        print(f"  Opponent: frozen gen {gen - 1} ({prev_model})")
    print(f"  Steps:    {args.steps_per_gen:,}")
    print(f"  Envs:     {args.n_envs}")
    print(f"  Device:   {args.device}")
    print(f"  Output:   {gen_dir(args.base_dir, gen)}")
    print(f"  Command:  {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nERROR: Gen {gen} training failed (exit code {result.returncode})")
        return False

    print(f"\nGen {gen} training complete in {elapsed / 60:.1f} minutes")
    return True


def run_eval_vs_rule(model_path: str, args) -> dict:
    """Evaluate a model against the rule-based opponent via subprocess."""
    cmd = [
        sys.executable, "-m", "training.evaluate",
        model_path,
        "--episodes", str(args.eval_episodes),
        "--n-envs", "64",
        "--device", args.device,
        "--combo-style", args.combo_style,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Eval failed: {result.stderr[:200]}")
        return {}

    # Parse key metrics from evaluate.py stdout
    stats = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if "Kills:" in line:
            stats["kill_rate"] = line.split(":")[1].strip()
        elif "Deaths:" in line:
            stats["death_rate"] = line.split(":")[1].strip()
        elif "Timeouts:" in line:
            stats["timeout_rate"] = line.split(":")[1].strip()
        elif "Mean reward:" in line:
            stats["mean_reward"] = line.split(":")[1].strip()
    return stats


def eval_head_to_head(model_a_path: str, model_b_path: str,
                       n_episodes: int = 500, n_envs: int = 64,
                       device: str = "cpu", combo_style: str = "stap") -> dict:
    """Evaluate model A vs model B directly (A=counter, B=charger role).

    Runs in-process to avoid needing a CLI flag for model-vs-model in evaluate.py.
    """
    _ensure_eval_imports()

    sim = VecPvPSim(n_envs, episode_length=1800, seed=789, combo_style=combo_style)

    # Load model A
    net_a = CombatNetwork()
    policy_a = PolicyWrapper(net_a).to(device)
    ckpt_a = torch.load(model_a_path, map_location=device, weights_only=False)
    if "network_state_dict" in ckpt_a:
        net_a.load_state_dict(ckpt_a["network_state_dict"])
    else:
        net_a.load_state_dict(ckpt_a)
    policy_a.eval()

    # Load model B
    net_b = CombatNetwork()
    policy_b = PolicyWrapper(net_b).to(device)
    ckpt_b = torch.load(model_b_path, map_location=device, weights_only=False)
    if "network_state_dict" in ckpt_b:
        net_b.load_state_dict(ckpt_b["network_state_dict"])
    else:
        net_b.load_state_dict(ckpt_b)
    policy_b.eval()

    obs = sim.reset_all()
    completed = 0
    a_kills = 0
    b_kills = 0
    timeouts = 0

    while completed < n_episodes:
        obs_t = torch.from_numpy(obs).to(device)
        b_obs = sim.get_b_obs()
        b_obs_t = torch.from_numpy(b_obs).to(device)

        with torch.no_grad():
            a_action, _, _, _, _ = policy_a.get_action_and_value(obs_t)
            b_action = policy_b.sample_actions(b_obs_t)

        a_np = a_action.cpu().numpy().astype(np.int8)
        b_np = b_action.cpu().numpy().astype(np.int8)

        # B faces A (opponent targeting)
        dx = sim.ax - sim.bx
        dz = sim.az - sim.bz
        sim.byaw[:] = np.degrees(np.arctan2(-dx, dz))

        next_obs, rewards, dones, infos = sim.step(a_np, b_np)

        for i in range(n_envs):
            if dones[i] and completed < n_episodes:
                if infos["b_health"][i] <= 0:
                    a_kills += 1
                elif infos["a_health"][i] <= 0:
                    b_kills += 1
                else:
                    timeouts += 1
                completed += 1

        obs = next_obs

    return {
        "a_kill_rate": a_kills / completed,
        "b_kill_rate": b_kills / completed,
        "timeout_rate": timeouts / completed,
        "n_episodes": completed,
    }


def run_evaluations(gen: int, args) -> dict:
    """Run all evaluations for a completed generation.

    Returns dict with results for printing/summary.
    """
    model = str(gen_model_path(args.base_dir, gen))
    results = {"gen": gen}

    # 1. Evaluate vs rule-based
    print(f"\n  Evaluating gen {gen} vs rule-based...")
    rule_stats = run_eval_vs_rule(model, args)
    results["vs_rule"] = rule_stats
    if rule_stats:
        print(f"    Kill rate: {rule_stats.get('kill_rate', '?')}")
        print(f"    Death rate: {rule_stats.get('death_rate', '?')}")

    # 2. Evaluate vs each previous generation (head-to-head)
    for prev_gen in range(gen):
        prev_model = str(gen_model_path(args.base_dir, prev_gen))
        if not Path(prev_model).exists():
            continue
        print(f"  Evaluating gen {gen} vs gen {prev_gen}...")
        h2h = eval_head_to_head(
            model, prev_model,
            n_episodes=args.eval_episodes,
            n_envs=64,
            device=args.device,
            combo_style=args.combo_style,
        )
        results[f"vs_gen{prev_gen}"] = h2h
        print(f"    Gen {gen} kills: {h2h['a_kill_rate']:.1%}  "
              f"Gen {prev_gen} kills: {h2h['b_kill_rate']:.1%}  "
              f"Timeouts: {h2h['timeout_rate']:.1%}")

    return results


def print_summary(all_results: list[dict]):
    """Print a summary table of all generations."""
    print(f"\n{'=' * 80}")
    print("PSRO TRAINING SUMMARY")
    print(f"{'=' * 80}\n")

    # Header
    header = f"{'Gen':>4} | {'vs Rule Kill%':>14} | {'vs Rule Death%':>15}"
    for r in all_results:
        gen = r["gen"]
        # Add columns for head-to-head vs previous gens
    # Determine max gen for dynamic columns
    max_gen = max(r["gen"] for r in all_results) if all_results else 0
    header = f"{'Gen':>4} | {'vs Rule Kill%':>14} | {'vs Rule Death%':>15}"
    for g in range(max_gen + 1):
        header += f" | {'vs Gen' + str(g):>10}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        gen = r["gen"]
        rule = r.get("vs_rule", {})
        kill = rule.get("kill_rate", "?")
        death = rule.get("death_rate", "?")
        row = f"{gen:>4} | {kill:>14} | {death:>15}"
        for g in range(max_gen + 1):
            key = f"vs_gen{g}"
            if key in r:
                h2h = r[key]
                win_pct = f"{h2h['a_kill_rate']:.0%}"
            elif g == gen:
                win_pct = "-"
            elif g > gen:
                win_pct = ""
            else:
                win_pct = "?"
            row += f" | {win_pct:>10}"
        print(row)

    print(f"\n{'=' * 80}")


def save_results(all_results: list[dict], base_dir: str):
    """Save results to JSON for later analysis."""
    out_path = Path(base_dir) / "psro_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    serializable = json.loads(json.dumps(all_results, default=convert))
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved to {out_path}")


def main():
    args = parse_args()
    base = Path(args.base_dir)
    base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"  PSRO — Policy-Space Response Oracles")
    print(f"  Generations: {args.generations}")
    print(f"  Steps/gen:   {args.steps_per_gen:,}")
    print(f"  Base dir:    {args.base_dir}")
    print(f"  Device:      {args.device}")
    print(f"  Combo style: {args.combo_style}")
    if args.resume_from_gen > 0:
        print(f"  Resuming from gen {args.resume_from_gen}")
    print(f"{'#' * 70}\n")

    all_results = []
    t_start = time.time()

    for gen in range(args.generations):
        # Check if this generation is already complete
        if gen_is_complete(args.base_dir, gen):
            if gen < args.resume_from_gen:
                print(f"Gen {gen}: already complete (skipping)")
                # Still run eval so we have results for the summary table
                results = run_evaluations(gen, args)
                all_results.append(results)
                continue
            elif gen >= args.resume_from_gen:
                print(f"Gen {gen}: checkpoint exists at {gen_model_path(args.base_dir, gen)}")
                print(f"  Skipping training, running evaluation only")
                results = run_evaluations(gen, args)
                all_results.append(results)
                continue

        # Skip gens before resume point if they don't have checkpoints
        if gen < args.resume_from_gen:
            if not gen_is_complete(args.base_dir, gen):
                print(f"ERROR: Gen {gen} not complete but --resume-from-gen {args.resume_from_gen}")
                print(f"  Expected: {gen_model_path(args.base_dir, gen)}")
                sys.exit(1)

        # Determine opponent
        prev_model = None
        if gen > 0:
            prev_model = str(gen_model_path(args.base_dir, gen - 1))
            if not Path(prev_model).exists():
                print(f"ERROR: Previous gen model not found: {prev_model}")
                sys.exit(1)

        # Train
        success = run_training(gen, args, prev_model)
        if not success:
            print(f"\nTraining failed at generation {gen}. Stopping.")
            break

        # Verify output
        if not gen_is_complete(args.base_dir, gen):
            print(f"ERROR: Training completed but {gen_model_path(args.base_dir, gen)} not found")
            print(f"  Check fast_train.py output for errors.")
            sys.exit(1)

        # Evaluate
        results = run_evaluations(gen, args)
        all_results.append(results)
        save_results(all_results, args.base_dir)

    total_time = time.time() - t_start
    print(f"\nTotal PSRO time: {total_time / 3600:.1f} hours")

    print_summary(all_results)
    save_results(all_results, args.base_dir)


if __name__ == "__main__":
    main()
