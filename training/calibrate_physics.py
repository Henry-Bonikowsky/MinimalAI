#!/usr/bin/env python3
"""
Physics calibration script for MinimalAI.

Reads a CSV recorded by PhysicsRecorder (from /mai record) and:
1. Derives actual server physics constants from real bot movement data
2. Validates the MCSimulator against recorded data (per-tick RMSE)
3. Optionally tunes simulator constants via scipy.optimize

Usage:
    python training/calibrate_physics.py <recording.csv>
    python training/calibrate_physics.py <recording.csv> --validate
    python training/calibrate_physics.py <recording.csv> --optimize
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from .mc_physics import (
    MCSimulator, PlayerState,
    state_from_csv_row, actions_from_csv_row,
    GRAVITY, AIR_DRAG, GROUND_FRICTION, DRAG_FACTOR, JUMP_IMPULSE,
)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} ticks from {csv_path}")
    return df


def compute_displacement(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-tick displacement columns."""
    df["dx"] = df["post_x"] - df["pre_x"]
    df["dy"] = df["post_y"] - df["pre_y"]
    df["dz"] = df["post_z"] - df["pre_z"]
    df["horiz_disp"] = np.sqrt(df["dx"] ** 2 + df["dz"] ** 2)
    return df


def measure_walk_speed(df: pd.DataFrame) -> float:
    """Measure walk speed: ticks where forward=1, sprint=0, on ground, no jump."""
    mask = (
        (df["act_fwd"] == 1)
        & (df["act_back"] == 0)
        & (df["act_sprint"] == 0)
        & (df["pre_onGround"] == 1)
        & (df["act_jump"] == 0)
        & (df["act_left"] == 0)
        & (df["act_right"] == 0)
    )
    walking = df[mask]
    if len(walking) < 5:
        print(f"  WARNING: Only {len(walking)} walking ticks found")
        return 0.0
    speed = walking["horiz_disp"].mean()
    std = walking["horiz_disp"].std()
    print(f"  Walk speed: {speed:.6f} blocks/tick ({speed*20:.3f} blocks/sec) "
          f"[n={len(walking)}, std={std:.6f}]")
    return speed


def measure_sprint_speed(df: pd.DataFrame) -> float:
    """Measure sprint speed: ticks where forward=1, sprint=1, on ground, no jump."""
    mask = (
        (df["act_fwd"] == 1)
        & (df["act_back"] == 0)
        & (df["act_sprint"] == 1)
        & (df["pre_onGround"] == 1)
        & (df["pre_sprinting"] == 1)
        & (df["act_jump"] == 0)
        & (df["act_left"] == 0)
        & (df["act_right"] == 0)
    )
    sprinting = df[mask]
    if len(sprinting) < 5:
        print(f"  WARNING: Only {len(sprinting)} sprint ticks found")
        return 0.0
    speed = sprinting["horiz_disp"].mean()
    std = sprinting["horiz_disp"].std()
    print(f"  Sprint speed: {speed:.6f} blocks/tick ({speed*20:.3f} blocks/sec) "
          f"[n={len(sprinting)}, std={std:.6f}]")
    return speed


def measure_gravity(df: pd.DataFrame) -> float:
    """Measure gravity: vertical velocity change while airborne and not jumping."""
    mask = (
        (df["pre_onGround"] == 0)
        & (df["post_onGround"] == 0)
        & (df["act_jump"] == 0)
    )
    airborne = df[mask]
    if len(airborne) < 3:
        print(f"  WARNING: Only {len(airborne)} airborne ticks found")
        return 0.0
    dvy = airborne["post_vy"] - airborne["pre_vy"]
    gravity = dvy.mean()
    std = dvy.std()
    print(f"  Gravity: {gravity:.6f} blocks/tick² ({gravity*400:.3f} blocks/sec²) "
          f"[n={len(airborne)}, std={std:.6f}]")
    return gravity


def measure_jump_velocity(df: pd.DataFrame) -> float:
    """Measure initial jump velocity: vertical velocity on the tick after jump."""
    mask = (
        (df["act_jump"] == 1)
        & (df["pre_onGround"] == 1)
    )
    jumps = df[mask]
    if len(jumps) < 2:
        print(f"  WARNING: Only {len(jumps)} jump ticks found")
        return 0.0
    jump_vy = jumps["post_vy"].mean()
    std = jumps["post_vy"].std()
    print(f"  Jump velocity: {jump_vy:.6f} blocks/tick ({jump_vy*20:.3f} blocks/sec) "
          f"[n={len(jumps)}, std={std:.6f}]")
    return jump_vy


def measure_friction(df: pd.DataFrame) -> float:
    """Measure ground friction: velocity decay when no movement input on ground."""
    mask = (
        (df["act_fwd"] == 0)
        & (df["act_back"] == 0)
        & (df["act_left"] == 0)
        & (df["act_right"] == 0)
        & (df["pre_onGround"] == 1)
        & (df["post_onGround"] == 1)
        & (df["pre_vx"].abs() + df["pre_vz"].abs() > 0.001)
    )
    decelerating = df[mask]
    if len(decelerating) < 3:
        print(f"  WARNING: Only {len(decelerating)} deceleration ticks found")
        return 0.0
    pre_speed = np.sqrt(decelerating["pre_vx"] ** 2 + decelerating["pre_vz"] ** 2)
    post_speed = np.sqrt(decelerating["post_vx"] ** 2 + decelerating["post_vz"] ** 2)
    ratios = post_speed / pre_speed.clip(lower=0.001)
    friction = ratios.mean()
    std = ratios.std()
    print(f"  Friction ratio: {friction:.6f} (velocity retained per tick) "
          f"[n={len(decelerating)}, std={std:.6f}]")
    return friction


def measure_strafe_speed(df: pd.DataFrame) -> float:
    """Measure strafe speed: ticks where only strafing, no forward/back."""
    mask = (
        (df["act_fwd"] == 0)
        & (df["act_back"] == 0)
        & ((df["act_left"] == 1) | (df["act_right"] == 1))
        & (df["act_sprint"] == 0)
        & (df["pre_onGround"] == 1)
        & (df["act_jump"] == 0)
    )
    strafing = df[mask]
    if len(strafing) < 5:
        print(f"  WARNING: Only {len(strafing)} strafe ticks found")
        return 0.0
    speed = strafing["horiz_disp"].mean()
    std = strafing["horiz_disp"].std()
    print(f"  Strafe speed: {speed:.6f} blocks/tick ({speed*20:.3f} blocks/sec) "
          f"[n={len(strafing)}, std={std:.6f}]")
    return speed


def measure_knockback(df: pd.DataFrame) -> dict:
    """Measure knockback from recorded damage events."""
    mask = df["dmg_taken"] > 0
    hits = df[mask]
    if len(hits) < 2:
        print(f"  WARNING: Only {len(hits)} damage-taken ticks found")
        return {}

    if "hit_vel_x" in hits.columns:
        kb_horiz = np.sqrt(hits["hit_vel_x"] ** 2 + hits["hit_vel_z"] ** 2)
        kb_vert = hits["hit_vel_y"]
    elif "kb_x" in hits.columns:
        kb_horiz = np.sqrt(hits["kb_x"] ** 2 + hits["kb_z"] ** 2)
        kb_vert = hits["kb_y"]
    else:
        print("  WARNING: No knockback columns found (expected hit_vel_x or kb_x)")
        return {}

    result = {
        "horiz_mean": kb_horiz.mean(),
        "horiz_std": kb_horiz.std(),
        "vert_mean": kb_vert.mean(),
        "vert_std": kb_vert.std(),
        "count": len(hits),
    }
    print(f"  KB horizontal: {result['horiz_mean']:.6f} ± {result['horiz_std']:.6f} "
          f"[n={result['count']}]")
    print(f"  KB vertical:   {result['vert_mean']:.6f} ± {result['vert_std']:.6f}")
    return result


def compare_with_sim(measured: dict):
    """Compare measured values with current sim constants."""
    sim = {
        "walk_speed": 0.1,
        "sprint_speed": 0.26,
        "gravity": -0.08,
        "jump_velocity": 0.42,
        "friction": 0.91,
    }

    print("\n" + "=" * 60)
    print("COMPARISON: Measured vs Current Sim")
    print("=" * 60)
    print(f"{'Constant':<20} {'Measured':>12} {'Sim':>12} {'Error %':>10}")
    print("-" * 60)

    for key in ["walk_speed", "sprint_speed", "gravity", "jump_velocity", "friction"]:
        m = measured.get(key, 0)
        s = sim.get(key, 0)
        if s != 0:
            err = abs(m - s) / abs(s) * 100
        else:
            err = 0
        print(f"{key:<20} {m:>12.6f} {s:>12.6f} {err:>9.1f}%")

    print()
    print("Recommended Python sim constants:")
    print(f"  GRAVITY      = {measured.get('gravity', -0.08):.6f}")
    print(f"  JUMP_VEL     = {measured.get('jump_velocity', 0.42):.6f}")
    print(f"  FRICTION     = {measured.get('friction', 0.91):.6f}")


# ─────────────────────────────────────────────────────────
#  Per-tick simulator validation
# ─────────────────────────────────────────────────────────

def validate_sim(df: pd.DataFrame, sim: MCSimulator = None) -> dict:
    """Validate simulator against recorded CSV per-tick.

    For each tick, feeds pre-state + actions into the simulator and
    compares the predicted post-state with the actual recorded post-state.

    Returns dict with RMSE per field.
    """
    # Determine arena bounds from data to avoid wall collision artifacts
    x_range = max(abs(df["pre_x"].max()), abs(df["pre_x"].min()))
    z_range = max(abs(df["pre_z"].max()), abs(df["pre_z"].min()))
    arena_half = max(x_range, z_range) + 100.0  # generous padding
    floor_y = df["pre_y"].min() - 5.0  # below lowest recorded Y

    if sim is None:
        sim = MCSimulator(arena_half_size=arena_half, arena_floor_y=floor_y)
    else:
        sim.arena_half_size = arena_half
        sim.arena_floor_y = floor_y

    # Filter to non-combat ticks for pure movement validation
    # (combat adds KB which makes position prediction harder without full combat sim)
    move_mask = (df["dmg_taken"] == 0) & (df["dmg_dealt"] == 0)
    move_df = df[move_mask]
    n = len(move_df)
    if n == 0:
        print("  No non-combat ticks to validate")
        return {}

    print(f"\n  Validating {n} movement-only ticks...")

    # Pre-allocate error arrays
    pos_errors = np.zeros(n)
    vx_errors = np.zeros(n)
    vy_errors = np.zeros(n)
    vz_errors = np.zeros(n)
    y_errors = np.zeros(n)

    for i, (_, row) in enumerate(move_df.iterrows()):
        pre = state_from_csv_row(row, "pre")
        actions = actions_from_csv_row(row)

        # Get block friction from recording
        friction_val = float(row.get("block_friction", GROUND_FRICTION))

        # Simulate one tick
        predicted = sim.step_movement(pre, actions, friction=friction_val)

        # Compare with actual post-state
        actual_x = float(row["post_x"])
        actual_y = float(row["post_y"])
        actual_z = float(row["post_z"])
        actual_vx = float(row["post_vx"])
        actual_vy = float(row["post_vy"])
        actual_vz = float(row["post_vz"])

        dx = predicted.x - actual_x
        dy = predicted.y - actual_y
        dz = predicted.z - actual_z
        pos_errors[i] = np.sqrt(dx**2 + dy**2 + dz**2)
        vx_errors[i] = predicted.vx - actual_vx
        vy_errors[i] = predicted.vy - actual_vy
        vz_errors[i] = predicted.vz - actual_vz
        y_errors[i] = dy

    results = {
        "position_rmse": float(np.sqrt(np.mean(pos_errors**2))),
        "vx_rmse": float(np.sqrt(np.mean(vx_errors**2))),
        "vy_rmse": float(np.sqrt(np.mean(vy_errors**2))),
        "vz_rmse": float(np.sqrt(np.mean(vz_errors**2))),
        "y_rmse": float(np.sqrt(np.mean(y_errors**2))),
        "position_mean_error": float(np.mean(pos_errors)),
        "position_max_error": float(np.max(pos_errors)),
        "n_ticks": n,
    }

    print(f"  Position RMSE:  {results['position_rmse']:.6f} blocks/tick")
    print(f"  Position mean:  {results['position_mean_error']:.6f} blocks/tick")
    print(f"  Position max:   {results['position_max_error']:.6f} blocks/tick")
    print(f"  Velocity RMSE:  vx={results['vx_rmse']:.6f}  vy={results['vy_rmse']:.6f}  vz={results['vz_rmse']:.6f}")

    return results


def optimize_params(df: pd.DataFrame) -> dict:
    """Optimize simulator parameters to minimize position RMSE.

    Uses scipy.optimize.minimize to tune gravity, drag, friction, etc.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("  scipy not installed, skipping optimization")
        return {}

    move_mask = (df["dmg_taken"] == 0) & (df["dmg_dealt"] == 0)
    move_df = df[move_mask]
    if len(move_df) < 50:
        print("  Not enough movement ticks for optimization")
        return {}

    # Subsample for speed
    if len(move_df) > 500:
        move_df = move_df.sample(500, random_state=42)

    # Determine arena bounds from data
    x_range = max(abs(move_df["pre_x"].max()), abs(move_df["pre_x"].min()))
    z_range = max(abs(move_df["pre_z"].max()), abs(move_df["pre_z"].min()))
    arena_half = max(x_range, z_range) + 100.0
    floor_y = move_df["pre_y"].min() - 5.0

    def objective(params):
        sim = MCSimulator(
            gravity=params[0],
            air_drag=params[1],
            ground_friction=params[2],
            drag_factor=params[3],
            jump_impulse=params[4],
            arena_half_size=arena_half,
            arena_floor_y=floor_y,
        )
        total_error = 0.0
        for _, row in move_df.iterrows():
            pre = state_from_csv_row(row, "pre")
            actions = actions_from_csv_row(row)
            friction_val = float(row.get("block_friction", GROUND_FRICTION))
            pred = sim.step_movement(pre, actions, friction=friction_val)

            dx = pred.x - float(row["post_x"])
            dy = pred.y - float(row["post_y"])
            dz = pred.z - float(row["post_z"])
            total_error += dx**2 + dy**2 + dz**2

        return total_error / len(move_df)

    x0 = [GRAVITY, AIR_DRAG, GROUND_FRICTION, DRAG_FACTOR, JUMP_IMPULSE]
    bounds = [
        (0.05, 0.12),    # gravity
        (0.90, 1.00),    # air_drag
        (0.40, 0.80),    # ground_friction
        (0.85, 0.95),    # drag_factor
        (0.35, 0.50),    # jump_impulse
    ]

    print(f"\n  Optimizing over {len(move_df)} ticks...")
    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                      options={"maxiter": 100, "ftol": 1e-10})

    optimized = {
        "gravity": result.x[0],
        "air_drag": result.x[1],
        "ground_friction": result.x[2],
        "drag_factor": result.x[3],
        "jump_impulse": result.x[4],
        "final_mse": result.fun,
        "success": result.success,
    }

    print(f"  Optimization {'succeeded' if result.success else 'failed'}")
    print(f"  Final MSE: {result.fun:.8f}")
    print(f"  Optimized constants:")
    for k in ["gravity", "air_drag", "ground_friction", "drag_factor", "jump_impulse"]:
        print(f"    {k} = {optimized[k]:.6f}")

    return optimized


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python -m training.calibrate_physics <recording.csv> [--validate] [--optimize]")
        print("  Generate a recording with: /mai record start 60")
        sys.exit(1)

    csv_path = args[0]
    do_validate = "--validate" in args
    do_optimize = "--optimize" in args

    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    df = load_data(csv_path)
    df = compute_displacement(df)

    print("\n--- Movement ---")
    walk = measure_walk_speed(df)
    sprint = measure_sprint_speed(df)
    strafe = measure_strafe_speed(df)

    print("\n--- Vertical ---")
    gravity = measure_gravity(df)
    jump_vel = measure_jump_velocity(df)

    print("\n--- Friction ---")
    friction = measure_friction(df)

    print("\n--- Knockback ---")
    kb = measure_knockback(df)

    measured = {
        "walk_speed": walk,
        "sprint_speed": sprint,
        "strafe_speed": strafe,
        "gravity": gravity,
        "jump_velocity": jump_vel,
        "friction": friction,
    }

    compare_with_sim(measured)

    # Per-tick validation
    if do_validate or do_optimize:
        print("\n--- Simulator Validation ---")
        results = validate_sim(df)
        measured["validation"] = results

    # Parameter optimization
    if do_optimize:
        print("\n--- Parameter Optimization ---")
        opt_results = optimize_params(df)
        if opt_results:
            measured["optimized_params"] = opt_results

            # Re-validate with optimized params
            print("\n--- Validation with Optimized Params ---")
            opt_sim = MCSimulator(
                gravity=opt_results["gravity"],
                air_drag=opt_results["air_drag"],
                ground_friction=opt_results["ground_friction"],
                drag_factor=opt_results["drag_factor"],
                jump_impulse=opt_results["jump_impulse"],
            )
            opt_validation = validate_sim(df, opt_sim)
            measured["optimized_validation"] = opt_validation

    # Save results
    out_path = Path(csv_path).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump({**measured, "knockback": kb}, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
