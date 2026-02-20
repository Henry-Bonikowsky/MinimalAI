#!/usr/bin/env python3
"""
Physics calibration script for MinimalAI.

Reads a CSV recorded by PhysicsRecorder (from /mai record) and derives
the actual server physics constants by analyzing real bot movement data.

Usage:
    python training/calibrate_physics.py plugins/MinimalAI/recordings/physics_XXX.csv

Outputs:
    - Measured walk speed, sprint speed (blocks/tick)
    - Friction coefficient
    - Gravity acceleration
    - Jump initial velocity
    - Knockback vectors
    - Comparison with current sim values
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path


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
        & (df["pre_sprinting"] == 1)  # actually sprinting (not just requesting)
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

    # Gravity = change in vertical velocity per tick
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
        & (df["pre_vx"].abs() + df["pre_vz"].abs() > 0.001)  # was moving
    )
    decelerating = df[mask]
    if len(decelerating) < 3:
        print(f"  WARNING: Only {len(decelerating)} deceleration ticks found")
        return 0.0

    # Friction ratio: post_speed / pre_speed
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

    # Support both old (kb_x/y/z) and new (hit_vel_x/y/z) column names
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
    # Current sim values from ActionExecutor.java
    sim = {
        "walk_speed": 0.1,
        "sprint_speed": 0.26,
        "gravity": -0.08,        # vanilla Minecraft
        "jump_velocity": 0.42,   # vanilla Minecraft
        "friction": 0.91,        # vanilla ground friction
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
    print("Recommended ActionExecutor constants:")
    print(f"  WALK_SPEED   = {measured.get('walk_speed', 0.1):.4f}f;")
    print(f"  SPRINT_SPEED = {measured.get('sprint_speed', 0.26):.4f}f;")
    print()
    print("Recommended Python sim constants:")
    print(f"  WALK_SPEED   = {measured.get('walk_speed', 0.1):.6f}")
    print(f"  SPRINT_SPEED = {measured.get('sprint_speed', 0.26):.6f}")
    print(f"  GRAVITY      = {measured.get('gravity', -0.08):.6f}")
    print(f"  JUMP_VEL     = {measured.get('jump_velocity', 0.42):.6f}")
    print(f"  FRICTION     = {measured.get('friction', 0.91):.6f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate_physics.py <recording.csv>")
        print("  Generate a recording with: /mai record start 60")
        sys.exit(1)

    csv_path = sys.argv[1]
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

    # Save results to JSON for programmatic use
    import json
    out_path = Path(csv_path).with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump({**measured, "knockback": kb}, f, indent=2, default=float)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
