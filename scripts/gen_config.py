#!/usr/bin/env python3
"""Generate Java and Python constants from dimensions.json.

Usage:
    python scripts/gen_config.py

Reads dimensions.json from the project root and generates:
  - src/main/java/com/minimalai/ai/ActionSpace.java
  - src/main/java/com/minimalai/ai/ObservationSpace.java
  - training/dimensions_generated.py
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DIMENSIONS_FILE = os.path.join(PROJECT_ROOT, "dimensions.json")
ACTION_SPACE_JAVA = os.path.join(
    PROJECT_ROOT, "src/main/java/com/minimalai/ai/ActionSpace.java"
)
OBS_SPACE_JAVA = os.path.join(
    PROJECT_ROOT, "src/main/java/com/minimalai/ai/ObservationSpace.java"
)
DIMENSIONS_PY = os.path.join(PROJECT_ROOT, "training/dimensions_generated.py")


def load_config():
    with open(DIMENSIONS_FILE, "r") as f:
        return json.load(f)


def generate_action_space_java(cfg):
    a = cfg["action_space"]
    groups = a["groups"]

    lines = []
    lines.append("// AUTO-GENERATED from dimensions.json -- DO NOT EDIT DIRECTLY")
    lines.append("package com.minimalai.ai;")
    lines.append("")
    lines.append("/**")
    lines.append(
        " * Constants matching the %d multi-binary action space from training/combat_sim/env.py."
        % a["num_actions"]
    )
    lines.append(" *")
    lines.append(
        " * Each action is an independent binary toggle (multi-binary, not one-hot)."
    )
    lines.append(" * The neural network outputs a probability for each; the executor reads")
    lines.append(" * int[] actions where actions[i] is 0 or 1.")
    lines.append(" */")
    lines.append("public final class ActionSpace {")
    lines.append("")

    # Movement
    mv = groups["movement"]
    lines.append("    // --- Movement (0-6) ---")
    for act in mv["actions"]:
        name = act["name"]
        idx = act["index"]
        java_name = "ACT_" + name
        lines.append(
            "    public static final int %-18s = %d;" % (java_name, idx)
        )
    lines.append("")

    # Combat
    cb = groups["combat"]
    lines.append("    // --- Combat (7-13) ---")
    for act in cb["actions"]:
        name = act["name"]
        idx = act["index"]
        java_name = "ACT_" + name
        comment = act.get("comment", "")
        comment_str = "   // " + comment if comment else ""
        lines.append(
            "    public static final int %-18s = %d;%s" % (java_name, idx, comment_str)
        )
    lines.append("")

    # Sigils
    sg = groups["sigils"]
    lines.append(
        "    // --- Sigil abilities (%d-%d, %d slots) ---"
        % (sg["start"], sg["start"] + sg["count"] - 1, sg["count"])
    )
    for i in range(sg["count"]):
        idx = sg["start"] + i
        pad = "%-18s" % ("ACT_SIGIL_%d" % i)
        lines.append("    public static final int %s = %d;" % (pad, idx))
    lines.append("")

    # Camera
    cam = groups["camera"]
    lines.append("    // --- Camera intent (26-29) ---")
    lines.append(
        "    // Priority: ENGAGE/FACE_TARGET > FACE_AWAY > LOOK_DOWN_SELF > FACE_MOVEMENT"
    )
    for act in cam["actions"]:
        name = act["name"]
        idx = act["index"]
        java_name = "ACT_" + name
        comment = act.get("comment", "")
        comment_str = "  // " + comment if comment else ""
        lines.append(
            "    public static final int %-18s = %d;%s" % (java_name, idx, comment_str)
        )
    # Aliases
    for alias_name, alias_idx in cam.get("aliases", {}).items():
        java_alias = "ACT_" + alias_name
        lines.append(
            "    public static final int %-18s = %d;  // alias for ENGAGE (backward compat)"
            % (java_alias, alias_idx)
        )
    lines.append("")

    # Target selection
    ts = groups["target_selection"]
    lines.append(
        "    // --- Target selection (%d-%d, up to %d nearby entities) ---"
        % (ts["start"], ts["start"] + ts["count"] - 1, ts["count"])
    )
    for i in range(ts["count"]):
        idx = ts["start"] + i
        pad = "%-18s" % ("ACT_TARGET_%d" % i)
        lines.append("    public static final int %s = %d;" % (pad, idx))
    lines.append("")

    # Aggregate constants
    lines.append(
        "    public static final int NUM_ACTIONS = %d;" % a["num_actions"]
    )
    lines.append("")
    lines.append("    /** Number of sigil ability slots. */")
    lines.append(
        "    public static final int NUM_SIGIL_SLOTS = %d;" % a["num_sigil_slots"]
    )
    lines.append("")
    lines.append("    /** Number of target selection slots. */")
    lines.append(
        "    public static final int NUM_TARGET_SLOTS = %d;" % a["num_target_slots"]
    )
    lines.append("")
    lines.append("    /** Smoothing factor for look transitions (1.0 = instant snap). */")
    lines.append(
        "    public static final float LOOK_SNAP_FACTOR = %sf;"
        % format_float(a["look_snap_factor"])
    )
    lines.append("")
    lines.append("    private ActionSpace() {}")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_obs_space_java(cfg):
    o = cfg["observation_space"]
    n = cfg["network"]

    lines = []
    lines.append("// AUTO-GENERATED from dimensions.json -- DO NOT EDIT DIRECTLY")
    lines.append("package com.minimalai.ai;")
    lines.append("")
    lines.append("/**")
    lines.append(" * Constants defining observation tensor dimensions.")
    lines.append(" * All values must match training/combat_sim/env.py exactly.")
    lines.append(" */")
    lines.append("public final class ObservationSpace {")
    lines.append(
        "    public static final int MAX_ENTITIES = %d;" % o["max_entities"]
    )
    lines.append(
        "    public static final int SELF_STATE_DIM = %d;" % o["self_state_dim"]
    )
    lines.append(
        "    public static final int ENTITY_FEATURE_DIM = %d;"
        % o["entity_feature_dim"]
    )
    lines.append(
        "    public static final int COMBAT_CTX_DIM = %d;" % o["combat_ctx_dim"]
    )
    lines.append(
        "    public static final int NUM_SIGIL_SLOTS = %d;" % o["num_sigil_slots"]
    )
    lines.append(
        "    public static final int SIGIL_STATE_DIM = NUM_SIGIL_SLOTS * %d; // %d"
        % (o["sigil_dims_per_slot"], o["num_sigil_slots"] * o["sigil_dims_per_slot"])
    )
    lines.append(
        "    public static final int ENV_STATE_DIM = %d;" % o["env_state_dim"]
    )
    lines.append(
        "    public static final int HIDDEN_DIM = %d;" % n["gru_hidden_dim"]
    )
    lines.append("")
    lines.append("    private ObservationSpace() {}")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_dimensions_py(cfg):
    a = cfg["action_space"]
    groups = a["groups"]
    o = cfg["observation_space"]
    n = cfg["network"]
    cons = cfg["consumables"]
    combat = cfg["combat"]
    sigils = cfg["sigils"]
    used_bits = cfg["used_bits"]
    used_bits_comment = cfg.get("used_bits_comment", "")

    lines = []
    lines.append(
        '"""AUTO-GENERATED from dimensions.json -- DO NOT EDIT DIRECTLY."""'
    )
    lines.append("")
    lines.append("import math")
    lines.append("")

    # Action space
    lines.append("# " + "=" * 60)
    lines.append("#  Action Space (matches Java ActionSpace.java)")
    lines.append("# " + "=" * 60)
    lines.append("")

    # Movement
    lines.append("# Movement (0-6)")
    for act in groups["movement"]["actions"]:
        py_name = "ACT_" + act["name"]
        # Use Python aliases: STRAFE_LEFT -> LEFT, STRAFE_RIGHT -> RIGHT
        if act["name"] == "STRAFE_LEFT":
            py_name = "ACT_LEFT"
            comment = "   # strafe left"
        elif act["name"] == "STRAFE_RIGHT":
            py_name = "ACT_RIGHT"
            comment = "   # strafe right"
        else:
            comment = ""
        lines.append("%-18s = %d%s" % (py_name, act["index"], comment))
    lines.append("")

    # Combat
    lines.append("# Combat (7-13)")
    for act in groups["combat"]["actions"]:
        py_name = "ACT_" + act["name"]
        comment_text = act.get("comment", "")
        comment = "   # " + comment_text if comment_text else ""
        lines.append("%-18s = %d%s" % (py_name, act["index"], comment))
    lines.append("")

    # Sigils
    sg = groups["sigils"]
    lines.append("# Sigil abilities (%d-%d)" % (sg["start"], sg["start"] + sg["count"] - 1))
    lines.append("%-18s = %d" % ("ACT_SIGIL_0", sg["start"]))
    lines.append("%-18s = %d" % ("ACT_SIGIL_11", sg["start"] + sg["count"] - 1))
    lines.append("")

    # Sigil constants
    lines.append("# Sigil constants")
    b = sigils["brace"]
    lines.append("SIGIL_BRACE_CHARGE_REQ = %d" % b["charge_req"])
    lines.append("SIGIL_BRACE_DR = %s" % format_py_float(b["dr"]))
    lines.append("SIGIL_BRACE_DURATION = %d" % b["duration"])
    lines.append("SIGIL_BRACE_COOLDOWN = %d" % b["cooldown"])
    c = sigils["cleo"]
    lines.append("SIGIL_CLEO_DMG_AMP = %s" % format_py_float(c["dmg_amp"]))
    lines.append("SIGIL_CLEO_DURATION = %d" % c["duration"])
    lines.append("SIGIL_CLEO_COOLDOWN = %d" % c["cooldown"])
    s = sigils["sand"]
    lines.append("SIGIL_SAND_DMG_AMP = %s" % format_py_float(s["dmg_amp"]))
    lines.append("SIGIL_SAND_DURATION = %d" % s["duration"])
    lines.append("SIGIL_SAND_COOLDOWN = %d" % s["cooldown"])
    g = sigils["grace"]
    lines.append("SIGIL_GRACE_DR = %s" % format_py_float(g["dr"]))
    lines.append("SIGIL_GRACE_DURATION = %d" % g["duration"])
    lines.append("SIGIL_GRACE_COOLDOWN = %d" % g["cooldown"])
    lines.append("")

    # Camera
    cam = groups["camera"]
    lines.append("# Camera intent (26-29)")
    for act in cam["actions"]:
        py_name = "ACT_" + act["name"]
        # Python uses shorter aliases
        if act["name"] == "LOOK_DOWN_SELF":
            py_name = "ACT_LOOK_DOWN"
        elif act["name"] == "FACE_MOVEMENT":
            py_name = "ACT_FACE_MOVE"
        comment_text = act.get("comment", "")
        comment = "  # " + comment_text if comment_text else ""
        lines.append("%-18s = %d%s" % (py_name, act["index"], comment))
    # Aliases
    for alias_name, alias_idx in cam.get("aliases", {}).items():
        py_alias = "ACT_" + alias_name
        lines.append(
            "%-18s = %d  # alias for backward compat" % (py_alias, alias_idx)
        )
    lines.append("")

    # Target selection
    ts = groups["target_selection"]
    lines.append("# Target selection (%d-%d)" % (ts["start"], ts["start"] + ts["count"] - 1))
    lines.append("%-18s = %d" % ("ACT_TARGET_0", ts["start"]))
    lines.append("%-18s = %d" % ("ACT_TARGET_4", ts["start"] + ts["count"] - 1))
    lines.append("")

    lines.append("%-18s = %d" % ("NUM_ACTIONS", a["num_actions"]))
    lines.append("")

    # Used bits
    lines.append("# " + "=" * 60)
    lines.append("#  Model-Controlled Bits")
    lines.append("#  Only these bits are output by the neural net.")
    lines.append("#  Everything else is forced, masked, or auto-derived.")
    lines.append("# " + "=" * 60)
    lines.append("")
    lines.append("USED_BITS = %s" % repr(used_bits))
    lines.append("# %s" % used_bits_comment)
    lines.append("#")
    lines.append("# Forced:  SPRINT(6)=always on, SNEAK(5)=always off")
    lines.append("# Auto:    ATTACK(7)=derived from ENGAGE + reach")
    lines.append("#          BACKWARD(1)=injected by combo style after sprint-hit")
    lines.append("# Masked:  BLOCK(8), SPRINT_RESET(12), SWAP(13),")
    lines.append("#          SIGILS(18-25), CAMERA(27-29), TARGETS(30-34)")
    lines.append("")

    # Observation space
    lines.append("# " + "=" * 60)
    lines.append("#  Observation Space")
    lines.append("# " + "=" * 60)
    lines.append("")
    lines.append("%-18s = %d" % ("SELF_STATE_DIM", o["self_state_dim"]))
    lines.append("%-18s = %d" % ("ENTITY_FEATURE_DIM", o["entity_feature_dim"]))
    lines.append("%-18s = %d" % ("MAX_ENTITIES", o["max_entities"]))
    lines.append("%-18s = %d" % ("COMBAT_CTX_DIM", o["combat_ctx_dim"]))
    sigil_state_dim = o["num_sigil_slots"] * o["sigil_dims_per_slot"]
    lines.append(
        "%-18s = %d   # %d slots * %d dims"
        % ("SIGIL_STATE_DIM", sigil_state_dim, o["num_sigil_slots"], o["sigil_dims_per_slot"])
    )
    lines.append("%-18s = %d" % ("ENV_STATE_DIM", o["env_state_dim"]))
    lines.append("%-18s = %d  # total flat obs" % ("OBS_DIM", o["obs_dim"]))
    lines.append("")

    # Self-state slot map
    lines.append("# self_state slot map (0-37):")
    lines.append("#  0  health/20           1  maxHealth/20         2  absorption/20")
    lines.append("#  3  armor/20            4  armorToughness/20    5-7  velocity xyz")
    lines.append("#  8  yaw/180             9  pitch/90            10  onGround")
    lines.append("# 11  isSprinting        12  isSneaking          13  isBlocking")
    lines.append("# 14  isUsingItem        15  fallDistance/10      16-18  weapon one-hot")
    lines.append("# 19  attackDamage/20    20  goldenApples/64     21  splashPots/64")
    lines.append("# 22  enderPearls/16     23  totems              24  gappleCooldown *")
    lines.append("# 25  eatingProgress *   26  potLockout *        27  pearlCooldown *")
    lines.append("# 28  regenActive *      29  regenRemaining *    30  ownComboStreak *")
    lines.append("# 31  enemyComboStreak * 32  (unused)            33  hurtTime/10")
    lines.append("# 34-37  sigil combat state (Java only)")
    lines.append("# (* = sim-only, Java uses 24-32 for status effects)")
    lines.append("")

    # Entity features slot map
    lines.append("# entity_features slot map (per entity, 24 dims):")
    lines.append("#  0  alliance        1-3  dx/dy/dz /30     4  dist/30")
    lines.append("#  5  health/20        6  maxHealth/20       7  armor/20")
    lines.append("#  8-10  velocity xyz  11  yaw/180          12  onGround")
    lines.append("# 13  isSprinting     14  isBlocking        15  isUsingItem")
    lines.append("# 16-18  weapon 1-hot  19  attackDmg/20     20  hurtTime/10")
    lines.append("# 21  isCurrentTarget  22-23  (unused)")
    lines.append("")

    # Network arch
    lines.append("# " + "=" * 60)
    lines.append("#  Network Architecture")
    lines.append("# " + "=" * 60)
    lines.append("")
    lines.append("%-18s = %d" % ("ENTITY_EMBED_DIM", n["entity_embed_dim"]))
    lines.append("%-18s = %d" % ("ATTENTION_HEADS", n["attention_heads"]))
    lines.append("%-18s = %d" % ("ATTENTION_LAYERS", n["attention_layers"]))
    lines.append("%-18s = %d" % ("HIDDEN_DIM", n["hidden_dim"]))
    lines.append("%-18s = %d" % ("GRU_HIDDEN_DIM", n["gru_hidden_dim"]))
    lines.append("")

    # Consumables
    lines.append("# " + "=" * 60)
    lines.append("#  Consumable Constants (match server kit)")
    lines.append("# " + "=" * 60)
    lines.append("")

    gap = cons["golden_apple"]
    lines.append("# Golden apples")
    lines.append("%-20s = %d    # 1.8s eat time" % ("GAP_EAT_TICKS", gap["eat_ticks"]))
    lines.append("%-20s = %d   # 9s cooldown" % ("GAP_COOLDOWN_TICKS", gap["cooldown_ticks"]))
    lines.append("%-20s = %s   # absorption HP gained" % ("GAP_ABSORPTION", format_py_float(gap["absorption"])))
    lines.append("%-20s = %d   # regen II duration" % ("GAP_REGEN_TICKS", gap["regen_ticks"]))
    lines.append("%-20s = %d    # 1 HP every 25 ticks" % ("GAP_REGEN_RATE", gap["regen_rate"]))
    lines.append("%-20s = %d    # per fight" % ("GAP_START_COUNT", gap["start_count"]))
    lines.append("")

    pot = cons["splash_pot"]
    lines.append("# Splash potions (Instant Health II)")
    lines.append("%-20s = %s" % ("POT_HEAL_AMOUNT", format_py_float(pot["heal_amount"])))
    lines.append("%-20s = %s" % ("POT_SPLASH_RADIUS", format_py_float(pot["splash_radius"])))
    lines.append("%-20s = %d     # ticks can't attack during throw" % ("POT_THROW_LOCKOUT", pot["throw_lockout"]))
    lines.append("%-20s = %d" % ("POT_START_COUNT", pot["start_count"]))
    lines.append("")

    pearl = cons["ender_pearl"]
    lines.append("# Ender pearls")
    lines.append("%-20s = %d  # 10s cooldown" % ("PEARL_COOLDOWN_TICKS", pearl["cooldown_ticks"]))
    lines.append("%-20s = %s" % ("PEARL_SELF_DAMAGE", format_py_float(pearl["self_damage"])))
    lines.append("%-20s = %d" % ("PEARL_START_COUNT", pearl["start_count"]))
    lines.append("")

    # Combat constants
    lines.append("# " + "=" * 60)
    lines.append("#  Combat Constants")
    lines.append("# " + "=" * 60)
    lines.append("")
    lines.append("%-20s = %d" % ("MELEE_ANGLE_DEG", combat["melee_angle_deg"]))
    lines.append("MELEE_ANGLE_COS     = math.cos(math.radians(MELEE_ANGLE_DEG))")
    lines.append("%-20s = %s" % ("DEFAULT_ATTACK_REACH", format_py_float(combat["default_attack_reach"])))
    lines.append("%-20s = %d" % ("I_FRAME_TICKS", combat["i_frame_ticks"]))
    lines.append("")

    return "\n".join(lines)


def format_float(v):
    """Format a float for Java (e.g., 1.0 not 1)."""
    s = str(v)
    if "." not in s:
        s += ".0"
    return s


def format_py_float(v):
    """Format a float for Python (e.g., 4.0 not 4). Preserves trailing zeros."""
    # If it came from JSON as e.g. 0.8, we need to keep at least one decimal
    s = str(v)
    if "." not in s:
        s += ".0"
    return s


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(content)
    print(f"  Generated: {os.path.relpath(path, PROJECT_ROOT)}")


def main():
    print("gen_config: reading dimensions.json")
    cfg = load_config()

    print("gen_config: generating files...")
    write_file(ACTION_SPACE_JAVA, generate_action_space_java(cfg))
    write_file(OBS_SPACE_JAVA, generate_obs_space_java(cfg))
    write_file(DIMENSIONS_PY, generate_dimensions_py(cfg))

    print("gen_config: done.")


if __name__ == "__main__":
    main()
