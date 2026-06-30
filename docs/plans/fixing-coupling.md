# How to Fix Coupling: Concrete Examples

This doc shows exactly what to change to implement the 4 priority fixes.

---

## PRIORITY 1: Add Model Metadata (20 min)

### Step 1: Update fast_train.py (Export models with metadata)

**Location:** `training/fast_train.py`, in the export/checkpoint saving code

**Add this after training completes (around line 900):**

```python
def export_model_with_metadata(model, output_path):
    """Export traced model with dimension metadata."""
    from .config import (
        NUM_ACTIONS, SELF_STATE_DIM, ENTITY_FEATURE_DIM, MAX_ENTITIES,
        COMBAT_CTX_DIM, SIGIL_STATE_DIM, ENV_STATE_DIM, GRU_HIDDEN_DIM,
    )

    # Trace the model as before
    traced = torch.jit.trace(model, example_inputs, check_trace=False)

    # Create metadata dict
    meta = {
        "version": "1.0",
        "framework": "pytorch",
        "architecture": "CombatNetwork",
        "action_space": NUM_ACTIONS,
        "observations": {
            "self_state_dim": SELF_STATE_DIM,
            "entity_feature_dim": ENTITY_FEATURE_DIM,
            "max_entities": MAX_ENTITIES,
            "combat_ctx_dim": COMBAT_CTX_DIM,
            "sigil_state_dim": SIGIL_STATE_DIM,
            "env_state_dim": ENV_STATE_DIM,
        },
        "hidden": {
            "gru_hidden_dim": GRU_HIDDEN_DIM,
        },
    }

    # Save as dict (TorchScript model + metadata)
    torch.save({
        "model": traced,
        "metadata": meta,
    }, output_path)


# In main training loop, replace:
# torch.save(traced_model, checkpoint_path)
# with:
export_model_with_metadata(traced_model, checkpoint_path)
```

### Step 2: Update ModelManager.java (Load and validate metadata)

**Location:** `src/main/java/com/minimalai/ai/ModelManager.java`

**Add a metadata validator class:**

```java
public static class ModelMetadata {
    public String version;
    public int actionSpace;
    public int selfStateDim;
    public int entityFeatureDim;
    public int maxEntities;
    public int combatCtxDim;
    public int sigilStateDim;
    public int envStateDim;
    public int gruHiddenDim;

    // Validate against current Java config
    public void validateOrThrow() throws IllegalStateException {
        if (actionSpace != 35) {
            throw new IllegalStateException(
                "Model expects " + actionSpace + " actions, but Java is configured for 35");
        }
        if (selfStateDim != 38) {
            throw new IllegalStateException(
                "Model expects self_state=" + selfStateDim + ", Java expects 38");
        }
        if (entityFeatureDim != 24) {
            throw new IllegalStateException(
                "Model expects entity_feature=" + entityFeatureDim + ", Java expects 24");
        }
        if (gruHiddenDim != 128) {
            throw new IllegalStateException(
                "Model expects gru_hidden=" + gruHiddenDim + ", Java expects 128");
        }
        // ... validate all dimensions
        logger.info("Model metadata validated: v" + version);
    }
}
```

**Update loadModel():**

```java
public void loadModel(String name) throws ModelNotFoundException, MalformedModelException, IOException {
    Path modelPath = modelsDir.resolve(name + ".pt");
    if (!Files.exists(modelPath)) {
        throw new IOException("Model file not found: " + modelPath);
    }

    ClassLoader original = Thread.currentThread().getContextClassLoader();
    Thread.currentThread().setContextClassLoader(getClass().getClassLoader());
    try {
        // Load with metadata extraction
        Object loaded = torch.load(modelPath.toString());  // PyTorch 1.13+
        ZooModel<NDList, NDList> model = extractModel(loaded);
        ModelMetadata meta = extractMetadata(loaded);

        // Validate before storing
        if (meta != null) {
            meta.validateOrThrow();
        } else {
            logger.warning("Model " + name + " has no metadata; skipping validation");
        }

        ZooModel<NDList, NDList> old = loadedModels.put(name, model);
        if (old != null) {
            old.close();
        }

        logger.info("Loaded model: " + name + " from " + modelPath);
    } finally {
        Thread.currentThread().setContextClassLoader(original);
    }
}

private ModelMetadata extractMetadata(Object loaded) {
    // Parse torch.load() dict output into ModelMetadata
    // Implementation depends on torch.load() return type
    // For now, return null (validation optional in v1)
    return null;
}

private ZooModel<NDList, NDList> extractModel(Object loaded) {
    // Extract TorchScript model from torch.load() dict
    // For now, assume it's the model directly
    return (ZooModel<NDList, NDList>) loaded;
}
```

**Cost:** ~50 lines, ~20 minutes

---

## PRIORITY 2: Config-Driven Dimensions (2-3 hours)

### Step 1: Create dimensions.json

**Location:** `dimensions.json` (root of project)

```json
{
  "version": "1.0",
  "description": "Single source of truth for all action/observation dimensions",

  "actions": {
    "total": 35,
    "movement": {
      "indices": [0, 6],
      "actions": [
        {"name": "forward", "index": 0, "forced": false, "masked": false},
        {"name": "backward", "index": 1, "forced": false, "masked": false},
        {"name": "strafe_left", "index": 2, "forced": false, "masked": false},
        {"name": "strafe_right", "index": 3, "forced": false, "masked": false},
        {"name": "jump", "index": 4, "forced": false, "masked": false},
        {"name": "sneak", "index": 5, "forced": false, "masked": false},
        {"name": "sprint", "index": 6, "forced": true, "masked": false}
      ]
    },
    "combat": {
      "indices": [7, 13],
      "actions": [
        {"name": "attack", "index": 7, "forced": false, "masked": false, "auto_derived_from": "engage"},
        {"name": "block", "index": 8, "forced": false, "masked": true},
        {"name": "eat_gap", "index": 9, "forced": false, "masked": false},
        {"name": "throw_pot", "index": 10, "forced": false, "masked": false},
        {"name": "throw_pearl", "index": 11, "forced": false, "masked": false},
        {"name": "sprint_reset", "index": 12, "forced": false, "masked": true},
        {"name": "swap_weapon", "index": 13, "forced": false, "masked": true}
      ]
    },
    "sigils": {
      "indices": [14, 25],
      "count": 12,
      "activatable_slots": [1, 2, 3, 5]
    },
    "camera": {
      "indices": [26, 29],
      "actions": [
        {"name": "engage", "index": 26, "forced": false, "masked": false},
        {"name": "face_away", "index": 27, "forced": false, "masked": false},
        {"name": "look_down_self", "index": 28, "forced": false, "masked": false},
        {"name": "face_movement", "index": 29, "forced": false, "masked": false}
      ]
    },
    "targeting": {
      "indices": [30, 34],
      "count": 5,
      "max_nearby_entities": 5
    },
    "used_bits": [0, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 26]
  },

  "observations": {
    "self_state_dim": 38,
    "entity_feature_dim": 24,
    "max_entities": 8,
    "combat_ctx_dim": 26,
    "sigil_state_dim": 48,
    "env_state_dim": 8,
    "total_flat": 320,
    "hidden_dim": 128,

    "self_state_slots": {
      "health": 0,
      "maxhealth": 1,
      "absorption": 2,
      "armor": 3,
      "armor_toughness": 4,
      "velocity_x": 5,
      "velocity_y": 6,
      "velocity_z": 7,
      "yaw": 8,
      "pitch": 9,
      "on_ground": 10,
      "is_sprinting": 11,
      "is_sneaking": 12,
      "is_blocking": 13,
      "is_using_item": 14,
      "fall_distance": 15,
      "weapon_type_sword": 16,
      "weapon_type_axe": 17,
      "weapon_type_other": 18,
      "attack_damage": 19,
      "golden_apples": 20,
      "splash_pots": 21,
      "ender_pearls": 22,
      "totems": 23,
      "effect_speed": 24,
      "effect_strength": 25,
      "effect_resistance": 26,
      "effect_fire_resistance": 27,
      "effect_regeneration": 28,
      "effect_poison": 29,
      "effect_wither": 30,
      "effect_slowness": 31,
      "effect_weakness": 32,
      "hurt_time": 33,
      "sigil_damage_amp": 34,
      "sigil_damage_reduction": 35,
      "sigil_kb_charges": 36,
      "sigil_invuln_hits": 37
    },

    "entity_feature_slots": {
      "alliance": 0,
      "relative_x": 1,
      "relative_y": 2,
      "relative_z": 3,
      "distance": 4,
      "health": 5,
      "maxhealth": 6,
      "armor": 7,
      "velocity_x": 8,
      "velocity_y": 9,
      "velocity_z": 10,
      "yaw": 11,
      "on_ground": 12,
      "is_sprinting": 13,
      "is_blocking": 14,
      "is_using_item": 15,
      "weapon_type_sword": 16,
      "weapon_type_axe": 17,
      "weapon_type_other": 18,
      "attack_damage": 19,
      "hurt_time": 20,
      "is_current_target": 21,
      "pharaoh_mark": 22,
      "distance_to_bot_target": 23
    }
  },

  "consumables": {
    "golden_apple": {
      "start_count": 64,
      "eat_ticks": 36,
      "cooldown_ticks": 180,
      "absorption_hp": 4.0,
      "regen_ticks": 100,
      "regen_rate": 25
    },
    "splash_pot": {
      "start_count": 31,
      "heal_amount": 8.0,
      "splash_radius": 4.0,
      "throw_lockout": 3
    },
    "ender_pearl": {
      "start_count": 16,
      "cooldown_ticks": 200,
      "self_damage": 5.0
    }
  }
}
```

### Step 2: Python Config Generator

**Location:** `scripts/gen_python_config.py`

```python
#!/usr/bin/env python3
"""Generate training/dimensions_generated.py from dimensions.json"""

import json
import sys
from pathlib import Path

def generate_python_config(json_path):
    with open(json_path) as f:
        dims = json.load(f)

    lines = [
        "# AUTO-GENERATED from dimensions.json — DO NOT EDIT",
        "# Run: python scripts/gen_python_config.py",
        "",
        "# ============================================================",
        "#  Action Space",
        "# ============================================================",
        "",
    ]

    # Generate action constants
    for group_name, group in dims["actions"].items():
        if group_name in ["total", "used_bits"]:
            continue
        if "actions" in group:
            for action in group["actions"]:
                lines.append(f"ACT_{action['name'].upper()} = {action['index']}")
        elif "count" in group:
            lines.append(f"NUM_{group_name.upper()}_SLOTS = {group['count']}")

    lines.extend([
        "",
        f"NUM_ACTIONS = {dims['actions']['total']}",
        f"USED_BITS = {dims['actions']['used_bits']}",
        "",
        "# ============================================================",
        "#  Observation Space",
        "# ============================================================",
        "",
    ])

    # Generate observation constants
    for key, value in dims["observations"].items():
        if not key.startswith("_") and isinstance(value, int):
            lines.append(f"{key.upper()} = {value}")

    lines.extend([
        "",
        "# ============================================================",
        "#  Consumables",
        "# ============================================================",
        "",
    ])

    # Generate consumable constants
    for item_name, item_config in dims["consumables"].items():
        prefix = item_name.upper().replace("_", "_").replace("_", "_")
        for key, value in item_config.items():
            const_name = f"{prefix}_{key.upper()}"
            if isinstance(value, float):
                lines.append(f"{const_name} = {value}")
            else:
                lines.append(f"{const_name} = {value}")

    output = "\n".join(lines)
    return output

if __name__ == "__main__":
    json_path = Path(__file__).parent.parent / "dimensions.json"
    output_path = Path(__file__).parent.parent / "training" / "dimensions_generated.py"

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        sys.exit(1)

    content = generate_python_config(json_path)
    output_path.write_text(content + "\n")
    print(f"Generated {output_path}")
```

### Step 3: Java Constants Generator

**Location:** `scripts/gen_java_constants.py`

```python
#!/usr/bin/env python3
"""Generate Java constant classes from dimensions.json"""

import json
from pathlib import Path

def generate_java_action_space(dims):
    lines = [
        "package com.minimalai.ai;",
        "",
        "/**",
        " * AUTO-GENERATED from dimensions.json — DO NOT EDIT",
        " * Run: python scripts/gen_java_constants.py",
        " */",
        "public final class ActionSpace {",
        "",
    ]

    # Movement actions
    lines.append("    // --- Movement ---")
    for action in dims["actions"]["movement"]["actions"]:
        lines.append(f"    public static final int ACT_{action['name'].upper()} = {action['index']};")

    # Combat actions
    lines.append("")
    lines.append("    // --- Combat ---")
    for action in dims["actions"]["combat"]["actions"]:
        lines.append(f"    public static final int ACT_{action['name'].upper()} = {action['index']};")

    # Sigil actions
    lines.append("")
    lines.append("    // --- Sigils ---")
    lines.append(f"    public static final int NUM_SIGIL_SLOTS = {dims['actions']['sigils']['count']};")
    for i in range(dims["actions"]["sigils"]["count"]):
        lines.append(f"    public static final int ACT_SIGIL_{i} = {dims['actions']['sigils']['indices'][0] + i};")

    # Camera actions
    lines.append("")
    lines.append("    // --- Camera Intent ---")
    for action in dims["actions"]["camera"]["actions"]:
        lines.append(f"    public static final int ACT_{action['name'].upper()} = {action['index']};")

    # Targeting
    lines.append("")
    lines.append("    // --- Target Selection ---")
    for i in range(dims["actions"]["targeting"]["count"]):
        lines.append(f"    public static final int ACT_TARGET_{i} = {dims['actions']['targeting']['indices'][0] + i};")

    lines.extend([
        "",
        f"    public static final int NUM_ACTIONS = {dims['actions']['total']};",
        f"    public static final int NUM_TARGET_SLOTS = {dims['actions']['targeting']['count']};",
        "",
        "    private ActionSpace() {}",
        "}",
    ])

    return "\n".join(lines)

def generate_java_observation_space(dims):
    obs = dims["observations"]
    lines = [
        "package com.minimalai.ai;",
        "",
        "/**",
        " * AUTO-GENERATED from dimensions.json — DO NOT EDIT",
        " * Run: python scripts/gen_java_constants.py",
        " */",
        "public final class ObservationSpace {",
        f"    public static final int SELF_STATE_DIM = {obs['self_state_dim']};",
        f"    public static final int ENTITY_FEATURE_DIM = {obs['entity_feature_dim']};",
        f"    public static final int MAX_ENTITIES = {obs['max_entities']};",
        f"    public static final int COMBAT_CTX_DIM = {obs['combat_ctx_dim']};",
        f"    public static final int SIGIL_STATE_DIM = {obs['sigil_state_dim']};",
        f"    public static final int ENV_STATE_DIM = {obs['env_state_dim']};",
        f"    public static final int HIDDEN_DIM = {obs['hidden_dim']};",
        "",
        "    private ObservationSpace() {}",
        "}",
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    json_path = Path(__file__).parent.parent / "dimensions.json"
    with open(json_path) as f:
        dims = json.load(f)

    # Generate Java files
    action_space = generate_java_action_space(dims)
    obs_space = generate_java_observation_space(dims)

    action_path = Path(__file__).parent.parent / "src/main/java/com/minimalai/ai/ActionSpace.java"
    obs_path = Path(__file__).parent.parent / "src/main/java/com/minimalai/ai/ObservationSpace.java"

    action_path.write_text(action_space + "\n")
    obs_path.write_text(obs_space + "\n")

    print(f"Generated {action_path}")
    print(f"Generated {obs_path}")
```

### Step 4: Update Build Pipeline

**Location:** `build.gradle` or `pom.xml`

Add a generation step before compilation. For Gradle:

```gradle
task generateConstants {
    doFirst {
        exec {
            commandLine 'python3', 'scripts/gen_python_config.py'
            commandLine 'python3', 'scripts/gen_java_constants.py'
        }
    }
}

compileJava.dependsOn generateConstants
```

### Step 5: Update Python to use generated config

**Location:** `training/config.py`

```python
# Import from generated dimensions file (auto-generated from dimensions.json)
from .dimensions_generated import *

# Keep everything else as-is for backward compat,
# but add note at top:
"""
Observation and action dimensions are now auto-generated from dimensions.json.
This file should not be edited directly — update dimensions.json instead.

To regenerate:
    python scripts/gen_python_config.py
"""
```

**Cost:** ~300 lines total (JSON + 2 generators), ~2-3 hours to write + test

---

## PRIORITY 3: Observation Slots Class (25 min, Quick Win)

**Add before Priority 2 to remove magic numbers immediately.**

### Step 1: Create ObservationSlots.java

**Location:** `src/main/java/com/minimalai/ai/ObservationSlots.java`

```java
package com.minimalai.ai;

/**
 * Named indices for observation tensor slots.
 * Replaces magic numbers like s[0] for health.
 * Keep synchronized with training/observations.py
 */
public final class ObservationSlots {

    // --- self_state[38] ---
    public static final int SELF_HEALTH = 0;
    public static final int SELF_MAXHEALTH = 1;
    public static final int SELF_ABSORPTION = 2;
    public static final int SELF_ARMOR = 3;
    public static final int SELF_ARMOR_TOUGHNESS = 4;
    public static final int SELF_VEL_X = 5;
    public static final int SELF_VEL_Y = 6;
    public static final int SELF_VEL_Z = 7;
    public static final int SELF_YAW = 8;
    public static final int SELF_PITCH = 9;
    public static final int SELF_ON_GROUND = 10;
    public static final int SELF_IS_SPRINTING = 11;
    public static final int SELF_IS_SNEAKING = 12;
    public static final int SELF_IS_BLOCKING = 13;
    public static final int SELF_IS_USING_ITEM = 14;
    public static final int SELF_FALL_DISTANCE = 15;
    public static final int SELF_WEAPON_SWORD = 16;
    public static final int SELF_WEAPON_AXE = 17;
    public static final int SELF_WEAPON_OTHER = 18;
    public static final int SELF_ATTACK_DAMAGE = 19;
    public static final int SELF_GAPPLES = 20;
    public static final int SELF_POTS = 21;
    public static final int SELF_PEARLS = 22;
    public static final int SELF_TOTEMS = 23;
    public static final int SELF_EFFECT_SPEED = 24;
    public static final int SELF_EFFECT_STRENGTH = 25;
    public static final int SELF_EFFECT_RESISTANCE = 26;
    public static final int SELF_EFFECT_FIRE_RESISTANCE = 27;
    public static final int SELF_EFFECT_REGENERATION = 28;
    public static final int SELF_EFFECT_POISON = 29;
    public static final int SELF_EFFECT_WITHER = 30;
    public static final int SELF_EFFECT_SLOWNESS = 31;
    public static final int SELF_EFFECT_WEAKNESS = 32;
    public static final int SELF_HURT_TIME = 33;
    public static final int SELF_SIGIL_DAMAGE_AMP = 34;
    public static final int SELF_SIGIL_DAMAGE_REDUCTION = 35;
    public static final int SELF_SIGIL_KB_CHARGES = 36;
    public static final int SELF_SIGIL_INVULN_HITS = 37;

    // --- entity_features[24] per entity ---
    public static final int ENTITY_ALLIANCE = 0;
    public static final int ENTITY_REL_X = 1;
    public static final int ENTITY_REL_Y = 2;
    public static final int ENTITY_REL_Z = 3;
    public static final int ENTITY_DISTANCE = 4;
    public static final int ENTITY_HEALTH = 5;
    public static final int ENTITY_MAXHEALTH = 6;
    public static final int ENTITY_ARMOR = 7;
    public static final int ENTITY_VEL_X = 8;
    public static final int ENTITY_VEL_Y = 9;
    public static final int ENTITY_VEL_Z = 10;
    public static final int ENTITY_YAW = 11;
    public static final int ENTITY_ON_GROUND = 12;
    public static final int ENTITY_IS_SPRINTING = 13;
    public static final int ENTITY_IS_BLOCKING = 14;
    public static final int ENTITY_IS_USING_ITEM = 15;
    public static final int ENTITY_WEAPON_SWORD = 16;
    public static final int ENTITY_WEAPON_AXE = 17;
    public static final int ENTITY_WEAPON_OTHER = 18;
    public static final int ENTITY_ATTACK_DAMAGE = 19;
    public static final int ENTITY_HURT_TIME = 20;
    public static final int ENTITY_IS_CURRENT_TARGET = 21;
    public static final int ENTITY_PHARAOH_MARK = 22;
    public static final int ENTITY_DISTANCE_TO_BOT_TARGET = 23;

    // --- combat_ctx[26] ---
    public static final int CTX_WEAPON_SWORD = 0;
    public static final int CTX_WEAPON_AXE = 1;
    public static final int CTX_WEAPON_OTHER = 2;
    public static final int CTX_GAPPLES = 3;
    public static final int CTX_POTS = 4;
    public static final int CTX_PEARLS = 5;
    public static final int CTX_TOTEMS = 6;
    public static final int CTX_DAMAGE_DEALT = 7;
    public static final int CTX_DAMAGE_TAKEN = 8;
    public static final int CTX_COMBO_COUNTER = 9;
    public static final int CTX_TIME_SINCE_HIT = 10;
    public static final int CTX_CPS_ESTIMATE = 11;
    public static final int CTX_SPRINT = 12;
    public static final int CTX_BLOCK = 13;
    public static final int CTX_EATING = 14;
    public static final int CTX_EATING_TICKS = 15;
    public static final int CTX_EFFECT_SPEED = 16;
    public static final int CTX_EFFECT_STRENGTH = 17;
    public static final int CTX_EFFECT_RESISTANCE = 18;
    public static final int CTX_EFFECT_FIRE_RESISTANCE = 19;
    public static final int CTX_EFFECT_REGENERATION = 20;
    public static final int CTX_EFFECT_POISON = 21;
    public static final int CTX_EFFECT_WITHER = 22;
    public static final int CTX_EFFECT_SLOWNESS = 23;
    public static final int CTX_EFFECT_WEAKNESS = 24;
    public static final int CTX_PEARL_COOLDOWN = 25;

    // --- env_state[8] ---
    public static final int ENV_ARENA_X = 0;
    public static final int ENV_ARENA_Z = 1;
    public static final int ENV_ARENA_Y = 2;
    public static final int ENV_HEIGHT_DIFF_TO_TARGET = 3;
    public static final int ENV_IN_ATTACK_RANGE = 4;
    public static final int ENV_ON_GROUND = 5;
    public static final int ENV_ALIVE_ENEMIES_RATIO = 6;
    public static final int ENV_EPISODE_PROGRESS = 7;

    private ObservationSlots() {}
}
```

### Step 2: Update ObservationBuilder.java to use slots

**Before:**
```java
s[0] = p.getHealth() / 20f;
s[1] = (float) p.getMaxHealth() / 20f;
s[24] = p.hasEffect(MobEffects.SPEED) ? 1f : 0f;
```

**After:**
```java
s[ObservationSlots.SELF_HEALTH] = p.getHealth() / 20f;
s[ObservationSlots.SELF_MAXHEALTH] = (float) p.getMaxHealth() / 20f;
s[ObservationSlots.SELF_EFFECT_SPEED] = p.hasEffect(MobEffects.SPEED) ? 1f : 0f;
```

**Cost:** ~200 lines (class definition + scattered updates)
**Benefit:** Immediate readability improvement, foundation for schema validation

---

## PRIORITY 4: Action Masking & USED_BITS Config

**Replace hardcoded USED_BITS in two files with data-driven approach.**

### Add to dimensions.json:

```json
"masking": {
  "used_bits": [0, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 26],
  "forced_on": [6],
  "forced_off": [5],
  "always_masked": [8, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34]
}
```

### Update ActionExecutor.buildActionMask():

```java
public static float[] buildActionMask(int[] actions) {
    float[] mask = new float[ActionSpace.NUM_ACTIONS];

    // Start: everything masked
    for (int i = 0; i < mask.length; i++) {
        mask[i] = 0f;
    }

    // Unmask used bits (from config)
    for (int bit : ActionSpace.USED_BITS) {
        mask[bit] = 1f;
    }

    return mask;
}
```

---

## Summary: What Changes Where

| File | Priority 1 | Priority 2 | Priority 3 | Priority 4 |
|------|-----------|-----------|-----------|-----------|
| fast_train.py | Add export fn | Use generated config | — | — |
| config.py | — | Switch to generated | — | Use USED_BITS from config |
| ActionSpace.java | — | Auto-generated | — | — |
| ObservationSpace.java | — | Auto-generated | — | — |
| ModelManager.java | Add metadata validation | — | — | — |
| ObservationBuilder.java | — | — | Use ObservationSlots | — |
| ActionExecutor.java | — | — | — | Use config for masking |
| BotBrain.java | — | — | — | Use config USED_BITS |
| dimensions.json | — | Create (full) | Update (slots) | Update (masking) |
| scripts/*.py | — | Create 2 generators | — | — |
| build.gradle | — | Add generation step | — | — |

**Total effort: 8-10 hours to implement all 4 priorities**
**Effort per feature after: 1 hour (change JSON, let build system sync)**
