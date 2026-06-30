# MinimalAI: Extensibility & Coupling Audit

**Bottom Line:** Your codebase is tightly coupled around action/observation dimensions. Every time you want to add a feature (new sigil, teammate HP, new item), you're manually updating 8-12 files and retraining everything. This scales horribly to Legion mode (20 coordinated bots). The root cause is treating 35 actions and 320 observations as "final" when they're obviously temporary.

---

## CRITICAL COUPLING POINTS

### 1. ACTION SPACE (35 actions) — Hardcoded in 8 Files

**Files:**
- `training/config.py`: ACT_FORWARD=0, ..., NUM_ACTIONS=35
- `src/main/java/com/minimalai/ai/ActionSpace.java`: All 35 constants
- `src/main/java/com/minimalai/bot/BotBrain.java`: RANDOM_PROBS array (35 entries)
- `src/main/java/com/minimalai/ai/ModelManager.java`: NUM_ACTIONS=35 hardcoded
- `src/main/java/com/minimalai/ai/ActionExecutor.java`: float[NUM_ACTIONS] masks
- `src/main/java/com/minimalai/training/ExperienceBuffer.java`: Serialization
- `training/vec_sim.py`: Action indexing throughout
- `training/fast_train.py`: USED_BITS, action building

**What breaks when adding 1 action:**
1. Add constant to config.py
2. Add constant to ActionSpace.java
3. Add entry to BotBrain.RANDOM_PROBS (for physics recording mode)
4. Update network.py policy head (34 → 35 outputs)
5. Update fast_train.py action masking
6. **Retrain all models from scratch**
7. **All 4000+ checkpoints become useless**

**Severity: CRITICAL** — Cascades to model retraining.

---

### 2. OBSERVATION SPACE (320 dims) — Hardcoded in 6 Places

**The structure:**
```
SELF_STATE_DIM = 38              (health, armor, velocity, effects, etc.)
ENTITY_FEATURE_DIM = 24          (per nearby entity, 8 max)
COMBAT_CTX_DIM = 26              (consumables, damage, combo, cooldowns)
SIGIL_STATE_DIM = 48             (12 sigils × 4 values)
ENV_STATE_DIM = 8                (arena, episode progress, alive enemies)
OBS_DIM = 320                    (flat total)
```

**Hardcoded locations:**
- `ObservationSpace.java`: All as public static final constants
- `ObservationBuilder.java`: Arrays allocated with exact sizes, slot indices hardcoded (e.g., s[0]=health/20, s[24-32]=status effects, s[34-37]=sigil combat state)
- `ModelManager.infer()`: Tensor shapes baked in: `TensorUtil.toNDArray(tickManager, selfState, 1, 38)`, `(1, 8, 24)`, `(1, 26)`, `(1, 48)`, `(1, 8)`
- `network.py`: `combined_dim = 38 + 64 + 26 + 48 + 8 = 184` (before entity attention)
- `vec_sim.py`: Observation slots hardcoded with exact indices
- `fast_train.py`: Implicitly validates OBS_DIM=320

**What breaks when adding teammate HP (Legion mode):**
1. Decide: increase SELF_STATE_DIM (38→40) OR ENTITY_FEATURE_DIM (24→26)?
2. Update ObservationSpace.java constants
3. Rewrite ObservationBuilder.buildSelfState() + buildEntityFeatures()
4. Update ModelManager.infer() tensor shapes
5. Update network.py combined_dim calculation
6. Update vec_sim.py observation building
7. Update all slot indices in vec_sim (everything shifts)
8. **Retrain all models**
9. **All checkpoints invalid**

**Severity: CRITICAL** — Cascades to complete model retraining.

---

### 3. Model Interface: Zero Versioning, Zero Schema Validation

**Current load/infer process:**

```java
// ModelManager.java lines 132-150
NDList input = new NDList(
    TensorUtil.toNDArray(tickManager, selfState, 1, 38),           // hardcoded 38
    TensorUtil.toNDArray(tickManager, entityFeatures, 1, 8, 24),   // hardcoded 8, 24
    TensorUtil.toNDArray(tickManager, entityMask, 1, 8),           // hardcoded 8
    TensorUtil.toNDArray(tickManager, combatCtx, 1, 26),           // hardcoded 26
    TensorUtil.toNDArray(tickManager, sigilState, 1, 48),          // hardcoded 48
    TensorUtil.toNDArray(tickManager, envState, 1, 8),             // hardcoded 8
    TensorUtil.toNDArray(tickManager, hidden, 1, 1, 128)           // hardcoded 128
);
NDList output = predictor.predict(input);
```

**Problems:**
- No runtime validation of shapes
- No version string in .pt file
- If someone exports a model with 40 features instead of 38 → **shape mismatch at inference, silent crash mid-fight**
- If someone retrains with 256-dim hidden → Java tries to store 256 values in 128-dim buffer → **data corruption or truncation**
- No way to detect incompatibility before deployment

**Severity: HIGH** — Causes runtime crashes in production.

---

### 4. Hidden State Size: Magic Number 128

**Locations:**
- `ObservationSpace.HIDDEN_DIM = 128`
- `ModelManager.java` line 139: `GRU_HIDDEN_DIM = 128` (hardcoded in infer())
- `BotBrain.java` line 57: `private float[] hidden = new float[ObservationSpace.HIDDEN_DIM]`
- `network.py`: `GRU_HIDDEN_DIM = 128`
- `config.py`: `GRU_HIDDEN_DIM = 128`

**What breaks:**
- If someone trains with GRU_HIDDEN_DIM=256, Java crashes on first inference
- No validation at load time → discovered mid-fight

**Severity: MEDIUM-HIGH** — Easy mistake, catastrophic consequence.

---

### 5. Sigil Action Mapping: Hardcoded Slot Indices

**BotBrain.java lines 129-134:**
```java
// Active ability slots: neural net output [0-3] → ArcaneSigils bind slot
// Based on config.yml registration order:
//   0=ancient_crown(passive), 1=kings_brace, 2=cleopatra, 3=quick_sand,
//   4=divine_intervention(passive), 5=niles_grace
// Net output: [0]=brace, [1]=cleopatra, [2]=quicksand, [3]=grace
private static final int[] ACTIVE_ABILITY_SLOTS = {1, 2, 3, 5};
```

**What breaks:**
- If sigils are reordered in config.yml, ACTIVE_ABILITY_SLOTS is now wrong
- Model trained to fire slot [0] for "Brace", but Java fires slot [5] instead
- No validation that indices match what the model expects
- Silent wrong ability fires during combat

**Severity: MEDIUM-HIGH** — Hard to debug, wrong behavior in production.

---

### 6. USED_BITS Hardcoding: Action Masking

**Locations:**
- `config.py`: `USED_BITS = [0, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 26]` (12 of 35)
- `BotBrain.java`: Reconstructs usedBits boolean[] from ActionSpace constants

**Problem:**
- Network outputs all 35 logits, but only 12 are actionable
- Rest masked post-hoc in ActionExecutor.buildActionMask()
- Wasted network capacity on unused outputs
- If new action should be learned, must update USED_BITS in two places with manual coordinate

**Severity: MEDIUM** — Annoying, not breaking, but wasteful.

---

### 7. Consumable Counts Scattered Across 3 Files

**Locations:**
- `config.py`: GAP_START_COUNT=64, POT_START_COUNT=31, PEARL_START_COUNT=16, GAP_COOLDOWN_TICKS=180, PEARL_COOLDOWN_TICKS=200
- `ObservationBuilder.java`: Normalization hardcoded (countItem / 64, / 16)
- `vec_sim.py`: Reimplements consumable mechanics

**What breaks:**
- Server admin changes loadout: 64 gapples → 32 gapples
- Python sim still normalizes by 64 (wrong feature scaling)
- Model trained on distribution that doesn't match reality
- Bot doesn't learn when to heal properly

**Severity: MEDIUM** — Causes training/deployment mismatch.

---

## ROOT CAUSE ANALYSIS

### Why did this happen?

1. **Constants chosen early, never abstracted.** 35 actions felt "final" when chosen → hardcoded as `static final` everywhere. Same with 38-dim self state, 320-dim total obs.

2. **No single source of truth.** config.py and ActionSpace.java independently define action indices. No auto-generated code from shared schema.

3. **Observation building done twice.** Java (ObservationBuilder.java) builds from live ServerPlayer. Python (vec_sim.py) rebuilds from scratch for simulation. No shared schema → easy to drift.

4. **No model versioning.** .pt files have no metadata about expected input/output shapes. TorchScript export doesn't embed config constants. Java blindly assumes shapes match.

5. **No validation at load time.** ModelManager.loadModel() doesn't check shapes. First inference that fails reveals incompatibility (mid-fight is bad).

### Why it matters now

- **v1 (current):** Rarely change dimensions → OK
- **v2 (Legion):** Need 20+ entity slots, team awareness, new actions → breaks build
- **v3+ (coordination):** New observation types per gameplay mode → cascades everywhere

---

## SEVERITY SUMMARY

| Severity | Issue | Breaks When | Impact |
|----------|-------|-------------|--------|
| **CRITICAL** | Action space hardcoding | Adding any new action | Model retraining, checkpoint loss |
| **CRITICAL** | Observation dims hardcoding | Adding any new observation | Model retraining, checkpoint loss |
| **HIGH** | Zero model versioning | Deploying incompatible model | Runtime crash mid-fight |
| **MEDIUM-HIGH** | Hidden state size magic | Retraining with different GRU | Java buffer overflow/truncation |
| **MEDIUM-HIGH** | Sigil slot hardcoding | Reordering sigils in config | Silent wrong ability fire |
| **MEDIUM** | USED_BITS in 2 places | Adding learnable action | Manual sync, easy to miss |
| **MEDIUM** | Consumable counts scattered | Server config change | Model feature scaling mismatch |

---

## MINIMUM VIABLE FIXES (Priority Order)

### Priority 1: Add Model Metadata (20 min, HIGH impact)

**Problem:** No way to detect incompatible models at load time.

**Solution:** Embed observation/action spec in exported models.

**Python side (fast_train.py):**
```python
model_meta = {
    "action_space": NUM_ACTIONS,
    "self_state_dim": SELF_STATE_DIM,
    "entity_feature_dim": ENTITY_FEATURE_DIM,
    "max_entities": MAX_ENTITIES,
    "combat_ctx_dim": COMBAT_CTX_DIM,
    "sigil_state_dim": SIGIL_STATE_DIM,
    "env_state_dim": ENV_STATE_DIM,
    "hidden_dim": GRU_HIDDEN_DIM,
}
torch.save({"model": traced_model, "meta": model_meta}, checkpoint_path)
```

**Java side (ModelManager.java):**
```java
// At load time
Map<String, Object> loaded = torch.load(modelPath);
Map<String, Integer> meta = (Map) loaded.get("meta");

if (meta.get("action_space") != 35) {
    throw new IllegalStateException("Model expects " +
        meta.get("action_space") + " actions, Java configured for 35");
}
// ... validate all dimensions
```

**Cost:** ~30 lines of code
**Benefit:** Immediate detection of incompatible models, blocks crashes at load
**ROI:** 10:1 (prevents hours of debugging)

---

### Priority 2: Config-Driven Dimensions (2-3 hours, CRITICAL)

**Problem:** Dimension constants hardcoded in 8+ places. Change one, update them all.

**Solution:** Single JSON source of truth, auto-generate both Java and Python.

**dimensions.json:**
```json
{
  "actions": {
    "total": 35,
    "movement": {"start": 0, "count": 7},
    "combat": {"start": 7, "count": 7},
    "sigils": {"start": 14, "count": 12},
    "camera": {"start": 26, "count": 4},
    "targeting": {"start": 30, "count": 5},
    "used_bits": [0, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 26]
  },
  "observations": {
    "self_state": 38,
    "entity_feature": 24,
    "max_entities": 8,
    "combat_ctx": 26,
    "sigil_state": 48,
    "env_state": 8,
    "hidden_dim": 128
  },
  "consumables": {
    "gapple": {"count": 64, "cooldown_ticks": 180, "eat_ticks": 36},
    "pot": {"count": 31, "lockout_ticks": 3},
    "pearl": {"count": 16, "cooldown_ticks": 200}
  }
}
```

**Python generator (scripts/gen_python_config.py):**
```python
def generate_python_config(json_path):
    with open(json_path) as f:
        dims = json.load(f)

    output = "# AUTO-GENERATED — Edit dimensions.json, not this file\n\n"
    output += f"NUM_ACTIONS = {dims['actions']['total']}\n"
    output += f"SELF_STATE_DIM = {dims['observations']['self_state']}\n"
    # ... etc
    output += f"USED_BITS = {dims['actions']['used_bits']}\n"
    return output
```

**Java generator (scripts/gen_java_constants.java):**
```java
// Generates ActionSpace.java and ObservationSpace.java
```

**Changes needed:**
- config.py: import from generated_dimensions.py (or read JSON at import time)
- ActionSpace.java: auto-generated from JSON
- ObservationSpace.java: auto-generated from JSON
- Build pipeline: Run generators before compile

**Cost:** ~200 lines (2 generator scripts)
**Benefit:** Single source of truth, any dimension change auto-updates both sides, prevents sync errors
**ROI:** 5:1 (enables adding features without manual coordination)

---

### Priority 3: Observation Schema Abstraction (4-5 hours, HIGH)

**Problem:** Observations built separately in Java and Python. Easy to drift. Slot indices hardcoded.

**Solution:** Define schema once, build/validate at both ends.

**Python side (training/observations.py):**
```python
@dataclass
class ObsSlots:
    # self_state indices
    HEALTH = 0
    MAXHEALTH = 1
    ABSORPTION = 2
    ARMOR = 3
    ARMOR_TOUGHNESS = 4
    VEL_X = 5
    VEL_Y = 6
    VEL_Z = 7
    YAW = 8
    # ... etc up to 37

    # combat_ctx indices
    GAPPLES = 3
    POTS = 4
    PEARLS = 5
    DAMAGE_DEALT = 7
    COMBO_COUNTER = 9
    # ... etc

class ObservationBuilder:
    def build_self_state(self, player_state):
        s = np.zeros(SELF_STATE_DIM)
        s[ObsSlots.HEALTH] = player_state.health / 20.0
        s[ObsSlots.MAXHEALTH] = player_state.maxhealth / 20.0
        # ... no magic numbers
        return s

    def validate(self, obs):
        assert obs['self_state'].shape == (SELF_STATE_DIM,)
        assert obs['entity_features'].shape == (MAX_ENTITIES, ENTITY_FEATURE_DIM)
        # ... etc
```

**Java side (ObservationSlots.java):**
```java
public final class ObservationSlots {
    // Auto-generated from Python schema
    public static final int HEALTH = 0;
    public static final int MAXHEALTH = 1;
    // ... etc
}

public class ObservationBuilder {
    private float[] buildSelfState(ServerPlayer p) {
        float[] s = new float[SELF_STATE_DIM];
        s[ObservationSlots.HEALTH] = p.getHealth() / 20.0f;
        // ... no magic numbers, matches Python exactly
        return s;
    }

    public void validate(Observation obs) throws ValidationException {
        assert obs.selfState.length == SELF_STATE_DIM;
        // ... etc
    }
}
```

**Cost:** ~300 lines (schema + builders + validation)
**Benefit:** Single definition of observation structure, automatic consistency checking, easy to extend
**ROI:** 4:1 (enables adding features without hard-to-find bugs)

---

### Priority 4: Action Space as Data (2-3 hours, MEDIUM)

**Problem:** Action groups hardcoded. New action requires changes in multiple places.

**Solution:** Define actions in YAML, auto-generate handlers.

**actions.yaml:**
```yaml
movement:
  - {name: forward, index: 0, forced: false, masked: false}
  - {name: backward, index: 1, forced: false, masked: false}
  - {name: left, index: 2, forced: false, masked: false}
  - {name: right, index: 3, forced: false, masked: false}
  - {name: jump, index: 4, forced: false, masked: false}
  - {name: sneak, index: 5, forced: false, masked: false}
  - {name: sprint, index: 6, forced: true, masked: false}  # always on

combat:
  - {name: attack, index: 7, forced: false, masked: false, auto_derived_from: engage}
  - {name: block, index: 8, forced: false, masked: true}
  # ... etc

sigils:
  - {count: 12, activatable_indices: [1, 2, 3, 5]}

targeting:
  - {count: 5, max_nearby: 5}
```

**Java generator (scripts/gen_action_handlers.py):**
```python
# Generates ActionHandler.java with movement(), combat(), sigils() methods
```

**Cost:** ~150 lines (config + generator)
**Benefit:** Actions become data-driven, Legion mode can add TEAM_SUPPORT without code changes
**ROI:** 6:1 (Legion needs this; makes it painless)

---

## WHAT TO IMPLEMENT FIRST

**If you only fix ONE thing:** **Priority 2 (dimensions.json)**
- Single change in one file
- Auto-updates Java + Python at build time
- Immediate payoff for any new feature
- Foundation for all other fixes

**Quick wins (do these first):**
1. Priority 1 (metadata) — 20 minutes, prevents crashes
2. Priority 2 (JSON config) — 2-3 hours, enables extensibility
3. Add ObservationSlots class (25 lines) — replaces magic indices with named constants

**Then tackle:**
4. Priority 3 (schema abstraction) — enables team observations for Legion
5. Priority 4 (action data) — enables new action types

---

## LEGION MODE: How Bad Is It?

**Current architecture breaks for 20-bot coordination because:**

1. **Observation space** assumes single opponent → ENTITY_FEATURE_DIM=24 doesn't fit team roles
2. **MAX_ENTITIES=8** → can't hold 19 teammates + enemies
3. **Action space** has no team-aware actions (SUPPORT_HEAL, FOLLOW_ALLY, etc.)
4. **Reward system** is 1v1 only (kill=+20, death=-20) → no team objectives
5. **Network architecture** has no inter-bot attention mechanism

**With Priority 2+3 fixes:**
- Dimensions become data-driven → add TEAM_STATE_DIM easily
- ObservationBuilder auto-validates new tensors
- Can extend action space without cascading changes
- Single-opponent code path still works (backward compatible)

**Still requires (unavoidable):**
- New network architecture (team attention heads)
- New reward logic (shared objectives)
- New training environment (team vs team)

**But:** You can do this WITHOUT invalidating all current code and checkpoints. Just add a new config profile for Legion mode.

---

## FINAL VERDICT

Your code is well-written for a **single-opponent 1v1 system**. But it's brittle for iteration because:

- ✅ Good: Rewards are config-driven
- ✅ Good: ActionExecutor is clean and extensible
- ✅ Good: Observation building is centralized in one place per language

- ❌ Bad: Dimension constants hardcoded as static finals
- ❌ Bad: No single source of truth (Java/Python duplicate definitions)
- ❌ Bad: No model versioning or schema validation
- ❌ Bad: Magic indices throughout observation building

**The fix is straightforward:** Move dimensions to config (JSON), generate code from it, and add model metadata. Then adding new features goes from "update 8 files + retrain everything" to "change JSON + let build system sync".

For Legion mode, you'll need architectural changes anyway (team attention, shared rewards). But with this fix, you can do it without invalidating existing work.

**Estimated effort to full extensibility: 8-10 hours of coding + testing.**
**Payoff: First new feature takes 1 hour instead of 8. Every subsequent feature is free (just edit JSON).**
