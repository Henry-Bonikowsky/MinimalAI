# Coupling Hotspots: Exact File Locations & Line Numbers

This doc maps out exactly where the coupling lives, so you can see the scope of changes.

---

## ACTION SPACE (35 actions) — 8 Files

### 1. training/config.py (lines 1-70)
- ACT_FORWARD=0, ACT_BACKWARD=1, ... ACT_TARGET_4=34
- NUM_ACTIONS=35
- USED_BITS=[0, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 26]

**Change:** If adding new sigil action:
- Add ACT_SIGIL_12=? to config.py
- Update NUM_ACTIONS to 36
- Add to USED_BITS if learnable

---

### 2. src/main/java/com/minimalai/ai/ActionSpace.java (lines 1-72)
- All 35 action constants as public static final int
- NUM_ACTIONS = 35
- NUM_SIGIL_SLOTS = 12
- NUM_TARGET_SLOTS = 5

**Change:** Add ACT_SIGIL_12, update NUM_ACTIONS

---

### 3. src/main/java/com/minimalai/bot/BotBrain.java (lines 96-118)
- RANDOM_PROBS array with exactly 35 floats (for physics recording mode)
- Used when recordingMode=true, randomizing actions

**Change:** Add entry to RANDOM_PROBS for new action

---

### 4. src/main/java/com/minimalai/ai/ModelManager.java (lines 37-38)
- private static final int NUM_ACTIONS = 35;
- Hardcoded, not imported from ActionSpace

**Change:** Either import or update constant

---

### 5. src/main/java/com/minimalai/ai/ActionExecutor.java (lines ~200-250)
- buildActionMask() creates float[NUM_ACTIONS]
- Iterates over ActionSpace.NUM_ACTIONS

**Change:** No change needed if using NUM_ACTIONS constant

---

### 6. src/main/java/com/minimalai/training/ExperienceBuffer.java (lines ~50-70)
- private static final int NUM_ACTIONS = ActionSpace.NUM_ACTIONS; // 35
- Used for serialization: writeInts(dos, exp.actions, NUM_ACTIONS);

**Change:** No change needed if using constant

---

### 7. training/vec_sim.py (lines 30-48)
- Imports NUM_ACTIONS from config
- Used throughout for action indexing, masking

**Change:** Imports auto-update from config.py

---

### 8. training/fast_train.py (lines 27-35)
- Imports NUM_ACTIONS, USED_BITS from config
- Used in action building/masking logic

**Change:** Imports auto-update from config.py

---

## OBSERVATION SPACE (320 dims) — 6 Files

### 1. src/main/java/com/minimalai/ai/ObservationSpace.java (lines 1-18)
```java
public static final int SELF_STATE_DIM = 38;
public static final int ENTITY_FEATURE_DIM = 24;
public static final int MAX_ENTITIES = 8;
public static final int COMBAT_CTX_DIM = 26;
public static final int SIGIL_STATE_DIM = 48;
public static final int ENV_STATE_DIM = 8;
```

**Change:** If adding team awareness:
- SELF_STATE_DIM = 38 → 40 (add team_id, team_size)
- Need to recalculate OBS_DIM

---

### 2. src/main/java/com/minimalai/ai/ObservationBuilder.java
Multiple locations with hardcoded dimensions:

- Line 165: `float[] entityFeatures = new float[MAX_ENTITIES * ENTITY_FEATURE_DIM];`
- Line 179: `float[] s = new float[SELF_STATE_DIM];`
- Lines 180-250: Hardcoded slot indices
  - `s[0] = p.getHealth() / 20f;` (health)
  - `s[24-32]` status effects
  - `s[34-37]` sigil state
- Line 264: `int base = slot * ENTITY_FEATURE_DIM;`
- Line 348: `float[] ctx = new float[COMBAT_CTX_DIM];`
- Line 409: `float[] state = new float[SIGIL_STATE_DIM];`
- Line 441: `float[] env = new float[ENV_STATE_DIM];`

**Change:** If adding team observations:
- Update buildSelfState() to add team fields at new slots
- Update buildEntityFeatures() if entity struct changes
- Update all slot indices everywhere

---

### 3. src/main/java/com/minimalai/ai/ModelManager.java (lines 132-140)
```java
NDList input = new NDList(
    TensorUtil.toNDArray(tickManager, selfState, 1, 38),
    TensorUtil.toNDArray(tickManager, entityFeatures, 1, 8, 24),
    TensorUtil.toNDArray(tickManager, entityMask, 1, 8),
    TensorUtil.toNDArray(tickManager, combatCtx, 1, 26),
    TensorUtil.toNDArray(tickManager, sigilState, 1, 48),
    TensorUtil.toNDArray(tickManager, envState, 1, 8),
    TensorUtil.toNDArray(tickManager, hidden, 1, 1, GRU_HIDDEN_DIM)
);
```

**Change:** Update tensor shapes if observation structure changes

---

### 4. training/models/network.py (lines 90-96)
```python
combined_dim = (
    SELF_STATE_DIM +           # 38
    self.entity_attention.output_dim +  # 64
    COMBAT_CTX_DIM +           # 26
    SIGIL_STATE_DIM +          # 48
    ENV_STATE_DIM              # 8
)  # = 184
```

**Change:** If SELF_STATE_DIM increases:
- combined_dim needs recalculation
- May need encoder architecture adjustment

---

### 5. training/vec_sim.py (lines 30-48, 1673-1768)
- Imports all dimension constants
- _get_obs() builds observation arrays with exact slot indices
- Lines 1673-1720 hardcode observation building

**Change:** If observation structure changes:
- Must update _get_obs() to match new slots
- Must align with Java ObservationBuilder.java exactly

---

### 6. training/config.py (lines 78-110)
- SELF_STATE_DIM=38 with detailed slot map comment
- ENTITY_FEATURE_DIM=24 with detailed slot map
- COMBAT_CTX_DIM=26 with detailed slot map
- OBS_DIM=320 comment (implicit validation)

**Change:** Update slot maps if adding observations

---

## HIDDEN STATE SIZE (128) — 3 Files

### 1. src/main/java/com/minimalai/ai/ObservationSpace.java (line 15)
```java
public static final int HIDDEN_DIM = 128;
```

### 2. src/main/java/com/minimalai/ai/ModelManager.java (line 38)
```java
private static final int GRU_HIDDEN_DIM = 128;
```
Also used on line 139 in infer() method.

### 3. src/main/java/com/minimalai/bot/BotBrain.java (line 57)
```java
private float[] hidden = new float[ObservationSpace.HIDDEN_DIM];
```

### 4. training/config.py (line 119)
```python
GRU_HIDDEN_DIM = 128
```

### 5. training/models/network.py (line 119)
```python
self.gru = nn.GRU(
    input_size=HIDDEN_DIM,
    hidden_size=GRU_HIDDEN_DIM,  # 128
    num_layers=1,
    batch_first=True,
)
```

**Change:** If retraining with different GRU size:
- Must update all 5 locations
- Java will crash if model output doesn't match buffer size

---

## SIGIL ACTION MAPPING — 1 File (But Critical)

### src/main/java/com/minimalai/bot/BotBrain.java (lines 129-134)
```java
private static final int[] ACTIVE_ABILITY_SLOTS = {1, 2, 3, 5};
// Net output [0]=brace, [1]=cleo, [2]=sand, [3]=grace
```

**Change:** If sigils reordered in config.yml:
- Must manually update ACTIVE_ABILITY_SLOTS
- No validation that it matches training setup

---

## CONSUMABLE COUNTS — 3 Files

### 1. training/config.py (lines 125-142)
```python
GAP_START_COUNT = 64
GAP_COOLDOWN_TICKS = 180
GAP_ABSORPTION = 4.0
GAP_REGEN_TICKS = 100
GAP_REGEN_RATE = 25

POT_HEAL_AMOUNT = 8.0
POT_SPLASH_RADIUS = 4.0
POT_THROW_LOCKOUT = 3
POT_START_COUNT = 31

PEARL_COOLDOWN_TICKS = 200
PEARL_SELF_DAMAGE = 5.0
PEARL_START_COUNT = 16
```

### 2. src/main/java/com/minimalai/ai/ObservationBuilder.java (lines 220-226)
```java
s[20] = countItem(p, Items.GOLDEN_APPLE) / 64f;  // HARDCODED 64
s[21] = countItem(p, Items.SPLASH_POTION) / 64f; // HARDCODED 64
s[22] = countItem(p, Items.ENDER_PEARL) / 16f;   // HARDCODED 16
```

Also line 358-362 in buildCombatCtx():
```java
ctx[3] = countItem(bot, Items.GOLDEN_APPLE) / 64f;
ctx[4] = countItem(bot, Items.SPLASH_POTION) / 64f;
ctx[5] = countItem(bot, Items.ENDER_PEARL) / 16f;
```

### 3. training/vec_sim.py (lines 130-145)
```python
self.gapple_cooldown = np.zeros(self.n)
self.gapple_count = np.full(self.n, GAP_START_COUNT)
# ... reimplements all consumable mechanics
```

**Change:** If server config changes consumable counts:
- Must update all 3 locations
- Otherwise feature scaling breaks

---

## USED_BITS (Action Masking) — 2 Files

### 1. training/config.py (lines 68-75)
```python
USED_BITS = [0, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 26]
# FWD, LEFT, RIGHT, JUMP, EAT, POT, PEARL, SIGIL0-3, ENGAGE
```

### 2. src/main/java/com/minimalai/bot/BotBrain.java (lines ~250-260)
```java
boolean[] usedBits = new boolean[ActionSpace.NUM_ACTIONS];
for (int i = 0; i < ActionSpace.NUM_ACTIONS; i++) {
    usedBits[i] = false;
}
// Manually set usedBits[0] = true, usedBits[2] = true, etc.
```

**Change:** If new action should be learnable:
- Add to USED_BITS in config.py
- Add to usedBits construction in BotBrain.java
- Easy to get out of sync

---

## MODEL VERSIONING — 1 File (Currently Missing)

### Training export code (location: training/fast_train.py, around line 900)
Currently does:
```python
torch.save(traced_model, checkpoint_path)
```

Should do:
```python
torch.save({
    "model": traced_model,
    "metadata": {
        "action_space": NUM_ACTIONS,
        "self_state_dim": SELF_STATE_DIM,
        # ... all dimensions
    }
}, checkpoint_path)
```

### Model loading code (location: src/main/java/com/minimalai/ai/ModelManager.java, line 57)
Currently does:
```java
// Load model, assume shapes match
Criteria<NDList, NDList> criteria = ...
ZooModel<NDList, NDList> model = criteria.loadModel();
```

Should do:
```java
// Load model AND metadata, validate
Object loaded = torch.load(...);
ModelMetadata meta = extractMetadata(loaded);
meta.validateOrThrow();  // Fail fast if incompatible
```

---

## SUMMARY TABLE

| Component | Files | Locations | Severity | Effort to Fix |
|-----------|-------|-----------|----------|--------------|
| Action space | 8 | 10+ locations | CRITICAL | Medium (need generator) |
| Observation dims | 6 | 20+ locations | CRITICAL | Medium (need generator) |
| Hidden state size | 5 | 5 locations | HIGH | Easy (find & replace) |
| Model versioning | 2 | 2 locations | HIGH | Easy (add metadata) |
| Sigil slot mapping | 1 | 1 location | MEDIUM-HIGH | Easy (but manual sync) |
| Consumable counts | 3 | 6 locations | MEDIUM | Easy (extract to config) |
| USED_BITS masking | 2 | 2 locations | MEDIUM | Easy (data-driven) |

**Total: ~50-60 locations across 10-15 files need coordination for any major change**

This is why adding a single feature requires updating 8+ files and retraining everything.
