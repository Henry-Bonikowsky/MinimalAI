# MinimalAI — Project Context for Brainstorming

## Vision

MinimalAI is a Paper plugin (MC 1.21.10, Java 21) that creates AI-controlled combat bots on a custom Minecraft PvP server. The long-term goal is a **Legion system**: 20 AI bots fighting as a coordinated hive-mind unit against a team of real players.

The bots should operate like a military unit — with roles (healer, frontline, assassin, anchor), coordinated plays (pearl-onto-healer for buffs, focus-fire, bait-and-switch), and real-time tactical adaptation. No human needs to command them.

---

## Server Environment

This is NOT vanilla Minecraft. The server runs a **Paper-Kitara fork** with custom combat:

- **1.8-style combat**: No attack cooldown for swords, spam-click PvP
- **Advanced knockback**: Sprint-hits send targets flying, KB is physics-based (not vanilla formula)
- **W-tap combos**: Releasing forward for 1 tick resets sprint, allowing consecutive sprint-hits that chain into combos. Good players land 3-8 hit combos.
- **Sword blocking**: Right-click block reduces damage by `(1+dmg)*0.5` and KB by 85% horizontal, 95% vertical
- **Standard loadout**: Sharpness 6 diamond sword, Protection 5 full diamond armor, 64 golden apples (9s cooldown), 31 splash health pots, 16 ender pearls (10s cooldown)

### Arcane Sigils Plugin

On top of Kitara combat, the **Arcane Sigils** plugin adds special abilities tied to gear:

| Sigil | Slot | Type | Effect |
|-------|------|------|--------|
| Ancient Crown | Helmet | Passive | Immunity to negative effects |
| King's Brace | Chestplate | Active (bind 1) | 80% damage reduction burst, requires 30 charge hits |
| Cleopatra | Leggings | Active (bind 2) | Strips target's buffs + 20% damage amp, 9s CD |
| Quick Sand | Boots | Active (bind 3) | AoE slow + pull on hit, 7s CD |
| Divine Intervention | Sword | Passive | Invuln hits while blocking |
| Nile's Grace | Axe | Active (bind 5) | Regen + 30% DR, 9s CD |

Plus 8 passive sigils: Iron Forged (fire res), Lifeforce (passive regen), Extra Padding (+max HP), Iron Fist (strength II), Rocket Boots (speed), Well Fed (on-hit food), Spring Shoes (jump boost), Meal Planning (auto-feed).

---

## Current Architecture

### Bot Brain (Java — server-side)

Each bot runs a **BotBrain** that ticks once per server tick (50ms). Three modes:

- **RULE**: `RuleCombatEngine` — a hand-coded state machine with 5 phases:
  - ENGAGE: Sprint toward target, attack in reach, occasional crit jumps
  - COMBO: W-tap after landing hits (release forward 1 tick → re-sprint → sprint-hit)
  - RETREAT: Face away, sprint, eat golden apples, emergency pot if critical, sword block if chased
  - PEARL_ESCAPE: Pearl away when very low HP
  - PEARL_AGGRO: Pearl toward distant target when healthy
  - Also handles sigil ability timing with rule-based logic (HP thresholds, cooldown tracking)
  - 5 difficulty tiers that interpolate: reaction delay (8→0 ticks), aim jitter (15→0 degrees), attack reach (2.0→3.0 blocks), heal threshold (8→14 HP), W-tap consistency (50%→100%), strafe frequency (10%→50%)

- **HYBRID**: Rule combat + neural net for sigil timing only. A small `SigilNet` takes 16-float observations (HP, absorption, cooldowns, active buffs, episode progress) and outputs 4 probabilities for activating each sigil ability.

- **NEURAL**: Full neural network inference. A `CombatNetwork` with entity attention + GRU processes 320-dim observations and outputs 35-dim multi-binary actions. Currently underperforming the rule system.

### Action Space (35 multi-binary actions)

```
Movement (0-6):   FORWARD, BACKWARD, STRAFE_LEFT, STRAFE_RIGHT, JUMP, SNEAK, SPRINT
Combat (7-13):    ATTACK, BLOCK, EAT_GAP, THROW_POT, THROW_PEARL, SPRINT_RESET, SWAP_WEAPON
Sigils (14-25):   SIGIL_0 through SIGIL_11 (4 active: Brace, Cleo, Sand, Grace)
Camera (26-29):   ENGAGE (face+auto-attack), FACE_AWAY, LOOK_DOWN, FACE_MOVEMENT
Targets (30-34):  TARGET_0 through TARGET_4
```

The neural net only controls 12 bits: `[FWD, LEFT, RIGHT, JUMP, EAT, POT, PEARL, SIGIL0-3, ENGAGE]`. Sprint is forced on, attack is auto-derived from ENGAGE + reach.

### Action Executor (Java)

Converts the 35-element action vector into NMS operations: setting movement fields (`zza`, `xxa`), looking at/away from targets, spawning projectiles (pearls, splash pots), triggering golden apple eating, calling ArcaneSigils API via reflection bridge. Handles direct HP manipulation for damage (vanilla pipeline broken for fake players), i-frames, kill detection.

### Training Pipeline (Python)

- **`vec_sim.py`**: Numpy-vectorized 1v1 simulator running N parallel fights. All Kitara physics faithfully reproduced (KB, sprint mechanics, friction, gravity). Supports 4 sigil abilities. ~460K steps/second raw, ~2.4K sps during training. Configurable combo styles (W-tap, S-tap, none).

- **`fast_train.py`**: PPO trainer. Can train against rule-based opponents (`--rule-opponent`, default) or self-play. Uses RTX 5080 GPU.

- **`CombatNetwork`** (network.py): Entity attention (Transformer encoder, 2 heads, 2 layers) over up to 8 nearby entities → mean pool → concat with self-state/combat/sigil/env → Linear → GRU(128) → policy head (35 actions) + value head.

- **`SigilNet`** (sigil_network.py): Small standalone net for sigil timing only. Deprecated in favor of integrated CombatNetwork.

### Reward System

Extensive reward shaping with multiple playstyle presets (Charger, Counter, Hybrid, Aggressive, Defensive). Key rewards:
- Kill/death: +20/-20
- Combo escalation: +3 base + 2 per streak hit
- Clean sprint-hits: +2, trades: -1.5
- Distance shaping: +0.3 per block closed, -0.5 per block retreated
- Consumable usage rewards/penalties (smart eat +3, wasteful eat -3, emergency pot +4, healing enemy -8)
- Counter-hit bonus: +8 (for counter style)

---

## What Works

1. **Rule-based combat** at difficulty 5 is competitive — W-tap combos, smart retreating, sigil timing, pearl usage
2. **Vectorized sim** faithfully reproduces server physics at 460K sps
3. **Bot spawning/lifecycle** is solid — fake players, kit application, sigil registration via reflection bridge
4. **Direct damage system** bypasses broken vanilla pipeline for fake players
5. **Difficulty scaling** from braindead (tier 1) to sweaty (tier 5) via parameter interpolation

## What Doesn't Work

1. **Neural models underperform rule-based** — all self-play training converges to mutual trading (both bots swing same tick, max combo 2-3). The only model that learned combos (v10) was trained against a dumb rule-based charger and doesn't generalize.
2. **No multi-agent anything** — bots are completely independent. No shared state, no team coordination, no role system.
3. **1v1 only** — sim, training, and brain are all built for 1v1. No concept of teammates, focus targets, or team strategy.
4. **Sim-to-server gap** — models trained in Python sim behave differently on the actual server (timing, physics edge cases).

---

## The Gap: Current State → Legion

### Individual Combat (Foundation)
- Rule engine works but is predictable. Human players will learn its patterns.
- Neural models need to either surpass rule-based or augment it (hybrid approach).
- Need bots that can adapt to different opponent playstyles, not just execute fixed patterns.

### Team Awareness (Next Step)
- Bots need to perceive teammates: who's nearby, their HP, their targets, their combat phase
- Target selection needs to be team-aware: focus the enemy the team is already fighting, don't spread damage
- Friendly fire avoidance for splash pots/pearls

### Coordination Layer (The Hard Part)
- **Shared state bus**: All 20 bots broadcast their state (HP, position, target, phase, cooldowns)
- **Role assignment**: Bots dynamically assigned roles based on game state (not hardcoded)
- **Team plays**: Coordinated maneuvers triggered by game state:
  - Pearl-onto-healer: Healer pops AoE buff → all bots pearl to healer's position → push with buff advantage
  - Focus fire: Coordinator marks a target → all bots in range switch to that target
  - Collapse: All bots pearl onto a single enemy for instant kill
  - Peel: When healer is being attacked, nearby bots switch to attacking the aggressor
  - Bait-and-switch: One bot retreats (bait), enemy chases, team pearls behind enemy
- **Formation**: Rough spatial coordination — don't clump, don't spread too far, maintain engagement range

### Coordination Architecture Options
1. **Centralized coordinator**: One "brain" sees all 20 bots' states, outputs role assignments and macro commands per bot. Each bot still runs its own combat loop. Could be rule-based or learned.
2. **Communication protocol**: Bots broadcast intent (attacking, retreating, need help) and react to teammates' broadcasts. Emergent coordination without central control.
3. **Hierarchical**: Squad leaders (5 bots each) coordinate within squad, squad leaders coordinate with each other.
4. **MARL (Multi-Agent RL)**: Train all 20 bots together with shared reward. Expensive but potentially emergent team play.

### Training Challenges
- Sim needs to support NvN (currently 1v1 only)
- Computational cost of 20v20 training
- Credit assignment: which bot's action led to the team win?
- Emergent communication vs explicit coordination

---

## Technical Details for Reference

### Build & Deploy
```
./gradlew clean build          # Build plugin JAR
python deploy.py deploy         # SFTP to GravelHost server
python deploy.py push <local> <remote>  # Push single file
python deploy.py pull <remote> <local>  # Pull file from server
```

### Spawn Commands
```
mai bot spawn <name> <world> <x> <y> <z> rule [kit] [difficulty]
mai bot spawn <name> <world> <x> <y> <z> hybrid [kit] [difficulty]
mai bot spawn <name> <world> <x> <y> <z> <model_name> [kit]
```

### Key Constants
- Tick rate: 20 tps (50ms per tick)
- Max bots: 8 (config, needs increasing for Legion)
- Arena radius: 30 blocks (BotBrain default)
- Attack reach: 3.0 blocks max
- I-frames: 10 ticks (500ms)
- Combo window: 15 ticks to count as combo hit

### File Structure
```
src/main/java/com/minimalai/
  ai/          ActionSpace, ActionExecutor, RuleCombatEngine, ModelManager, ObservationBuilder
  bot/         BotBrain, FakePlayerManager, KitManager
  commands/    BotCommand, MaiCommand
  integration/ ArcaneSigilsBridge (reflection to ArcaneSigils plugin)
  training/    ExperienceBuffer, RewardComputer, PhysicsRecorder
training/
  config.py    Action/obs constants, reward configs
  vec_sim.py   Vectorized 1v1 physics sim
  fast_train.py PPO trainer
  models/      CombatNetwork, SigilNetwork
```

---

## Questions to Brainstorm

1. **How do we close the gap between rule-based and neural combat?** The rule engine is competitive but predictable. Can we make it adaptive? Or should we keep trying to train neural models that actually work?

2. **What's the right coordination architecture for 20 bots?** Centralized brain vs decentralized communication vs hierarchical. What's realistic to build and train?

3. **How do we extend the sim from 1v1 to NvN?** The vectorized sim is fast but only does 1v1. What's the most efficient way to support team fights?

4. **What team plays should the Legion know?** Beyond the obvious (focus fire, pearl-onto-healer), what coordinated tactics would be devastating against human teams?

5. **How do we handle the training curriculum?** Train individual combat first then add coordination? Or train coordination from scratch with simple combat?

6. **Is MARL the right approach, or can we get away with a simpler coordination layer on top of individual RL/rule agents?**

7. **How do we make the bots feel like a coordinated team rather than 20 independent fighters?** What's the minimum viable coordination that creates that "hive mind" feeling?

8. **Server performance**: Can a single MC server handle 20 fake players with per-tick AI? What optimizations are needed?
