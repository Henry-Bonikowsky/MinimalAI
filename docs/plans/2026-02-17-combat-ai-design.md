# MinimalAI Combat AI - Design Document
## 2026-02-17

## Vision
Train neural networks for Minecraft PvP combat with Arcane Sigils integration. Multiple skill levels (5 checkpoints). Multi-agent awareness for clan fights (32+ players). Deployable as server-side bots.

## Architecture: Train Python, Deploy Java

### Training Side (Python)
- **Custom Combat Simulator** (Gymnasium): Lightweight PvP physics at 100K+ TPS
  - v1: Core melee (knockback, cooldowns, sprint, W-tap, crits)
  - v2: + Arcane Sigils abilities (damage, buffs, AoE, cooldowns)
  - v3: + Items, potions, shields, terrain
- **PPO Trainer** (PyTorch): Proximal Policy Optimization with GAE
- **Network**: Entity Attention (variable entities) + GRU (temporal) + Action Masking
- **Export**: TorchScript .pt files for each skill level checkpoint

### Deployment Side (Java/Fabric)
- Existing infrastructure: PvPStateCollector, PvPActionExecutor, ArcaneSigilsBridge
- DJL loads TorchScript models for inference-only
- No training code in Java

## State Space (designed for Sigils from day one)

### Self State (30 dims)
- Health, hunger, absorption, armor (4)
- Velocity, ground state (2)
- Look direction (4)
- Combat: cooldown, sprint, sneak, block (4)
- Potion effects: 3 slots x 4 (12)
- Reserved for sigil self-buffs (4)

### Per-Entity Features (20 dims per entity, up to 32)
- Alliance flag: -1 enemy, 0 neutral, +1 ally (1)
- Relative position XYZ (3)
- Distance (1)
- Health, armor (2)
- Velocity XYZ (3)
- Facing direction (2)
- Is facing me (1)
- Weapon type, blocking (2)
- Attack cooldown (1)
- Sigil damage amp/reduction on this entity (2)
- Is current target (1)
- On ground (1)

### Combat Context (22 dims)
- Same as current PvPStateCollector

### Sigil State (12 dims)
- 4 bind slots x 3 (available, cooldown_remaining, cooldown_max)

### Environment (8 dims)
- Same as current

## Action Space (28 discrete actions)

- [0-3] Movement: forward, back, left, right (with conflict masking)
- [4-6] Jump, sneak, sprint
- [7-8] Attack, use (right-click)
- [9-10] Switch weapon, switch consumable
- [11] Consume
- [12-15] Combo: W-tap, strafe-left-attack, strafe-right-attack, block-hit
- [16-19] Sigil bind activation (slots 1-4) - masked when on cooldown
- [20-23] Camera: look up/down/left/right
- [24-27] Target selection: target nearest, target lowest HP, target highest threat, cycle target

## Network Architecture

```
Per-Entity Features (up to 32 x 20)
    --> Entity Encoder: Linear(20, 64) + ReLU (shared weights)
    --> 2-Head Self-Attention (64 dim, 2 layers)
    --> Mean Pool --> 64-dim entity summary

[self_state(30) | entity_summary(64) | combat_ctx(22) | sigils(12) | env(8)]
    = 136 total dims
    --> Linear(136, 256) + ReLU
    --> GRU(256, 128) <-- hidden state from previous tick
    --> Policy Head: Linear(128, 28) with action masking
    --> Value Head: Linear(128, 1)
```

## Training Pipeline

### Phase 1: Custom Sim + PPO (this build)
- Build Gymnasium combat sim with core melee physics
- Implement PPO with GAE, entity attention, action masking
- Sigil action slots present but no-op in sim v1
- Train on CPU (GPU occupied)
- Verify convergence on 1v1 melee

### Phase 2: Add Sigils to Sim
- Implement ability effects in simulator
- Train with full action space

### Phase 3: CraftGround Fine-tuning
- Transfer to real Minecraft at 300 TPS

### Phase 4: Self-Play + Skill Levels
- League training, save 5 checkpoints as difficulty levels

## Reward Function (normalized [-1, +1])

```
Sparse Events:
  Kill:     +1.0
  Death:    -1.0

Dense Combat (per tick):
  Damage dealt:  +(damage / 20) * 0.1
  Damage taken:  -(damage / 20) * 0.05
  Critical hit:  +0.02 bonus

Potential-Based Shaping:
  Phi(s) = 0.3*(my_hp - enemy_hp)/20 + 0.1*in_range + 0.05*facing_enemy
  Reward += gamma * Phi(s') - Phi(s)
```

## PPO Hyperparameters

| Param | Value |
|-------|-------|
| Learning rate | 3e-4 (linear decay) |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Epochs per batch | 4 |
| Minibatch size | 128 |
| Batch size (steps) | 2048 |
| Entropy coeff | 0.01 (decay from 0.05) |
| Value loss coeff | 0.5 |
| Max grad norm | 0.5 |

## Difficulty Levels (Training Checkpoints)

| Level | Name | Checkpoint | Behavior |
|-------|------|------------|----------|
| 1 | Novice | ~10% training | Basic approach, slow reactions |
| 2 | Apprentice | ~25% training | Decent spacing, some combos |
| 3 | Fighter | ~50% training | Good timing, W-taps |
| 4 | Warrior | ~75% training | Strong mechanics, ability use |
| 5 | Master | ~100% training | Full potential, optimal play |

## File Structure

```
training/
  combat_sim/
    __init__.py
    env.py              # Gymnasium combat environment
    physics.py          # Minecraft PvP physics engine
    entities.py         # Agent/entity state management
    renderer.py         # Optional pygame visualization
  models/
    __init__.py
    network.py          # Entity attention + GRU + policy/value heads
    ppo.py              # PPO algorithm with GAE
  train.py              # Main training script
  export.py             # TorchScript export
  config.py             # Hyperparameters
  requirements.txt
  tests/
    test_physics.py     # Physics accuracy tests
    test_env.py         # Gymnasium env tests
    test_network.py     # Network shape/forward pass tests
    test_ppo.py         # PPO gradient/update tests
    test_training.py    # Integration: does training converge?
```
