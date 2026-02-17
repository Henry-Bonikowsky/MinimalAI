# MinimalAI - PVP Combat Bot

Training neural networks for Minecraft PVP combat using reinforcement learning.

**Status: Active Development** | **Minecraft 1.21.8** | **Fabric** | **PyTorch 2.5** | **Java 21**

## What it is

A Minecraft mod that uses **real neural networks** (PyTorch via DJL) to learn PVP combat through reinforcement learning. The AI trains by fighting mobs and players, learning advanced techniques like:

- **Combat timing** - Attack cooldown management, critical hits
- **Movement** - Strafing, W-tapping for knockback, positioning
- **Targeting** - Enemy tracking, threat assessment, distance control
- **Item usage** - Totems, potions, ender pearls, shields in combat
- **Tactics** - Combo attacks, health advantage, cover usage

## Why This Exists

Most Minecraft "AI" mods use hardcoded if-else logic. This mod embeds a **full ML framework** and runs **real-time neural network inference** during combat using PPO (Proximal Policy Optimization) reinforcement learning.

**Technical challenge:** Getting PyTorch to run client-side in Minecraft's JVM while maintaining <50ms inference for smooth combat.

## Features

### Reinforcement Learning PVP Training
- **State space:** 100-dimensional vector (health, armor, enemy position, weapons, effects, tactical info)
- **Action space:** 24 discrete actions + 2 continuous (camera)
  - Basic: movement, jump, sprint, attack, use
  - Combos: W-tap, strafe-attack, block-hit
- **Reward shaping:** Damage dealt (+50/heart), kills (+500), combos (+10), health advantage, optimal distance

### Neural Network Integration
- **Deep Java Library** (DJL) for Java ↔ PyTorch
- **PPO algorithm** for policy optimization
- **Real-time inference** every frame (<50ms)
- **Training loop** with episode management

### PVP-Specific Features
- Detects damage dealt/taken automatically
- Tracks kills, deaths, combo counters
- Monitors attack cooldown, health advantage
- Encodes weapons, armor, potions in state
- Combos: W-tap, strafe attacks, block-hitting

## Tech Stack

**AI/ML:**
- Deep Java Library (DJL) 0.31.0
- PyTorch Engine 0.31.0
- PyTorch Native CPU 2.5.1
- PPO reinforcement learning algorithm

**Minecraft:**
- Fabric Mod Loader (MC 1.21.8)
- Java 21
- Gradle with DJL bundled

## Project Structure

```
MinimalAI/
├── combat/            # PVP state collection, rewards, actions
├── ai/                # Neural network, training loop, PPO
└── ui/                # Training metrics overlay
```

## Build

```bash
./gradlew build
```

Output: `build/libs/MinimalAI-*.jar`

## How It Works

1. **State Collection** - Every tick, collect 100 floats: player health/armor/effects, enemy position/health/weapon, hotbar contents, tactical info
2. **Neural Network** - Feed state → network → get action probabilities + value estimate
3. **Action Execution** - Sample actions, execute movement/attack/items, move camera
4. **Reward Calculation** - Track damage dealt/taken, kills, health advantage
5. **Training** - After episode ends (kill/death), update network weights using PPO

## Research Basis

Based on successful Minecraft RL projects:
- Stanford CS229: Deep Q-Learning for zombie combat
- PPO-based PVP bots with 50x damage rewards
- Curriculum learning (start easy, increase difficulty)

**Key insight:** Shaped rewards work better than sparse rewards. Heavy reward for damage dealt, penalties for damage taken, bonuses for kills.

## Current Status

**Working:**
- PyTorch integration ✓
- PVP state collection (100 dims) ✓
- PVP action executor with combos ✓
- Reward detector (damage/kills) ✓

**In Progress:**
- PPO training loop refactoring
- Combat environment manager
- Training UI overlay

**Planned:**
- Auto-spawn training dummies
- Curriculum learning (easy → hard enemies)
- Model checkpointing & evaluation
- Multi-agent training (vs other AIs)

## License

MIT License

---

*Training neural networks to PVP in Minecraft.*
