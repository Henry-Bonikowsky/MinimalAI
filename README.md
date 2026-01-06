# MinimalAI

Embedding PyTorch in a Minecraft Fabric mod to train AI on in-game tasks.

**Status: Experimental/WIP** | **Minecraft 1.21.8** | **Fabric** | **PyTorch 2.5** | **Java 21**

## What it is

A Minecraft mod that runs actual neural networks client-side using PyTorch. The goal was to train AI to perform skills in Minecraft by:
- Recording game state (player position, inventory, nearby entities, etc.)
- Running neural network inference every tick
- Executing actions based on network output

It's broken right now but the core PyTorch integration works.

## Why This is Rare

Most Minecraft "AI" mods use simple if-else logic or basic pathfinding. This embeds a full ML framework (PyTorch via Deep Java Library) and runs real neural network inference alongside game logic.

**Technical challenge:** Getting PyTorch to run in a sandboxed JVM alongside Minecraft without exploding.

## Features (When Working)

### Game Automation
- Record gameplay sessions (state + actions)
- Playback system for replays
- Neural network decides actions based on game state
- UI overlay showing AI decisions in real-time

### Neural Network Integration
- Deep Java Library (DJL) for Java interface to PyTorch
- PyTorch 2.5 CPU inference
- Load pre-trained models
- Run inference every game tick (<50ms requirement)

### Recording System
- Capture player position, inventory, health, nearby entities
- Log movement, attacks, item use, interactions
- Frame-perfect replay
- Save/load multiple recordings

## Tech Stack

**AI/ML:**
- Deep Java Library (DJL) 0.31.0
- PyTorch Engine 0.31.0
- PyTorch Native CPU 2.5.1

**Minecraft:**
- Fabric Mod Loader (MC 1.21.8)
- Java 21
- Gradle with DJL bundled

All DJL dependencies included in mod JAR:
```gradle
include "ai.djl:api:0.31.0"
include "ai.djl.pytorch:pytorch-engine:0.31.0"
include "ai.djl.pytorch:pytorch-native-cpu:2.5.1"
```

## Project Structure

```
MinimalAI/
├── ai/                # Neural network integration
├── recording/         # Game state capture
├── executor/          # Action execution
└── overlay/           # Visual feedback
```

## Use Cases (Planned)

- Train neural networks on player gameplay
- Test different architectures in real game scenarios
- Automated testing through gameplay
- Reinforcement learning in actual game environment

## What Works

- PyTorch integration (DJL loads and runs)
- Basic game state recording
- Action playback system

## What's Broken

- Network inference timing (sometimes lags)
- State encoding could be better
- Training loop not implemented
- UI overlay needs work

## Build

```bash
./gradlew build
```

Output: `build/libs/MinimalAI-*.jar`

## Stats

- 20 Java files
- Full PyTorch integration
- Fabric mod for MC 1.21.8

## License

MIT License

---

*Bringing actual neural networks to Minecraft, even if it's janky.*
