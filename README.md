# MinimalAI

Neural network inference running inside Minecraft - PyTorch embedded in a Fabric mod.

<img src="https://img.shields.io/badge/Minecraft-1.21.8-brightgreen" alt="MC 1.21.8"/> <img src="https://img.shields.io/badge/Fabric-Mod-red" alt="Fabric"/> <img src="https://img.shields.io/badge/PyTorch-2.5-orange" alt="PyTorch"/> <img src="https://img.shields.io/badge/Java-21-blue" alt="Java 21"/>

## Overview

MinimalAI is a groundbreaking experiment: **running actual neural networks inside the Minecraft client using PyTorch**. This is extremely rare in Minecraft modding—most AI mods use simple heuristics. This mod embeds the Deep Java Library (DJL) with PyTorch engine to perform real neural network inference in-game.

## What Makes This Special

### 🧠 Real Neural Networks in Minecraft
- Full PyTorch integration via Deep Java Library (DJL)
- Neural network inference running client-side
- Not just "AI-like behavior"—actual deep learning models

### 🎮 Game Automation Framework
- **Game state recording**: Capture gameplay sessions
- **Playback system**: Replay recorded actions
- **Neural network control**: AI makes decisions based on game state
- **UI overlay**: Real-time visualization of AI decisions

### 🔬 Research Platform
- Test neural network architectures in a game environment
- Train models on gameplay data
- Experiment with reinforcement learning in Minecraft

## Technical Achievement

**Challenge:** Minecraft mods run in a sandboxed JVM. Most ML frameworks don't play nice with this.

**Solution:** DJL provides a Java-native interface to PyTorch, allowing neural networks to run alongside game logic without external processes.

**Result:** A Minecraft mod that can:
- Load pre-trained PyTorch models
- Run inference every game tick
- Make decisions based on game state
- Execute actions through normal game controls

## Architecture

```
MinimalAI/
├── ai/                        # Neural network integration
│   ├── Network.java          # DJL/PyTorch wrapper
│   └── AIController.java     # Decision making
├── recording/                 # Game state capture
│   ├── GameRecorder.java     # Record player actions
│   ├── RecordingStorage.java # Save/load recordings
│   └── PlaybackSystem.java   # Replay actions
├── executor/                  # Action execution
│   └── ActionExecutor.java   # Execute AI decisions
└── overlay/                   # Visual feedback
    └── UIRenderer.java        # Display AI state
```

## Technology Stack

### AI/ML
- **Deep Java Library (DJL) 0.31.0** - Java interface for deep learning
- **PyTorch Engine 0.31.0** - Native PyTorch integration
- **PyTorch Native CPU 2.5.1** - CPU-based inference

### Minecraft Integration
- **Fabric Mod Loader** for Minecraft 1.21.8
- **Java 21** with modern language features
- **Gradle** for dependency management with DJL bundling

### Key Dependencies
All DJL dependencies are bundled directly into the mod:
```gradle
include "ai.djl:api:0.31.0"
include "ai.djl.pytorch:pytorch-engine:0.31.0"
include "ai.djl.pytorch:pytorch-native-cpu:2.5.1"
```

## Use Cases

### 1. Game AI Research
- Train neural networks on player gameplay
- Test different architectures in real game scenarios
- Compare AI performance to human play

### 2. Automated Testing
- Record optimal gameplay sequences
- Use AI to detect deviations or bugs
- Regression testing through gameplay

### 3. Player Assistance
- AI co-pilot for complex tasks
- Pattern recognition for game events
- Predictive UI overlays

### 4. Reinforcement Learning
- Real game environment for training
- Direct feedback from game state
- Natural reward signals from game mechanics

## Features

### Recording System
- **State capture**: Player position, inventory, health, nearby entities
- **Action logging**: Movement, attacks, item use, interactions
- **Timestamped sequences**: Frame-perfect replay
- **Storage management**: Save/load multiple recordings

### Neural Network Controller
- **State encoding**: Convert game state to neural network input
- **Inference**: Run PyTorch models every tick
- **Action decoding**: Convert network output to game actions
- **Performance optimization**: Efficient inference without lag

### Visualization Overlay
- **Real-time display** of AI decisions
- **Debug information** about network state
- **Performance metrics** (inference time, accuracy)
- **Recording/playback indicators**

## Why This Is Rare

**Most Minecraft "AI" mods:**
- Use simple if-else logic
- Implement basic pathfinding
- Maybe add some state machines

**MinimalAI:**
- Embeds a full ML framework (PyTorch)
- Runs actual neural network inference
- Can load and execute trained models
- Processes game state through deep learning

**Technical barriers:**
- DJL/PyTorch integration in modded Minecraft is non-trivial
- Performance constraints (inference must be < 50ms/tick)
- Packaging native libraries with Fabric mods
- Debugging ML issues in a game environment

## Future Directions

- **Reinforcement learning**: Train agents within Minecraft
- **Multi-agent systems**: Coordinate multiple AI entities
- **Transfer learning**: Apply models trained elsewhere to Minecraft
- **Real-time training**: Update models during gameplay

## Stats

- **20 Java files**
- Full PyTorch integration
- Fabric mod for Minecraft 1.21.8
- DJL bundled for standalone operation

## Development

### Prerequisites
- Java 21
- Gradle 8+
- Fabric development environment

### Build
```bash
./gradlew build
```

Output: `build/libs/MinimalAI-*.jar`

## License

MIT License

---

*Bringing real neural networks to Minecraft, one inference at a time.*
