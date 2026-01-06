## Session State - December 21, 2025

### What was done
- Added Phase 7: Reinforcement Learning Training
  - `SimpleNetworkRL.java` - Network with frame stacking (4 frames × 64 = 256 input) and value head
  - `ExperienceBuffer.java` - Stores episode experiences for REINFORCE
  - `RewardDetector.java` - Hooks into block break events for reward signals
  - `EpisodeManager.java` - Handles episode lifecycle, timeout, death, auto-reset
  - `RLTrainer.java` - REINFORCE algorithm with interleaved BC training
  - `RLController.java` - Main RL training loop controller
  - Added 'L' keybind for RL training mode

### Project structure
```
MinimalAI/
├── src/main/java/com/minimalai/
│   ├── MinimalAI.java (entry point)
│   ├── ModKeybinds.java (Y=Dashboard, R=Record, P=Play, T=Train, I=AI, L=RL, End=Stop)
│   ├── ai/
│   │   ├── SimpleNetwork.java (64→256→128→20+2, for BC)
│   │   ├── SimpleNetworkRL.java (256→256→128→20+2+1, frame stacking + value head)
│   │   ├── Trainer.java (behavior cloning)
│   │   ├── RLTrainer.java (REINFORCE + interleaved BC)
│   │   ├── RLController.java (RL training loop)
│   │   ├── AIController.java (runs trained BC model)
│   │   ├── ExperienceBuffer.java (episode experience storage)
│   │   ├── EpisodeManager.java (episode lifecycle)
│   │   └── RewardDetector.java (block break reward detection)
│   ├── recording/
│   │   ├── GameStateCollector.java
│   │   ├── ActionRecorder.java
│   │   ├── Recording.java
│   │   └── RecordingStorage.java, RecordingManager.java
│   ├── playback/
│   │   ├── ActionExecutor.java
│   │   └── PlaybackController.java
│   └── ui/
│       ├── OverlayRenderer.java
│       └── DashboardScreen.java
```

### RL Training Workflow
1. Record gameplay (R key) to teach action vocabulary
2. BC train (T key) to initialize network with basic behaviors
3. Position player at training spot, press L to start RL training
4. RL loop:
   - BC pass (prevents forgetting)
   - Episode runs until block mined (reward=1), timeout (reward=0), or death (reward=0)
   - REINFORCE update
   - Auto-reset to start position
   - Repeat
5. End key stops training and saves model

### Key Design Decisions
- Frame stacking (4 frames) for temporal context instead of LSTM
- Sparse reward only (block mined = +1), no shaping
- Interleaved BC to prevent catastrophic forgetting
- Episode ends on first block break (simple goal)
- Auto-reset via teleport command

### Current status
RL training infrastructure complete. Ready for testing.

### Next steps
- Test RL training loop in-game
- Verify reward detection works
- Tune hyperparameters if needed
- Consider adding UI indicators for RL mode status
