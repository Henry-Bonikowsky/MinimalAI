# 2026-02-21 | MinimalAI — Continuous PvP Bots Working

DONE:
- Deployed 10M self-play model (pvp_selfplay_10m) — bots fight on server
- Fixed direct damage system: i-frame check, kill threshold (<1.0 HP → 0), force death state
- Auto-respawn: 20-tick cooldown, deathTime reset, kit reapply (diamond sword + 64 golden apples + diamond armor)
- Anti-idle system: 60-tick threshold, 100-tick forced engagement (sprint + attack)
- Arena radius reduced 30→15 blocks for tighter fights
- Kill stats tracking: deaths per bot, periodic logging, enhanced `mai bot list` display
- Calibrated Python sim to match server: I-frame 20→10 ticks, removed protection DR, simplified KB to binary sprint model
- Trained new calibrated model (checkpoints_v3/pvp_calibrated_10m.pt): 10M steps, 88.5% win rate, 140-tick episodes
- Deployed calibrated model to server — bots fighting at ~2 kills/min sustained
- Result: 214 total kills in ~75 min (old model), ~2.0 kills/min (calibrated model)

NEXT:
1. Train longer (20-50M steps) for more refined tactics
2. Add golden apple healing to sim for better strategy learning
3. Compare calibrated vs uncalibrated model in head-to-head
4. Record physics data for further sim calibration
5. Try 4-bot FFA fights

CONTEXT:
- Branch: feature/bot-brain-fixes-and-physics-recording
- Calibrated checkpoints: `checkpoints_v3/` (v2 was uncalibrated sim)
- Server models: pvp_selfplay_10m (uncalibrated), pvp_calibrated_10m (calibrated)
- Console spawn: `mai bot spawn <name> <world> <x> <y> <z> <model> plain`
- Arena: 61x61 stone at y=77-80, centered at (0,0)
- Both models work — calibrated should be more realistic long-term
