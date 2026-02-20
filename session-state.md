# 2026-02-20 | MinimalAI Pipeline Bug Fixes & Action Space Alignment

DONE:
- Fixed golden apple attack bug: applyAttack now guards against eating state and non-sword items, tickEating restores original hotbar slot after eating completes
- Aligned camera action space: Python sim now uses intent-based camera (FACE_TARGET/FACE_AWAY/LOOK_DOWN_SELF/FACE_MOVEMENT) matching Java, instead of incremental turns
- Wired all dead reward callbacks: ActionExecutor now calls RewardComputer.onBotAttack (whiff/iframe), onBotSprintReset, onBotUseGap, onBotUsePot during execution when training is active
- Fixed calibrate_physics.py to support both old (kb_x/y/z) and new (hit_vel_x/y/z) column names
- Fixed server.py action mask: reconstructs mask from actionProbs instead of hardcoding all-ones (swap weapon + sigils always masked, near-zero probs masked)
- Build successful, Python sim smoke test passed, TorchScript export verified

FILES: ActionExecutor.java, BotBrain.java, training/combat_sim/env.py, training/diagnose.py, training/calibrate_physics.py, training/server.py

NEXT:
1. Deploy to server: `./gradlew clean build && python deploy.py deploy` then restart server
2. Connect MC client, test `/mai bot spawn TestBot default warrior` — verify sword in hand, attacks work
3. Test `/mai record start 60 warrior` — verify recording captures all actions
4. Run `python training/calibrate_physics.py <csv>` on new recording
5. Train in sim: `cd training && python train.py` — first 1M steps, verify rewards trend up
6. Export model: `python training/export.py checkpoints/best.pt --output models/warrior.pt`
7. Load in-game: `/mai model load warrior` then `/mai model duel warrior warrior`
8. If all pass: start batch training with self-play, then deploy trained model

CONTEXT:
- Branch: feature/bot-brain-fixes-and-physics-recording
- Build: `./gradlew clean build` → `python deploy.py deploy`
- MC server connection failing (2026-02-20) — server may be offline or MS auth token expired
- ActionExecutor.execute() now has 6-param overload accepting RewardComputer + botName
- Camera actions 26-29 are now semantically aligned between Java and Python
- All reward callbacks are wired: whiff (-0.05), iframe waste (-0.03), pot timing (±0.3/-0.2), gap timing (+0.1), sprint reset (+0.15)
- Python sim verified: env.step with FACE_TARGET works, network forward pass OK, TorchScript export OK
