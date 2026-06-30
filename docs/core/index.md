# Codebase Index
**Use this file instead of searching.** Do NOT use Grep, Glob, or Explore agents to find files/structure unless you are 100% certain the answer is not in this index. This file exists to save tokens and time.

## Directory Structure
- `src/main/java/com/minimalai/` — Java plugin source (Paper 1.21.10, Java 21)
  - `ai/` — Action space, observations, model inference, rule engine
  - `bot/` — BotBrain tick loop, FakePlayerManager, KitManager
  - `commands/` — BotCommand (spawn/lifecycle), MaiCommand (/mai handler)
  - `integration/` — ArcaneSigilsAPI interface + ArcaneSigilsBridge (reflection)
  - `training/` — ExperienceBuffer, RewardComputer, PhysicsRecorder, EpisodeManager
- `src/main/resources/` — plugin.yml, config.yml, models/__template.pt
- `training/` — Python training pipeline (PPO, vectorized sim, models)
  - `models/` — CombatNetwork (full), SigilNetwork (standalone, deprecated)
- `models/` — Trained model artifacts
  - `sigil_combat/` — Latest combat model with sigil support
  - `sigils_v6/` — Latest standalone sigil decision model
- `docs/`
  - `core/` — Codebase index, architecture docs
  - `plans/` — Design docs, brainstorm context

## Key Java Files
- `ai/ActionSpace.java` — 35 multi-binary action constants (movement 0-6, combat 7-13, sigils 14-25, camera 26-29, targets 30-34)
- `ai/ActionExecutor.java` — Converts action vector to NMS operations. `ACTIVE_ABILITY_SLOTS = {1,2,3,5}` maps net outputs to ArcaneSigils bind slots
- `ai/RuleCombatEngine.java` — State machine: ENGAGE/COMBO/RETREAT/PEARL_ESCAPE/PEARL_AGGRO + rule sigil timing. 5 difficulty tiers.
- `ai/ModelManager.java` — TorchScript model loading, `infer()` (7 inputs), `inferSigil()` (2 inputs)
- `ai/ObservationBuilder.java` — Builds 320-dim observation tensors from server state
- `ai/ObservationSpace.java` — Dimension constants (SELF=38, ENTITY=24, COMBAT=26, SIGIL=48, ENV=8, HIDDEN=128)
- `bot/BotBrain.java` — 3 modes: NEURAL (full model), RULE (RuleCombatEngine), HYBRID (rule combat + neural sigils)
- `bot/FakePlayerManager.java` — NMS ServerPlayer creation for bots
- `bot/KitManager.java` — Loads kit YAMLs, applies items/enchants
- `commands/BotCommand.java` — Bot spawn/lifecycle. `spawnBot()` reads mode from config or arg ("rule"/"hybrid"/model name)
- `commands/MaiCommand.java` — /mai command handler (bot spawn, train, duel, model management)
- `integration/ArcaneSigilsAPI.java` — Interface: getEquippedSigils, activateAbility, getCooldownProgress, etc.
- `integration/ArcaneSigilsBridge.java` — Reflection bridge to ArcaneSigils plugin
- `MinimalAIPlugin.java` — Plugin entry point, wires everything together

## Key Python Files
- `training/config.py` — Action indices, obs dims, USED_BITS=[0,2,3,4,9,10,11,14,15,16,17,26], consumable/combat constants, reward configs
- `training/vec_sim.py` — Numpy-vectorized 1v1 sim with full physics + 4 sigil abilities (Brace/Cleo/Sand/Grace). ~460K sps.
- `training/fast_train.py` — PPO trainer with self-play or --rule-opponent. Uses CombatNetwork.
- `training/models/network.py` — CombatNetwork: entity attention + GRU, 35-action policy + value heads
- `training/models/sigil_network.py` — SigilNet (standalone, 4 actions, deprecated in favor of integrated training)
- `training/mc_physics.py` — Kitara combat model constants (KB, damage, friction)
- `training/rule_test.py` — Rule-based opponents (rule_charger, rule_charger_noisy, rule_charger_smart)
- `training/evaluate.py` — Evaluation script for models
- `training/sigil_sim.py` — Standalone sigil sim (deprecated — integrated into vec_sim.py)
- `training/train_sigils.py` — Standalone sigil trainer (deprecated)

## Config
- `src/main/resources/config.yml` — Bot mode (default-mode: hybrid), difficulty (1-5), sigil-model, default-sigils, arena center, training settings, rewards
- `build.gradle` — Gradle build, Paper 1.21.10 dependency, DJL (PyTorch) for model inference

## Build & Deploy
- Build: `./gradlew clean build` → `build/libs/MinimalAI-1.0.0.jar`
- Deploy: `python deploy.py deploy` (SFTP to GravelHost)
- Push model: `python deploy.py push <local> <remote>`
- Pull logs: `python deploy.py pull ./logs/latest.log latest-server-log.txt`

## Sigil Loadout (config.yml)
- Exclusives (1 per gear): ancient_crown(helmet), kings_brace(chest), cleopatra(legs), quick_sand(boots), divine_intervention(sword), niles_grace(axe)
- Regulars (passive): iron_forged, lifeforce, extra_padding, iron_fist, rocket_boots, well_fed, spring_shoes, meal_planning
- Active abilities (neural net controls bits 14-17): [0]=Brace(bind 1), [1]=Cleo(bind 2), [2]=Sand(bind 3), [3]=Grace(bind 5)

## Spawn Commands
- `mai bot spawn <name> <world> <x> <y> <z> rule [kit] [difficulty]` — rule combat + rule sigils
- `mai bot spawn <name> <world> <x> <y> <z> hybrid [kit] [difficulty]` — rule combat + neural sigils
- `mai bot spawn <name> <world> <x> <y> <z> <model> [kit]` — full neural mode
