# Training Approaches Beyond Real-Time RL: Comprehensive Research

## Executive Summary

The fundamental bottleneck in MinimalAI is that real-time RL in Minecraft runs at 20 TPS (ticks per second), meaning 1 million training steps takes ~14 hours of wall-clock time. Industry solutions to this fall into 7 categories, each with dramatically different speed/fidelity/complexity tradeoffs. This document evaluates every known approach.

**Bottom line:** The most promising path for MinimalAI is a **hybrid approach** combining:
1. **Offline imitation learning** from recorded PvP data (bootstrap a competent policy in hours, zero game time)
2. **CraftGround (~300 TPS)** for accelerated online RL fine-tuning (15x faster than vanilla MC)
3. A **custom lightweight combat simulator** in Python for hyperparameter search and architecture iteration (10,000x+ faster)

---

## 1. Simulation / Faster-than-Realtime Approaches

### 1a. CraftGround (~300 TPS, 15x Realtime)

**What it is:** A Fabric-based RL environment that runs actual Minecraft at ~300 TPS by bypassing rendering and optimizing the game loop. Created by researcher yhs0602, actively maintained.

**Speed:** ~300 TPS vs Minecraft's 20 TPS = **15x faster than realtime**. With headless mode (no display), this is achievable on a single machine.

**Fidelity:** PERFECT -- it runs actual Minecraft Java Edition. All combat mechanics (knockback, attack cooldowns, critical hits, sprint reset, entity collisions) are 100% faithful because it IS Minecraft.

**Setup:** Python Gymnasium API wrapper around a modified Minecraft client. Supports custom action spaces, observation spaces (visual or structured), initial inventory/mob commands, and flat worlds. Docker support for headless servers.

**Combat suitability:** Environment configuration supports `initialMobsCommands` to spawn enemies, custom inventory, flat worlds for arenas. Action wrappers define discrete movement/attack actions.

**Limitation:** 300 TPS is still only 15x realtime. Training 100M steps would take ~93 hours (~4 days). Not fast enough for rapid experimentation, but viable for final training runs.

**Implementation complexity:** MODERATE. Install via pip, write a Gymnasium env wrapper, integrate with Stable Baselines 3 or custom PPO.

**Source:** [CraftGround GitHub](https://github.com/yhs0602/CraftGround) | [Benchmark Comparison](https://github.com/yhs0602/minecraft-simulator-benchmark)

### 1b. Craftax (250x faster, JAX-based, but NOT Minecraft)

**What it is:** A ground-up rewrite of the Crafter game (Minecraft-inspired) in JAX, running entirely on GPU with massive parallelization (up to 4096 parallel environments).

**Speed:** 250x faster than Crafter, which is itself much faster than Minecraft. PPO with **1 billion steps finishes in under 1 hour** on a single GPU. Hundreds of thousands of steps per second.

**Fidelity to Minecraft PvP:** LOW. Craftax has simplified 2D grid-based combat with physical/fire/ice damage categories, bow/melee weapons, and boss fights. But it has NO PvP mechanics whatsoever -- no knockback, no attack cooldowns, no sprint mechanics, no hitbox calculations, no W-tapping. The combat is turn-based grid combat, fundamentally different from Minecraft PvP.

**Use case:** Excellent for testing RL algorithms and architectures quickly. You could validate that your PPO implementation, reward shaping, and network architecture converge on Craftax before deploying to real Minecraft. But a policy trained on Craftax will NOT transfer to Minecraft PvP.

**Implementation complexity:** LOW. `pip install craftax`, comes with PureJaxRL integration.

**Source:** [Craftax Paper (ICML 2024)](https://arxiv.org/abs/2402.16801) | [GitHub](https://github.com/MichaelTMatthews/Craftax)

### 1c. Craftium (2000+ steps/sec, Minetest-based)

**What it is:** A framework built on Minetest (open-source C++ voxel engine) that creates Minecraft-like 3D RL environments with full Gymnasium compatibility.

**Speed:** Achieves +2,000 steps per second more than Minecraft-based alternatives. Uses Minetest's C++ engine for efficiency. Accepted at ICML 2025.

**Fidelity:** MODERATE. Minetest has voxel-based 3D physics and combat, but the specific mechanics differ from Minecraft Java Edition. No attack cooldowns, different knockback formulas, different entity physics. Could potentially be modded to match Minecraft mechanics more closely since Minetest is open source and extensible.

**Combat suitability:** Supports custom 3D environments with enemies, but PvP mechanics would need custom implementation through Minetest Lua scripting.

**Implementation complexity:** MODERATE-HIGH. Requires understanding Minetest modding (Lua) to create faithful combat mechanics.

**Source:** [Craftium GitHub](https://github.com/mikelma/craftium) | [Paper](https://arxiv.org/abs/2407.03969)

### 1d. MineRL / Malmo (Legacy, ~20 TPS)

**What it is:** Microsoft's Project Malmo and the MineRL competition wrapper. The original Minecraft AI research platform.

**Speed:** Runs at approximately real-time (~20 TPS). No meaningful acceleration. MineRL is built on Malmo which runs a modified Minecraft client.

**Fidelity:** PERFECT (it IS Minecraft), but locked to older MC versions (typically 1.11-1.16). The combat mechanics differ significantly between these versions and 1.21.8 (your target).

**Status:** Largely superseded by CraftGround and MineStudio. Malmo is no longer actively maintained. MineRL dataset is still valuable.

**Source:** [MineRL GitHub](https://github.com/minerllabs/minerl) | [MineDojo](https://minedojo.org/)

### 1e. Vanilla Minecraft Tick Acceleration

**Can Minecraft be run faster than 20 TPS?** Yes, with mods like TickrateChanger that change the server/client tickrate. However:
- Physics become unstable above ~100 TPS (entities clip through blocks, knockback calculations break)
- CPU becomes the bottleneck quickly -- Minecraft's single-threaded game loop cannot process faster than the hardware allows
- Combat mechanics are tied to tick timing (attack cooldown = `20 / attackSpeed` ticks), so changing TPS changes the feel of combat
- Practical limit: maybe 2-3x without breaking things

**Verdict:** Not a viable path for large-scale acceleration.

---

## 2. Imitation Learning / Behavior Cloning

### 2a. VPT (Video Pre-Training) by OpenAI

**What it is:** The most successful Minecraft imitation learning project. Trained a 0.5B parameter model to play Minecraft by watching 70,000 hours of YouTube videos.

**How it works (two-stage pipeline):**
1. **Inverse Dynamics Model (IDM):** Train a small model on ~2,000 hours of contractor data where both video frames AND keyboard/mouse actions are recorded. The IDM learns to predict what action was taken given past and future frames.
2. **Foundation Model:** Use the IDM to pseudo-label 70,000 hours of unlabeled YouTube gameplay. Then train a large behavioral cloning model on this pseudo-labeled data.

**Results:** The behavioral cloning model learned to craft a diamond pickaxe (~24,000 consecutive actions). When fine-tuned with RL on a small reward signal, performance improved further.

**Speed:** Training was on 720 V100 GPUs for 9 days (foundation model). But the KEY insight is that the behavioral cloning phase requires ZERO game interaction -- it's pure supervised learning on video data.

**Relevance to MinimalAI PvP:**
- The IDM approach could work for PvP: record skilled players' screen + inputs during combat, train IDM, then pseudo-label PvP YouTube videos
- HOWEVER: VPT uses pixel input (video frames), while MinimalAI uses structured state (112-dim vectors). You'd need to adapt the approach to work with structured state OR switch to pixel input.
- The compute requirements (720 V100s) are completely out of scope for an individual project
- A much smaller version (1-10 hours of labeled PvP data, no YouTube pseudo-labeling) is feasible

**Source:** [VPT Paper](https://arxiv.org/abs/2206.11795) | [OpenAI Blog Post](https://openai.com/index/vpt/) | [GitHub](https://github.com/openai/Video-Pre-Training)

### 2b. MineRL Competition Approach

**What it is:** Annual competition (2019-2022) where teams trained Minecraft agents with limited compute (8M environment steps + demonstration data).

**Key finding:** Pure RL solutions took ~8 hours to reach the same performance that imitation learning agents achieved in 15 minutes. Every winning team used behavioral cloning as a starting point.

**BASALT Dataset:** 26 million image-action pairs from ~14,000 videos of human players. Available for download. Tasks include building houses, finding caves, creating waterfalls -- not PvP combat specifically.

**Winning approaches:**
- **JueWu-MC (2021 winner):** Hierarchical RL + imitation learning + discriminator-based self-imitation + ensemble behavior cloning with consistency filtering
- **BASALT 2022 baseline:** Behavior cloning using VPT embeddings as input features

**Relevance:** The 53x speed advantage of IL over RL (15 min vs 8 hours) is the single most important finding. For MinimalAI, even a basic behavior cloning bootstrap would dramatically reduce the RL training time needed.

**Source:** [MineRL GitHub](https://github.com/minerllabs/minerl) | [BASALT 2022 BC Baseline](https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline) | [BEDD Dataset](https://arxiv.org/abs/2312.02405)

### 2c. Behavior Cloning for PvP Combat (Practical Path)

**The approach for MinimalAI:**

1. **Data Collection:** Create a Fabric mod that records (state_vector, action) pairs during human PvP play:
   - The 112-dim state vector from PvPStateCollector (already built)
   - The discrete action taken (from PvPActionExecutor's action space, already defined)
   - Record at 20 TPS, 1 hour of play = 72,000 training examples

2. **Training (offline, Python):** Simple supervised learning:
   - Input: 112-dim state vector
   - Output: action probabilities (24 discrete actions + camera direction)
   - Loss: cross-entropy loss on actions
   - Train for 50-100 epochs on the dataset
   - Takes minutes on a single GPU

3. **Result:** A policy that mimics the recorded player's behavior. Not optimal, but a massive head start for RL fine-tuning.

**Speed advantage:** Data collection is 1:1 real-time (1 hour of play = 1 hour), but TRAINING is instant (minutes). The policy doesn't need to explore or discover strategies -- it copies human knowledge directly.

**Proven to work:** Fighting game AI research shows behavior cloning from replay data successfully captures complex combat behaviors including timing, spacing, and combos. The key is sufficient data diversity.

**Implementation complexity:** LOW. You already have the state collector and action executor. You just need to add a recording mode that saves (state, action) tuples to disk.

---

## 3. Offline Reinforcement Learning

### 3a. Decision Transformer

**What it is:** Casts RL as sequence modeling. Given a sequence of (return-to-go, state, action) tuples, a transformer predicts the next action conditioned on a desired return.

**How it works:** At inference time, you specify the desired return (e.g., "I want +100 reward") and the model generates actions that should achieve that return, based on patterns in the offline dataset.

**Speed:** Training is pure supervised learning on logged data -- no environment interaction needed. Google trained a Multi-Game Decision Transformer on 1B Atari frames that plays 41 games simultaneously.

**Fidelity:** Depends entirely on the quality of the offline dataset. If the dataset contains only mediocre play, the model cannot exceed mediocre performance.

**Relevance to MinimalAI:**
- Could train a Decision Transformer on logged PvP combat data (state, action, reward sequences)
- Condition on high returns at inference time to get "optimal" behavior
- Requires a large, diverse dataset with varying skill levels
- The return-conditioning mechanism naturally handles the "what skill level to imitate" problem

**Implementation complexity:** MODERATE. Standard transformer architecture, but requires careful dataset curation and return normalization.

**Source:** [Decision Transformer Paper](https://arxiv.org/abs/2106.01345) | [Multi-Game DT (Google)](https://research.google/blog/training-generalist-agents-with-multi-game-decision-transformers/)

### 3b. AlphaStar Unplugged (Offline RL Benchmark)

**What it is:** DeepMind's benchmark for offline RL using millions of StarCraft II human replays. Demonstrated that offline RL can achieve 90% win rate against the supervised AlphaStar agent.

**Methods tested:**
- Behavior cloning (baseline)
- Offline actor-critic
- Offline MuZero (model-based)
- Conservative Q-Learning (CQL)

**Key finding:** Offline RL significantly outperformed behavior cloning, but required careful algorithm design to avoid distributional shift (the agent encountering states not in the dataset).

**Relevance:** Proves that offline RL on combat game data IS viable at scale. The Minecraft equivalent would be: collect a large dataset of PvP replays, train offline, then optionally fine-tune online.

**Source:** [AlphaStar Unplugged Paper](https://arxiv.org/abs/2308.03526) | [DeepMind Blog](https://www.marktechpost.com/2023/08/14/deepmind-researchers-introduce-alphastar-unplugged/)

### 3c. Conservative Q-Learning (CQL) / Implicit Q-Learning (IQL)

**What they are:** Offline RL algorithms that learn from fixed datasets without environment interaction. CQL adds a regularization term that penalizes Q-values for out-of-distribution actions. IQL avoids querying the Q-function for unseen actions entirely.

**Speed:** Pure offline training. Hundreds of thousands of gradient steps per hour on GPU.

**Practical limitation:** These methods are conservative by design -- they never exceed the best behavior in the dataset. For PvP, this means the AI will never develop novel strategies beyond what humans demonstrated.

**Verdict:** Useful as a bootstrap method (like BC but better), not as a final training approach for superhuman PvP.

---

## 4. Sim-to-Real / Simplified Environments

### 4a. Domain Randomization for Sim-to-Real Transfer

**Concept:** Train in a simplified/fast simulator with randomized physics parameters, so the policy learns to be robust to parameter variations. When deployed in the "real" environment (full Minecraft), the policy generalizes because it has seen many parameter configurations.

**Applied to Minecraft PvP:** Train in a fast simulator with randomized:
- Knockback strength (+/- 20%)
- Attack cooldown timing (+/- 10%)
- Movement speed variation
- Hit registration delay (simulating latency)
- Damage values

**Proven track record:** OpenAI used domain randomization to transfer a dexterous manipulation policy from simulation to a real robot hand (zero-shot). The policy trained in simulation with randomized physics worked on the physical robot without any fine-tuning.

**Relevance:** If we build a custom combat simulator (Section 7), domain randomization is essential for ensuring the trained policy transfers to real Minecraft. Without it, the policy would overfit to the simulator's specific physics.

**Implementation complexity:** LOW (on top of an existing simulator). Just add random perturbations to physics parameters at the start of each episode.

**Source:** [Domain Randomization Survey](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/) | [Understanding DR for Sim-to-Real](https://arxiv.org/abs/2110.03239)

### 4b. Training in 2D Then Transferring to 3D

**Concept:** Build a 2D top-down combat simulator that captures the essential PvP dynamics (distance, knockback, attack timing), train a policy there at 100,000x speed, then fine-tune in real Minecraft.

**What transfers:** Distance management, attack timing, knockback exploitation (W-tapping patterns), health advantage tracking, engagement/disengagement decisions.

**What doesn't transfer:** Vertical movement (jumping for crits), 3D camera control, visual targeting, block-level terrain navigation.

**Speed advantage:** A numpy/JAX 2D simulator can run millions of steps per second. Combined with vectorized environments, you could train billions of steps in minutes.

**Implementation complexity:** MODERATE. The 2D simulator is simple, but the transfer gap requires careful bridge design (what features are shared between 2D and 3D representations).

---

## 5. Hybrid Approaches

### 5a. AlphaStar Pipeline (Supervised Pretraining + RL Fine-tuning)

**The gold standard for competitive game AI.**

**Pipeline:**
1. **Supervised pretraining:** Behavior cloning on 971,000 human replays. Bot reaches ~Diamond level (competitive human level).
2. **Fine-tuning on winning replays:** Further BC on only the top 22% of games. Win rate against elite bot: 87% -> 96%.
3. **RL fine-tuning via self-play:** PPO/V-trace against copies of itself and a league of past agents. Reaches Grandmaster (top 0.2% of humans).

**Key insight:** Supervised learning provides 90% of the final performance. RL provides the last 10% that pushes past human-level. Starting RL from scratch is orders of magnitude slower.

**Applied to MinimalAI:**
1. Record 5-20 hours of skilled PvP play (structured state + actions)
2. Train behavior cloning model offline (minutes)
3. Fine-tune with PPO in CraftGround (~300 TPS) or real Minecraft (20 TPS)
4. Optional: add self-play once base policy is competent

**Source:** [AlphaStar Paper](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)

### 5b. RLHF for Game AI (Human Preference Reward Model)

**Concept:** Instead of hand-crafting a reward function, show humans pairs of combat clips and ask "which agent played better?" Train a reward model from these preferences, then use the reward model for RL training.

**Proven track record:** OpenAI used this approach for Atari games before applying it to language models. The human preference reward model sometimes taught BETTER behavior than score-based rewards because humans encode more nuanced criteria.

**Advantage over hand-crafted rewards:** Humans can judge "good PvP play" holistically (positioning, timing, aggression, adaptation) without needing to decompose it into 15 separate reward terms.

**Practical limitation for MinimalAI:** Requires a human in the loop to label preferences. Need ~1,000-5,000 preference comparisons. Each comparison takes ~10 seconds = ~8-14 hours of labeling work.

**Source:** [RLHF for Game AI (OpenAI)](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback)

### 5c. MineStudio: Full Pipeline Integration

**What it is:** A comprehensive Minecraft AI development package (December 2024) that integrates simulator, data management, models, offline pretraining, online finetuning, inference, and benchmarking into a single framework.

**Key features:**
- Customizable Minecraft simulator wrapper (based on MineRL)
- Efficient trajectory data structure for storing/retrieving experience
- Pre-integrated models: VPT, STEVE-1, GROOT-1, ROCKET-1
- Offline pretraining pipeline
- Distributed online RL with crash recovery
- Ray-based parallel inference

**Why it matters:** MineStudio is essentially a pre-built version of the infrastructure you'd need for the hybrid approach. Instead of building data recording, offline training, and online fine-tuning from scratch, you could potentially use MineStudio's pipeline.

**Limitation:** Built around MineRL (older MC versions), may need adaptation for 1.21.8. Focused on general tasks, not PvP specifically.

**Source:** [MineStudio GitHub](https://github.com/CraftJarvis/MineStudio) | [Paper](https://arxiv.org/abs/2412.18293)

### 5d. LLM-Guided Planning (Voyager-style)

**What it is:** Using GPT-4 or similar LLMs to generate high-level combat plans, with a lower-level policy executing the plans.

**Voyager's approach:** GPT-4 generates JavaScript code (via Mineflayer API) for Minecraft tasks. Achieves 3.3x more unique items and unlocks tech tree milestones 15.3x faster than prior methods.

**Relevance to PvP:** LOW. LLMs operate at ~1 second latency (API call round-trip), while PvP combat requires decisions every 50ms (20 TPS). LLMs could potentially handle strategic decisions (when to engage, weapon selection, potion timing) but cannot handle the moment-to-moment combat mechanics (strafing, W-tapping, aim tracking).

**Possible hybrid:** LLM for strategic layer + RL for tactical execution. The LLM says "engage the low-health enemy from the right flank" and the RL policy handles the actual movement and attacks.

**Source:** [Voyager](https://voyager.minedojo.org/) | [GitHub](https://github.com/MineDojo/Voyager)

---

## 6. Minecraft-Specific Projects & Platforms

### Comprehensive Platform Comparison

| Platform | Speed | MC Version | Combat Support | Active | License |
|----------|-------|------------|----------------|--------|---------|
| **CraftGround** | ~300 TPS | Latest (1.21+) | Good (configurable mobs/arena) | Yes | LGPL v3 |
| **MineRL** | ~20 TPS | 1.11-1.16 | Limited (mob fighting) | Declining | MIT |
| **MineDojo** | ~20 TPS | 1.16 | Task suite includes combat | Moderate | MIT |
| **Malmo** | ~20 TPS | 1.11 | Basic (mob combat) | Archived | MIT |
| **Craftax** | 250x Crafter | N/A (custom) | 2D grid combat (no PvP) | Yes | Apache 2.0 |
| **Craftium** | +2K SPS | N/A (Minetest) | Customizable (Lua scripting) | Yes | GPL 3.0 |
| **MineStudio** | ~20 TPS | 1.16 (MineRL) | Via MineDojo tasks | Yes | Apache 2.0 |

### PvP-Specific GitHub Projects

| Project | Approach | Status | Noteworthy |
|---------|----------|--------|------------|
| **GiaoShou66/Minecraft-PVP-bot** | PPO + CNN (screen capture) | Proof of concept | Uses screen pixels, not structured state |
| **a9800/minecraft-pvp-ai** | Bot controlling player character | Active | Enclosed arena PvP specialization |
| **Meindo/meinbot** | JavaScript (Mineflayer) | Active | Rule-based + scripted PvP |
| **tdvne/Bot-Practice** | Practice bot plugin | Active | Not ML-based, rule-driven |
| **G1axPracticeBot** | Crystal PvP plugin | Active | Server-side practice bots |

**Key finding:** NO published project has achieved strong PvP performance through ML. All successful PvP bots use rule-based systems or scripted behavior. The ML-based attempts (GiaoShou66) are proof-of-concept quality. This is an open research problem.

---

## 7. The Nuclear Option: Custom Combat Simulator

### Why Build One?

The speed numbers tell the story:
- Real Minecraft: 20 TPS = ~2 hours per 100K steps
- CraftGround: 300 TPS = ~6 minutes per 100K steps
- Custom Python sim: 100K+ TPS = **1 second per 100K steps**
- Custom JAX sim (GPU): 1M+ TPS = **0.1 seconds per 100K steps**

For hyperparameter search, architecture iteration, and reward function tuning, you need to run HUNDREDS of experiments. At 2 hours per experiment (Minecraft) vs 1 minute per experiment (custom sim), the custom sim is the only viable option for research velocity.

### What Physics Matter for PvP?

Based on Minecraft Wiki documentation and PvP guides, the critical mechanics are:

**Attack System:**
- Attack cooldown: `T = 20 / attackSpeed` ticks (sword = 1.6 speed = 12.5 ticks)
- Damage multiplier: `0.2 + ((t + 0.5) / T)^2 * 0.8`, clamped to [0.2, 1.0]
- Critical hit: 1.5x damage when falling + cooldown full
- Sweep attack: reduced damage to nearby entities

**Knockback System:**
- Base: halve current velocity, subtract directional force (0.4 magnitude)
- Vertical: add 0.4 upward velocity (capped)
- Sprint: extra knockback from sprinting attacker
- Knockback enchantment: +105% (I) or +190% (II)
- Knockback resistance: 10% per netherite armor piece
- Direction: calculated from position delta between attacker and target

**Movement System:**
- Walk speed: 4.317 m/s
- Sprint speed: 5.612 m/s (1.3x walk)
- Sprint-jump: ~7.127 m/s with momentum
- W-tap: releasing W briefly resets sprint for maximum knockback on next hit
- Strafing: full speed perpendicular to facing direction
- Jump: 1.25 blocks high, gravity = 0.08 blocks/tick downward

**Health/Armor:**
- 20 HP (10 hearts)
- Armor reduces damage by percentage
- Protection enchantment adds further reduction
- Absorption hearts from golden apples
- Regeneration from food/potions

### Implementation Estimate

**Python/NumPy version (CPU):**
- ~500-800 lines of code
- 2D top-down (ignore Y axis for v1) or simplified 3D
- Two agents with position, velocity, health, cooldown state
- Gymnasium-compatible interface
- Expected speed: 50K-200K steps/second on CPU
- Development time: 2-3 days

**JAX version (GPU-vectorized):**
- Same logic but jit-compiled and vmapped
- 1024-4096 parallel environments on a single GPU
- Expected speed: 1M-10M steps/second
- Development time: 1-2 additional days on top of Python version
- Based on PureJaxRL patterns (same approach as Craftax)

**What to include in v1:**
- 2D arena (top-down, ignore height for now)
- Two agents with: position (x,z), velocity (x,z), facing angle, health, attack cooldown
- Melee attack with range check, cooldown, damage calculation
- Knockback physics (simplified)
- Sprint/walk toggle affecting speed and knockback
- W-tap mechanic (sprint reset)
- Basic critical hit (simplified: random chance when moving fast)

**What to add in v2:**
- Y axis (jumping, falling, critical hit detection)
- Armor and enchantments
- Item switching (sword, axe, shield)
- Shield blocking
- Potion effects (speed, strength, regeneration)

### Has Anyone Done This for Other Games?

YES. Multiple precedents:

- **LF2Gym:** OpenAI Gym environment for Little Fighter 2 (2.5D fighting game). Simplified physics, custom reward functions. Used for DRL research.
- **Street Fighter RL:** Custom Gym environment wrapping Street Fighter 2 ROM. Tournament-style training between agents.
- **Blade & Soul AI (NCSOFT):** Custom simulator for 1v1 PvP combat training. Achieved 62% win rate against professional players. Key insight: the simulator captured the "essence" of combat (hitboxes, cooldowns, positioning) without full graphical fidelity.
- **Imitation Learning for RTS Combat:** Custom simplified combat simulators using influence maps and case-based reasoning for strategy games.

### Sim-to-Real Transfer Viability

**Will a policy trained in a custom simulator work in real Minecraft?**

For the STRATEGIC components (when to attack, distance management, engagement decisions): YES, with domain randomization.

For PRECISE MECHANICS (exact knockback values, pixel-perfect aim, frame-perfect timing): NO. These require fine-tuning in real Minecraft or CraftGround.

**Recommended transfer pipeline:**
1. Train base policy in custom simulator (fast: millions of steps in minutes)
2. Fine-tune in CraftGround (medium: hours to days at 300 TPS)
3. Final polish in real Minecraft if needed (slow: real-time)

This is the same pattern used in robotics: train in fast simulation, transfer to slow reality. Domain randomization during step 1 is critical for making step 2 work.

---

## 8. Comparative Analysis: Speed vs Fidelity vs Complexity

| Approach | Speed (vs realtime) | PvP Fidelity | Implementation | Proven? | Transfer to MC? |
|----------|---------------------|--------------|----------------|---------|-----------------|
| **Vanilla MC RL** | 1x (20 TPS) | Perfect | Already built | Partially | N/A (is MC) |
| **CraftGround** | 15x (300 TPS) | Perfect | Moderate | Yes (papers) | N/A (is MC) |
| **Craftax** | 250x+ | None (2D grid) | Low | Yes (ICML) | No |
| **Craftium** | ~100x+ | Low-Moderate | Moderate-High | Partial | Likely not |
| **Behavior Cloning** | Infinite (offline) | N/A (data quality) | Low | Yes (VPT, MineRL) | N/A |
| **Decision Transformer** | Infinite (offline) | N/A (data quality) | Moderate | Yes (Atari, SC2) | N/A |
| **Custom Sim (Python)** | 5,000-10,000x | Low-Moderate | Moderate | Yes (other games) | With fine-tuning |
| **Custom Sim (JAX/GPU)** | 50,000-500,000x | Low-Moderate | Moderate-High | Yes (Craftax) | With fine-tuning |
| **Offline RL (CQL/IQL)** | Infinite (offline) | N/A (data quality) | Moderate | Yes (SC2) | N/A |
| **Hybrid (BC + RL)** | BC: instant, RL: 15x | Perfect (RL in MC) | Moderate | Yes (AlphaStar) | N/A |

---

## 9. Recommended Strategy for MinimalAI

### Phase 0: Data Collection Infrastructure (1-2 days)
Build a recording mode into the existing Fabric mod:
- On key press, start recording (state_vector, action, reward) tuples at 20 TPS
- Save to disk as binary or CSV
- Record 5-20 hours of human PvP play (or mob fighting as proxy)
- This data enables ALL offline approaches

### Phase 1: Behavior Cloning Bootstrap (1-2 days)
- Train a simple BC model in Python (supervised learning on recorded data)
- Input: 112-dim state vector
- Output: 24 discrete action probabilities + camera direction
- Export to TorchScript, load in Java via DJL
- Result: an agent that mimics your play style. Not optimal, but a starting point.

### Phase 2: Custom Combat Simulator + PPO (1-2 weeks)
- Build a lightweight Python/Gymnasium combat simulator
- Implement core PvP physics (knockback, cooldowns, sprint, damage)
- Train PPO with domain randomization at 100K+ TPS
- Iterate on: network architecture, reward shaping, hyperparameters
- Run hundreds of experiments quickly to find what works

### Phase 3: CraftGround Fine-Tuning (1-2 weeks)
- Take the best policy from Phase 2
- Fine-tune in CraftGround (real Minecraft at 300 TPS)
- Bridge the sim-to-real gap with gradual transfer
- Add full 3D mechanics (jumping, vertical aim, critical hits)

### Phase 4: Self-Play + Polish (ongoing)
- Implement self-play in CraftGround
- Save checkpoints, train against league of past selves
- Optionally add RLHF for reward model refinement

### Expected Timeline

| Phase | Wall-Clock Time | Training Steps | Notes |
|-------|-----------------|----------------|-------|
| Phase 0 | 1-2 days coding + 5-20 hours recording | N/A | Human data collection |
| Phase 1 | Hours | N/A (supervised) | Pure offline, no game time |
| Phase 2 | 1-2 weeks | 100M-1B (in simulator) | Fast iteration in Python |
| Phase 3 | 1-2 weeks | 10M-50M (in CraftGround) | 300 TPS, real MC mechanics |
| Phase 4 | Ongoing | Continuous | Self-play improvement |

---

## 10. What to Drop

The current approach of pure real-time RL in vanilla Minecraft (20 TPS) with REINFORCE should be **abandoned**. The math is clear:

- 100M steps at 20 TPS = 5,000,000 seconds = **58 days** of continuous training
- Even with PPO, you need ~10M-50M steps for a combat domain
- That's 6-30 days of wall-clock time with zero interruptions

No published Minecraft RL project has achieved strong PvP through online RL alone. Every successful approach uses some combination of:
1. Offline data (BC or offline RL)
2. Faster-than-realtime simulation
3. Transfer from simpler environments

The MinimalAI architecture (state collector, action executor, reward detector) is EXCELLENT infrastructure. The bottleneck is the training loop, not the game interface.

---

## Sources

### Simulation & Environments
- [CraftGround GitHub](https://github.com/yhs0602/CraftGround) - ~300 TPS Minecraft RL environment
- [CraftGround Benchmark](https://github.com/yhs0602/minecraft-simulator-benchmark) - Simulator performance comparison
- [Craftax (ICML 2024)](https://arxiv.org/abs/2402.16801) - JAX-based 250x faster Minecraft-like benchmark
- [Craftium (ICML 2025)](https://github.com/mikelma/craftium) - Minetest-based RL environments
- [MineDojo](https://minedojo.org/) - Open-ended Minecraft AI platform
- [MineRL GitHub](https://github.com/minerllabs/minerl) - MineRL competition package

### Imitation Learning
- [VPT by OpenAI](https://openai.com/index/vpt/) - Video Pre-Training for Minecraft
- [VPT Paper](https://arxiv.org/abs/2206.11795)
- [VPT GitHub](https://github.com/openai/Video-Pre-Training)
- [MineRL BASALT BC Baseline](https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline)
- [BEDD Dataset](https://arxiv.org/abs/2312.02405) - 26M image-action pairs
- [Playing Minecraft with Behavioural Cloning](http://proceedings.mlr.press/v123/kanervisto20a/kanervisto20a.pdf)

### Offline RL
- [AlphaStar Unplugged](https://arxiv.org/abs/2308.03526) - Offline RL on StarCraft II replays
- [Multi-Game Decision Transformer](https://research.google/blog/training-generalist-agents-with-multi-game-decision-transformers/)
- [Decision Transformer Paper](https://arxiv.org/abs/2106.01345)

### Hybrid & Competitive Game AI
- [AlphaStar Blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- [Blade & Soul Combat AI](https://arxiv.org/abs/1904.03821) - Pro-level fighting game AI
- [JueWu-MC (MineRL 2021 Winner)](https://www.ijcai.org/proceedings/2022/452)

### Sim-to-Real Transfer
- [Domain Randomization Survey](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/)
- [Understanding DR for Sim-to-Real](https://arxiv.org/abs/2110.03239)

### Combat Mechanics
- [Minecraft Damage Wiki](https://minecraft.fandom.com/wiki/Damage)
- [Minecraft Knockback Wiki](https://minecraft.fandom.com/wiki/Knockback)
- [Minecraft PvP Tutorial Wiki](https://minecraft.fandom.com/wiki/Tutorials/Player_versus_Player)
- [Minecraft Combat Calculator](https://www.qxbytes.com/combat/)

### Frameworks & Tools
- [MineStudio](https://github.com/CraftJarvis/MineStudio) - Streamlined Minecraft AI development
- [Voyager](https://voyager.minedojo.org/) - LLM-powered Minecraft agent
- [LF2Gym](https://github.com/elvisyjlin/lf2gym) - Fighting game RL environment
- [Street Fighter RL](https://github.com/corbosiny/AIVO-StreetFigherReinforcementLearning)
- [Gymnasium Custom Environments](https://gymnasium.farama.org/introduction/create_custom_env/)

### PvP Bot Projects
- [Minecraft-PVP-bot (GiaoShou66)](https://github.com/GiaoShou66/Minecraft-PVP-bot) - PPO + CNN PvP
- [minecraft-pvp-ai (a9800)](https://github.com/a9800/minecraft-pvp-ai) - Enclosed arena PvP
- [meinbot (Meindo)](https://github.com/Meindo/meinbot) - JavaScript PvP bot
- [G1axPracticeBot](https://github.com/AkaTriggered/G1axPracticeBot) - Crystal PvP practice
