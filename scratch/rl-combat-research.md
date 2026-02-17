# Reinforcement Learning for Real-Time Combat AI: Research & Recommendations

## Executive Summary

After researching the state of the art in RL for real-time game combat, the recommendation for MinimalAI is **PPO with Generalized Advantage Estimation (GAE)** as the core algorithm, paired with an **entity attention module** for variable entity counts, and a **GRU recurrent layer** for temporal context (replacing frame stacking). This document explains why, compares alternatives, and provides a concrete phased implementation plan.

---

## 1. Why REINFORCE Is Not Converging (Current Problem Diagnosis)

The current implementation has several structural issues that prevent convergence:

### Issue 1: REINFORCE Has Extremely High Gradient Variance
REINFORCE (vanilla policy gradient) scales the gradient of every action by the full episode return. Even with a baseline (value function), the variance is enormous because there is no clipping mechanism to prevent destructive policy updates. A single lucky or unlucky episode can swing the policy wildly.

### Issue 2: On-Policy Without Sufficient Samples
The current batch size of 4 episodes is far too small. REINFORCE needs hundreds or thousands of trajectories to get stable gradient estimates. At 20 ticks/sec with episodes lasting 15-30 seconds, collecting enough samples is prohibitively slow.

### Issue 3: Reward Scale Mismatch
The reward values span from -1000 (death) to +500 (kill) with per-tick rewards of +/-5. This creates massive return variance. The normalized advantages help but cannot fully compensate for reward signals differing by 200x in magnitude.

### Issue 4: No Trust Region / Clipping
Without PPO's clipping mechanism, a single batch update can completely destroy the policy. The agent can go from "reasonable behavior" to "spin in circles" in one update because there is no constraint on how much the policy can change.

### Issue 5: Manual Backpropagation Fragility
Hand-coded backpropagation in Java (with DJL NDArrays) is extremely prone to subtle numerical bugs. The memory management overhead with `.close()` calls creates risk of computing on stale or freed arrays.

---

## 2. Algorithm Comparison

### PPO (Proximal Policy Optimization) -- RECOMMENDED

**How it works:** On-policy actor-critic with a clipped surrogate objective that prevents destructive policy updates. Uses GAE (Generalized Advantage Estimation) for variance-reduced advantage computation.

**Combat domain fit:** EXCELLENT
- Used by OpenAI Five to defeat world champions at Dota 2 (real-time, multi-agent, combat-heavy)
- Used by AlphaStar alongside V-trace for StarCraft II (also real-time strategy with combat)
- Handles mixed discrete actions well (move, attack, ability use)
- Stable training even with sparse/noisy reward signals
- The clipped objective is specifically designed to prevent the exact failure mode you're seeing (policy collapse after a bad batch)

**Variable entity handling:** Good with modifications
- Standard PPO uses fixed-size input; needs an attention or pooling layer to handle variable entities
- Can be combined with entity encoders that produce fixed-size representations

**CPU inference:** EXCELLENT
- Forward pass is just matrix multiplications; very fast on CPU
- OpenAI Five runs inference on CPU with GPU used only for training
- Inference time scales linearly with network size, easily under 1ms for the sizes needed

**Implementation complexity in Java/DJL:** MODERATE
- The clipped surrogate objective is straightforward to implement
- GAE computation is ~20 lines of code
- The main complexity is getting the ratio computation right (old_probs vs new_probs)
- DJL's PyTorch backend handles autograd, which eliminates manual backprop bugs

**Track record:**
- OpenAI Five (Dota 2): PPO + self-play, 256 GPUs, defeated world champions
- Hide and Seek (OpenAI): PPO, emergent tool use and team strategies
- Roller Champions (Ubisoft): PPO + self-play for full-game AI
- PracticeBotPvP (Minecraft plugin): Uses PPO-like updates for combat bots

**Key hyperparameters:**
- Clip range: 0.1-0.2 (0.2 is standard, 0.1 for fine-tuning)
- GAE lambda: 0.95-0.99
- Learning rate: 3e-4 (with linear or cosine decay)
- Epochs per batch: 3-10 (4 is standard)
- Minibatch size: 64-256
- Entropy coefficient: 0.01-0.05 (with decay)
- Value loss coefficient: 0.5-1.0

### SAC (Soft Actor-Critic)

**How it works:** Off-policy actor-critic that maximizes both expected reward and entropy. Uses a replay buffer to reuse old experiences.

**Combat domain fit:** MODERATE
- Designed for continuous action spaces (robotic control); less natural for discrete combat actions
- Discrete SAC exists but is less mature than PPO for games
- The entropy maximization is built-in (no need for separate entropy coefficient tuning)
- Off-policy nature means higher sample efficiency per interaction

**Variable entity handling:** Good with modifications (same as PPO)

**CPU inference:** GOOD (comparable to PPO)

**Implementation complexity:** HIGH
- Requires TWO Q-networks (plus target networks) in addition to the policy network
- Replay buffer management adds memory overhead
- Temperature (alpha) auto-tuning adds another moving part
- More total parameters to train and maintain

**Track record:** Primarily used in robotics (manipulation tasks, locomotion), less common in game AI. No major published game AI results comparable to PPO's.

**Verdict:** More sample-efficient than PPO but more complex to implement and less proven in game combat specifically. The discrete action version is immature. **Not recommended as primary algorithm.**

### IMPALA / APPO (Asynchronous PPO)

**How it works:** Distributed architectures where multiple workers collect experience in parallel while a central learner updates the policy. IMPALA uses V-trace importance correction; APPO combines this with PPO's clipping.

**Combat domain fit:** GOOD (when you have the infrastructure)
- Designed for massively parallel training (hundreds of environments)
- V-trace corrects for the policy lag between actors and learner

**Variable entity handling:** Same as PPO (orthogonal concern)

**CPU inference:** EXCELLENT (same as PPO, just the inference part)

**Implementation complexity:** VERY HIGH
- Requires distributed computing infrastructure (multiple game instances)
- Actor-learner architecture with asynchronous communication
- V-trace importance sampling is mathematically complex
- Debugging distributed RL is notoriously difficult

**Track record:** DeepMind IMPALA paper showed state-of-the-art on multiple Atari games. Sample Factory's APPO runs efficiently on a single machine.

**Verdict:** Overkill for your current setup. This becomes relevant when you need 10-100x faster training by running many Minecraft instances in parallel. **Consider for Phase 3+, not now.**

### Self-Play (OpenAI Five / AlphaStar style)

**How it works:** The agent trains against copies of itself (or a league of past versions). This automatically generates a curriculum of increasing difficulty.

**Combat domain fit:** EXCELLENT for PvP specifically
- Naturally creates an arms race that pushes skill levels higher
- Avoids the need to hand-craft opponent difficulty
- OpenAI Five used self-play to reach superhuman Dota 2 play
- AlphaStar used a league of agents with different playstyles

**Implementation complexity:** MODERATE (on top of PPO)
- Needs the ability to run two agents simultaneously (or serialize past policies)
- League training (AlphaStar style) requires managing multiple checkpoints
- Simplest form: current policy vs. recent checkpoint (every N episodes)

**Track record:** The gold standard for competitive game AI. All major game AI breakthroughs used self-play.

**Verdict:** Essential for reaching strong PvP play, but requires a working base algorithm first. **Implement after PPO is converging (Phase 2).**

### Attention-Based Architecture for Variable Entities

**How it works:** Instead of hard-coding "closest enemy" into a fixed-size state vector, pass all visible entities through a small transformer/attention layer that produces a fixed-size representation regardless of entity count.

**Why it matters for your project:**
- Current state collector only tracks the closest enemy (40 dims). In a 32-player game, the agent is blind to 31 other entities.
- Attention mechanisms naturally learn who to focus on (closest threat? lowest health enemy? ally being attacked?)
- AlphaStar processes all visible units through a 2-layer transformer with 2-headed self-attention and embedding size 128

**Architecture pattern (AlphaStar-inspired):**
```
Per-entity features (alliance, health, distance, weapon, facing, velocity)
    --> Entity Encoder (shared MLP: per_entity_dim -> 64)
    --> Multi-Head Self-Attention (2 heads, 64 dim, 2 layers)
    --> Mean Pool over all entities -> 64-dim entity summary
    --> Concatenate with self-state and context features
    --> Standard MLP/GRU policy network
```

**CPU inference cost:** ~0.5ms for 32 entities with 64-dim embeddings and 2-layer attention. Completely feasible at 20 ticks/sec.

**Implementation in DJL:** DJL supports transformer layers through PyTorch backend. The attention computation is just scaled dot-product attention: `softmax(QK^T / sqrt(d)) * V`.

**Verdict:** HIGH VALUE improvement. The single biggest architectural upgrade you can make. **Implement in Phase 1 alongside PPO.**

---

## 3. Reward Shaping Best Practices

### What Worked for OpenAI Five (Dota 2)

OpenAI Five's reward function included:
- **Net worth changes** (gold earned/lost)
- **Kills, deaths, assists** (large discrete events)
- **Last hits** (farming reward for economy)
- **Health changes** (damage dealt/taken)
- **Building damage** (strategic objectives)
- **Zero-sum constraint**: Each team's mean reward is subtracted from the enemy's, ensuring the total is always zero. This prevents agents from finding positive-sum exploits.
- **"Team spirit" parameter**: Interpolates between individual reward and team-average reward. Started at 0 (selfish) and annealed to 1.0 (fully cooperative) during training.

### What Worked for AlphaStar (StarCraft II)

AlphaStar used a very sparse reward: **+1 for winning, -1 for losing, 0 otherwise**. But it was bootstrapped from human replay data through supervised learning before RL fine-tuning. The sparse reward worked because:
1. Supervised learning provided a strong initial policy
2. Self-play with a league of opponents provided the curriculum
3. The reward was simple and unambiguous

### Best Practices Applied to Your Project

**Current Reward Problems:**
1. Reward magnitudes are wildly unbalanced: -1000 (death) vs +0.5/tick (distance optimal). The per-tick rewards dominate episode returns for long episodes.
2. Per-tick rewards create high-frequency noise that obscures the signal from important combat events.
3. The -5/tick for low health incentivizes avoiding combat entirely (safest way to avoid low health).

**Recommended Reward Redesign (Potential-Based):**

```
SPARSE EVENT REWARDS (scale: [-1, +1]):
  Kill:            +1.0
  Death:           -1.0
  Assist:          +0.3

DENSE COMBAT REWARDS (scale: [-0.1, +0.1] per tick):
  Damage dealt:    +(damage / max_health) * 0.1
  Damage taken:    -(damage / max_health) * 0.05  (asymmetric: dealing > taking)
  Critical hit:    +0.02 bonus

POTENTIAL-BASED SHAPING (guaranteed to preserve optimal policy):
  Phi(s) = 0.3 * (my_health - enemy_health) / max_health
         + 0.1 * in_optimal_range(distance)    // 3-5 blocks
         + 0.05 * facing_enemy                  // dot product
  Shaping reward = gamma * Phi(s') - Phi(s)     // Only reward for IMPROVEMENT
```

**Key principles:**
1. All rewards normalized to [-1, +1] range to prevent gradient scale issues
2. Dense rewards should be 10-100x smaller than sparse event rewards
3. Use potential-based shaping (gamma * Phi(s') - Phi(s)) which mathematically guarantees the optimal policy is unchanged
4. The asymmetry (damage dealt > damage taken penalty) encourages aggression over passivity

---

## 4. Temporal Context: Frame Stacking vs. Recurrent Networks

### Current Approach: Frame Stacking (4 frames of 112 dims = 448 input)
- Pros: Simple, no hidden state management, deterministic
- Cons: Only sees last 4 ticks (~200ms), no long-term memory, 4x input size

### Recommended: GRU (Gated Recurrent Unit)
- Pros: Learns what to remember, compact hidden state, handles variable-length history
- Cons: Hidden state must be managed across episodes, adds training complexity

**Why GRU over LSTM:** GRU has 2 gates vs LSTM's 3, making it ~33% fewer parameters and faster inference. Research shows comparable performance to LSTM for most RL tasks, with faster convergence. At 20 ticks/sec, the computational savings matter.

**Why not Transformer (for temporal)?** While AlphaStar uses transformers for entity processing, it uses LSTM for temporal context. Full transformer temporal models (like GTrXL) are overkill for combat where relevant history is typically 2-5 seconds.

**Recommended GRU configuration:**
- Hidden size: 128 (matches current HIDDEN2_SIZE)
- Single layer (multi-layer GRU adds latency for minimal benefit in this domain)
- Reset hidden state at episode boundaries
- Input: Current frame (112 dims from state collector) + entity summary (64 dims from attention)

---

## 5. Minecraft AI Projects: Lessons Learned

### MineRL (Competition)
- Used behavioral cloning from human demonstrations as a starting point
- Pure RL from scratch failed to learn meaningful behavior within compute budgets
- Key lesson: **Start from demonstrations or pre-programmed heuristics, don't learn from zero**

### STEVE-1
- Instruction-following model built on Video Pretraining (VPT)
- Uses MineCLIP embeddings for goal conditioning
- Trained via self-supervised behavioral cloning + hindsight relabeling
- Key lesson: **Vision-based approaches are compute-intensive; structured state input (like your PvPStateCollector) is much more efficient**

### Voyager
- Uses GPT-4 as a planning engine with code generation for actions
- Not RL at all; relies on LLM world knowledge
- Key lesson: **Irrelevant to your approach but shows that structured action spaces beat raw pixel input**

### PracticeBotPvP (Minecraft Plugin)
- Uses rule-based + RL hybrid approach
- Scales difficulty from beginner to expert
- Key lesson: **The curriculum approach you already have is on the right track**

### Minecraft PVP Bot (GiaoShou66/GitHub)
- Deep Q-learning for PvP combat
- Selective reinforcement learning with exploration
- Key lesson: **Pure DQN struggles with the continuous nature of combat; actor-critic methods (PPO) work better**

---

## 6. Recommended Architecture

```
                    ENTITY ATTENTION MODULE
                    =====================
Per-entity features ─── Entity Encoder (Linear 16→64, ReLU)
 (up to 32 entities     │
  x 16 features each)   Multi-Head Self-Attention (2 heads, 64 dim)
                         │
                         Mean Pool ──→ entity_summary (64 dims)


                    MAIN POLICY NETWORK
                    ===================
[self_state(30) | entity_summary(64) | combat_ctx(22) | env(8) | sigils(12)]
                         │
                    136 total dims
                         │
                    Linear(136, 256) + ReLU
                         │
                    GRU(256, 128)  ←─── hidden_state from previous tick
                         │
                    ┌────┴────┐
                    │         │
             Policy Head   Value Head
          Linear(128,24)  Linear(128,1)
                    │
             Categorical sampling per action dimension
             (with action masking for conflicting moves)
```

**Total parameters:** ~150K (down from current 148K but with much more expressive architecture)
**Inference time:** ~0.5ms on CPU (well within 50ms tick budget)

---

## 7. Recommended Training Pipeline

### Phase 1: PPO + Entity Attention (Weeks 1-3)
**Goal:** Replace REINFORCE with PPO, add entity attention, verify convergence on simple scenarios.

1. **Implement PPO core** (clipped surrogate + GAE)
   - Train in Python with PyTorch first (rapid iteration)
   - Export trained model to TorchScript for Java inference via DJL
   - This decouples training from the game client

2. **Add entity attention module**
   - Replace single-enemy state (40 dims) with attention over all visible entities
   - Each entity gets 16 features: [alliance(1), health(1), armor(1), distance(3 relative xyz), velocity(3), facing(2), weapon(1), blocking(1), attack_cooldown(1), on_ground(1), is_target(1)]
   - Process through 2-layer self-attention with 2 heads

3. **Fix reward function**
   - Normalize all rewards to [-1, +1]
   - Implement potential-based shaping
   - Add zero-sum constraint (reward_dealt = -reward_taken for opponent)

4. **Verify convergence** on curriculum Stage 1a (stationary target)
   - Should converge in <100 episodes (30 minutes of play)
   - If not converging, debug reward signal and gradient flow

### Phase 2: GRU + Self-Play (Weeks 4-6)
**Goal:** Add temporal memory and train against itself.

1. **Add GRU layer** (replace frame stacking)
   - Single GRU layer with 128 hidden units
   - Reduces input size from 448 to 176 (entity attention + self state)
   - Train with truncated BPTT (32-step windows)

2. **Implement simple self-play**
   - Save checkpoint every 50 episodes
   - 80% of games: play against latest checkpoint
   - 20% of games: play against random historical checkpoint (prevents forgetting)

3. **Expand curriculum** through Stages 1b-1d automatically via self-play difficulty

### Phase 3: Scaling & Polish (Weeks 7+)
**Goal:** Multi-instance training, league play, deployment.

1. **Parallel training** with multiple game instances (APPO-style)
   - Each instance collects experience independently
   - Central trainer aggregates and updates
   - 4-8 instances on a single machine gives 4-8x speedup

2. **League training** (AlphaStar-style)
   - Maintain a league of past checkpoints with different playstyles
   - Train specialist agents (aggressive, defensive, ranged)
   - Main agent must beat all specialists

3. **Team spirit** for multi-agent cooperation
   - Start with individual rewards (team_spirit = 0)
   - Gradually increase to team_spirit = 0.5-0.8
   - Enables ally/enemy awareness through reward sharing

---

## 8. Key Hyperparameters That Matter Most

Listed in order of sensitivity (most impactful first):

| Parameter | Recommended | Range | Why It Matters |
|-----------|-------------|-------|----------------|
| **Learning rate** | 3e-4 | 1e-4 to 1e-3 | Too high = divergence, too low = no learning. Use linear decay to 0. |
| **GAE lambda** | 0.95 | 0.9 to 0.99 | Controls bias-variance tradeoff. 0.95 is robust default. |
| **Clip range (epsilon)** | 0.2 | 0.1 to 0.3 | The "safety rail" that prevents policy collapse. Lower = more stable, slower. |
| **Entropy coefficient** | 0.01 | 0.001 to 0.05 | Too high = random behavior. Too low = premature convergence. Decay from 0.05 to 0.005. |
| **Discount (gamma)** | 0.99 | 0.95 to 0.999 | Your current 0.9 is too low for combat (discounts future 10+ ticks away by 65%). Use 0.99. |
| **PPO epochs** | 4 | 3 to 10 | How many passes over the batch. Too many = overfitting to batch. |
| **Minibatch size** | 128 | 64 to 512 | Larger = more stable but slower per update. |
| **Batch size (steps)** | 2048 | 512 to 8192 | Total steps per training iteration. More = more stable. |
| **Value loss coeff** | 0.5 | 0.25 to 1.0 | Balances policy vs value gradient through shared layers. |
| **Max grad norm** | 0.5 | 0.1 to 1.0 | Global gradient clipping (not per-weight like current implementation). |

### Critical: Gamma = 0.99, Not 0.9

Your current gamma of 0.9 means that a reward 20 ticks in the future (1 second) is weighted as 0.9^20 = 0.12 -- only 12% of its true value. Combat requires planning 2-5 seconds ahead (40-100 ticks), where 0.9^100 = 0.00003. The agent literally cannot see consequences beyond ~1 second.

With gamma = 0.99: 0.99^100 = 0.37, meaning rewards 5 seconds in the future still have 37% weight. This is necessary for learning to set up combos, time abilities, and plan approach angles.

---

## 9. Common Pitfalls and How to Avoid Them

### Pitfall 1: Reward Hacking
**Problem:** Agent finds degenerate strategies that maximize reward without playing well (e.g., dealing damage then running away to maintain health advantage).
**Solution:** Zero-sum reward design (your gain = opponent's loss). Add small negative reward for distance from combat zone. Test against human players early.

### Pitfall 2: Catastrophic Forgetting in Self-Play
**Problem:** Agent learns to beat version N, forgets how to beat version N-5.
**Solution:** League training with historical checkpoints. Always play 20% of games against random past versions.

### Pitfall 3: Entropy Collapse
**Problem:** Entropy decays too fast, agent becomes deterministic and stops exploring.
**Solution:** Use PPO's entropy bonus with a slow decay schedule. Monitor entropy in training logs; if it drops below 0.1 * initial_entropy before episode 500, increase coefficient.

### Pitfall 4: Value Function Lag
**Problem:** Value function lags behind the rapidly changing policy, providing bad advantage estimates.
**Solution:** Use GAE (already recommended). Pre-train value function for a few epochs at the start of training before enabling policy updates. Use separate optimizer or lower learning rate for value function.

### Pitfall 5: Training in Java vs Python
**Problem:** Python ecosystem (PyTorch, Stable Baselines 3, WandB) is 10x more productive for RL research than Java/DJL.
**Solution:** Train in Python, deploy in Java. Use TorchScript to export trained models. DJL loads TorchScript models natively. This is how production game AI works (AlphaStar trained in Python/TensorFlow, deployed in C++).

### Pitfall 6: Debugging RL is Hard
**Problem:** RL fails silently -- the agent just doesn't learn, and it's unclear why.
**Solution:** Log EVERYTHING. Track these metrics per episode:
- Episode return (total reward)
- Episode length
- Policy entropy
- Value function explained variance (how well value predicts returns)
- KL divergence between old and new policy
- Clip fraction (how often PPO clip is active)
- Gradient norms
If clip fraction > 0.3, learning rate is too high. If explained variance < 0, value function is broken. If entropy drops to 0, agent is stuck.

---

## 10. Build Order (What to Build First vs Later)

### NOW (Highest Impact, Lowest Risk)
1. **Switch from REINFORCE to PPO with GAE** -- This alone will likely fix convergence
2. **Normalize rewards to [-1, +1] range** -- Prevents gradient explosion
3. **Increase gamma from 0.9 to 0.99** -- Lets agent see consequences beyond 1 second
4. **Increase batch size from 4 episodes to 2048+ steps** -- Reduces gradient variance

### SOON (High Impact, Moderate Effort)
5. **Entity attention module** -- Enables multi-agent awareness
6. **Train in Python, export to Java** -- 10x faster iteration cycle
7. **Potential-based reward shaping** -- Mathematically sound dense rewards

### LATER (Important but Depends on Earlier Steps)
8. **GRU for temporal context** -- Better than frame stacking
9. **Self-play training** -- Required for strong PvP
10. **Parallel training (multi-instance)** -- Speed up training wall-clock time

### MUCH LATER (Polish & Scale)
11. **League training** -- Diversity of playstyles
12. **Team spirit / cooperative rewards** -- Multi-agent team play
13. **Hierarchical actions** -- Strategic layer on top of tactical layer

---

## Sources

- [OpenAI Five Paper](https://cdn.openai.com/dota-2.pdf)
- [AlphaStar Architecture](https://deepwiki.com/google-deepmind/alphastar/3.3-standard-architecture)
- [AlphaStar DeepMind Blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- [37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- [PPO Stabilisation Tips](https://vegapit.com/article/modern-tale-of-deep-learning-ppo-stabilisation/)
- [Stable Baselines 3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [DJL (Deep Java Library)](https://github.com/deepjavalibrary/djl)
- [Attention in Multi-Agent RL (Survey)](https://www.sciencedirect.com/science/article/abs/pii/S0925231224007860)
- [AI-QMIX: Attention for Dynamic Multi-Agent RL](https://deepai.org/publication/ai-qmix-attention-and-imagination-for-dynamic-multi-agent-reinforcement-learning)
- [Curriculum Learning for RL Domains (JMLR)](https://jmlr.org/papers/volume21/20-212/20-212.pdf)
- [Reward Hacking in RL (Lilian Weng)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)
- [Potential-Based Reward Shaping](https://medium.com/@sophiezhao_2990/potential-based-reward-shaping-in-reinforcement-learning-05da05cfb84a)
- [Sparse Rewards in RL](https://medium.com/@m.k.daaboul/dealing-with-sparse-reward-environments-38c0489c844d)
- [Self-Play Survey (2024)](https://arxiv.org/abs/2408.01072)
- [Multi-Agent RL in Video Games](https://arxiv.org/pdf/2509.03682)
- [Minecraft PVP Bot](https://github.com/GiaoShou66/Minecraft-PVP-bot)
- [STEVE-1](https://github.com/Shalev-Lifshitz/STEVE-1)
- [Voyager](https://voyager.minedojo.org/)
- [Scale Flexibility in RTS Games](https://www.sciencedirect.com/science/article/abs/pii/S1875952124002118)
- [Netflix DJL for Real-Time Inference](https://aws.amazon.com/blogs/opensource/how-netflix-uses-deep-java-library-djl-for-distributed-deep-learning-inference-in-real-time/)
