#!/usr/bin/env python3
"""Fast PPO training with self-play using vectorized simulator.

Key improvements over train_ppo.py:
- Numpy-vectorized sim (all N envs run simultaneously)
- Self-play: agent trains against a frozen copy of itself
- Opponent updated every K rollouts for stable training
- Larger effective batches for stable gradients

Usage:
    python -m training.fast_train --steps 500000
    python -m training.fast_train --steps 5000000 --n-envs 512 --device cuda
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from .vec_sim import VecPvPSim
from .config import (
    OBS_DIM, NUM_ACTIONS, USED_BITS,
    GAP_EAT_TICKS, GAP_COOLDOWN_TICKS,
    POT_START_COUNT, POT_SPLASH_RADIUS, POT_HEAL_AMOUNT,
    PEARL_START_COUNT, PEARL_COOLDOWN_TICKS,
    ACT_BLOCK, ACT_THROW_POT, ACT_THROW_PEARL, ACT_ENGAGE, ACT_EAT_GAP,
    DEFAULT_ATTACK_REACH,
    GRU_HIDDEN_DIM,
)
from .models.network import CombatNetwork
from .rule_test import rule_charger, rule_charger_noisy, rule_charger_smart, init_noisy_charger, reset_noisy_charger


def parse_args():
    p = argparse.ArgumentParser(description="Fast PPO with self-play")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint-interval", type=int, default=100_000)
    p.add_argument("--log-interval", type=int, default=2)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--export", type=str, default=None)
    p.add_argument("--episode-length", type=int, default=1800)
    # Self-play
    p.add_argument("--opponent-update-interval", type=int, default=10,
                    help="Rollouts between opponent policy updates")
    p.add_argument("--self-play", action="store_true", default=True,
                    help="Use self-play (agent vs frozen copy)")
    p.add_argument("--no-self-play", dest="self_play", action="store_false",
                    help="Use rule-based opponent instead")
    p.add_argument("--style", type=str, default="combo",
                    choices=["combo", "wtap", "stap", "strafe", "crit"],
                    help="Combat style for technique-specific reward shaping")
    p.add_argument("--rule-opponent", action="store_true", default=True,
                    help="Use rule-based charger as opponent (default)")
    p.add_argument("--no-rule-opponent", dest="rule_opponent", action="store_false",
                    help="Use dual neural net training instead")
    p.add_argument("--noisy-rule-opponent", action="store_true", default=False,
                    help="Use randomized rule charger + hybrid rewards (human-like opponent)")
    p.add_argument("--smart-opponent", action="store_true", default=False,
                    help="Use smart charger that heals (forces model to also heal)")
    # PSRO (iterated training)
    p.add_argument("--frozen-counter", type=str, default=None,
                    help="PSRO: frozen counter checkpoint path (trains charger vs it)")
    p.add_argument("--frozen-charger", type=str, default=None,
                    help="PSRO: frozen charger checkpoint path (trains counter vs it)")
    p.add_argument("--attack-reach", type=float, default=3.0,
                    help="Engage auto-attack reach (default 3.0=max, lower=easier opponent)")
    p.add_argument("--combo-style", type=str, default="stap",
                    choices=["stap", "wtap", "none"],
                    help="Combo style: stap (S-tap), wtap (W-tap), none (raw NN)")
    p.add_argument("--gauntlet", action="store_true", default=False,
                    help="Gauntlet mode: killing opponent respawns fresh one, bot keeps HP")
    p.add_argument("--mixed-opponent", action="store_true", default=False,
                    help="Mixed: half envs vs rule charger, half vs neural self-play")
    p.add_argument("--population-dir", type=str, default=None,
                    help="Directory of frozen checkpoint .pt files for population-based training")
    p.add_argument("--asymmetric", action="store_true", default=False,
                    help="Asymmetric dual-net: A=aggressive rewards, B=defensive rewards")
    return p.parse_args()


class PolicyWrapper(nn.Module):
    """Wraps CombatNetwork for flat obs input."""

    def __init__(self, network: CombatNetwork):
        super().__init__()
        self.network = network
        self._action_mask = None

    def _get_action_mask(self, logits):
        if self._action_mask is None or self._action_mask.device != logits.device:
            mask = torch.zeros(logits.shape[-1], device=logits.device)
            for b in USED_BITS:
                mask[b] = 1.0
            self._action_mask = mask
        return self._action_mask

    def forward(self, flat_obs, hidden=None):
        obs_dict = self._split_obs(flat_obs)
        logits, value, hidden = self.network(obs_dict, hidden)
        # Clamp dead bits to -10 (near-zero prob, near-zero entropy)
        mask = self._get_action_mask(logits)
        logits = logits * mask + (-10.0) * (1.0 - mask)
        return logits, value, hidden

    def get_action_and_value(self, flat_obs, hidden=None, action=None):
        logits, value, hidden = self.forward(flat_obs, hidden)
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs=probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Entropy only over used bits (dead bits have ~0 entropy from clamping)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, value.squeeze(-1), hidden

    def get_value(self, flat_obs, hidden=None):
        _, value, hidden = self.forward(flat_obs, hidden)
        return value.squeeze(-1), hidden

    def sample_actions(self, flat_obs):
        """Fast action sampling (no grad, no value)."""
        with torch.no_grad():
            logits, _, _ = self.forward(flat_obs)
            probs = torch.sigmoid(logits)
            actions = torch.bernoulli(probs)
        return actions

    def _split_obs(self, flat_obs):
        batch = flat_obs.shape[0] if flat_obs.dim() > 1 else 1
        if flat_obs.dim() == 1:
            flat_obs = flat_obs.unsqueeze(0)
        return {
            "self_state": flat_obs[:, 0:38],
            "entity_features": flat_obs[:, 38:230].reshape(batch, 8, 24),
            "entity_mask": flat_obs[:, 230:238],
            "combat_ctx": flat_obs[:, 238:264],
            "sigil_state": flat_obs[:, 264:312],
            "env_state": flat_obs[:, 312:320],
        }


def rule_based_opponent(sim: VecPvPSim, rng: np.random.Generator) -> np.ndarray:
    """Simple rule-based opponent actions."""
    n = sim.n
    actions = np.zeros((n, NUM_ACTIONS), dtype=np.int8)

    dx = sim.ax - sim.bx
    dz = sim.az - sim.bz
    dist = np.sqrt(dx**2 + dz**2)

    # Always engage (face + auto-attack when in reach)
    actions[:, ACT_ENGAGE] = 1

    # Movement
    far = dist > 5.0
    mid = (dist > 2.5) & ~far
    close = ~far & ~mid

    # Forward when far/mid
    actions[far, 0] = 1  # forward
    actions[mid, 0] = (rng.random(n) < 0.6)[mid].astype(np.int8)

    # Sprint when far
    actions[far, 6] = 1  # sprint

    # Strafe when close
    strafe_dir = rng.choice([2, 3], n)  # left or right
    actions[close, strafe_dir[close]] = 1

    # Occasional jump
    actions[:, 4] = (rng.random(n) < 0.08).astype(np.int8)

    # Eat golden apple when health is low
    low_hp = sim.b_health < 10.0
    actions[low_hp, ACT_EAT_GAP] = (rng.random(n) < 0.3)[low_hp].astype(np.int8)

    return actions


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # ── Determine training mode ──
    train_charger_mode = args.frozen_counter is not None
    use_frozen_charger = args.frozen_charger is not None
    use_noisy_opp = args.noisy_rule_opponent and not use_frozen_charger and not train_charger_mode
    use_smart_opp = args.smart_opponent and not use_frozen_charger and not train_charger_mode
    use_mixed_opp = args.mixed_opponent and not use_frozen_charger and not train_charger_mode
    # Mixed mode: half envs rule charger, half frozen self-play copy (no dual-net needed)
    use_rule_opp = (args.rule_opponent or use_noisy_opp) and not use_frozen_charger and not train_charger_mode and not use_mixed_opp
    if use_mixed_opp:
        use_rule_opp = True  # treat as rule-opp for buffer purposes (only train counter)

    # Population-based training
    use_population = args.population_dir is not None and not use_frozen_charger and not train_charger_mode
    if use_population:
        use_rule_opp = True  # treat as rule-opp for buffer purposes (only train counter)

    # Asymmetric dual-net training
    use_asymmetric = args.asymmetric and not use_frozen_charger and not train_charger_mode and not use_population
    if use_asymmetric:
        use_rule_opp = False

    # Vectorized sim
    reward_mode = "hybrid" if use_noisy_opp else "default"
    print(f"Creating vectorized sim with {args.n_envs} parallel environments (reward_mode={reward_mode})...")
    sim = VecPvPSim(args.n_envs, episode_length=args.episode_length, seed=args.seed,
                    style=args.style, reward_mode=reward_mode,
                    attack_reach=args.attack_reach,
                    combo_style=args.combo_style,
                    gauntlet=args.gauntlet)

    if train_charger_mode:
        # PSRO: train charger (A) vs frozen counter (B)
        charger_net = CombatNetwork()
        charger = PolicyWrapper(charger_net).to(device)
        charger_opt = torch.optim.Adam(charger.parameters(), lr=args.lr, eps=1e-5)

        counter_net = CombatNetwork()
        counter = PolicyWrapper(counter_net).to(device)
        fc_ckpt = torch.load(args.frozen_counter, map_location=device, weights_only=False)
        sd = fc_ckpt.get("counter_state_dict", fc_ckpt.get("network_state_dict"))
        counter_net.load_state_dict(sd)
        counter.eval()
        for p in counter.parameters():
            p.requires_grad = False
        counter_opt = None
    else:
        # Counter network (player B) — always trained in counter mode
        counter_net = CombatNetwork()
        counter = PolicyWrapper(counter_net).to(device)
        counter_opt = torch.optim.Adam(counter.parameters(), lr=args.lr, eps=1e-5)

        if use_frozen_charger:
            charger_net = CombatNetwork()
            charger = PolicyWrapper(charger_net).to(device)
            fc_ckpt = torch.load(args.frozen_charger, map_location=device, weights_only=False)
            sd = fc_ckpt.get("charger_state_dict", fc_ckpt.get("counter_state_dict", fc_ckpt.get("network_state_dict")))
            charger_net.load_state_dict(sd)
            charger.eval()
            for p in charger.parameters():
                p.requires_grad = False
            charger_opt = None
        elif use_rule_opp:
            charger = charger_net = charger_opt = None
        else:
            charger_net = CombatNetwork()
            charger = PolicyWrapper(charger_net).to(device)
            charger_opt = torch.optim.Adam(charger.parameters(), lr=args.lr, eps=1e-5)

    total_steps = 0
    total_episodes = 0
    update_count = 0
    best_reward = -float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if train_charger_mode:
            sd = ckpt.get("charger_state_dict", ckpt.get("network_state_dict"))
            if sd:
                charger_net.load_state_dict(sd)
            if "optimizer_state_dict" in ckpt:
                charger_opt.load_state_dict(ckpt["optimizer_state_dict"])
        elif "counter_state_dict" in ckpt:
            counter_net.load_state_dict(ckpt["counter_state_dict"])
            if "counter_opt_state" in ckpt:
                counter_opt.load_state_dict(ckpt["counter_opt_state"])
            if charger_opt and "charger_state_dict" in ckpt:
                charger_net.load_state_dict(ckpt["charger_state_dict"])
                if "charger_opt_state" in ckpt:
                    charger_opt.load_state_dict(ckpt["charger_opt_state"])
        elif "network_state_dict" in ckpt:
            if train_charger_mode:
                charger_net.load_state_dict(ckpt["network_state_dict"])
            else:
                counter_net.load_state_dict(ckpt["network_state_dict"])
                if charger_opt:
                    charger_net.load_state_dict(ckpt["network_state_dict"])
        total_steps = ckpt.get("total_steps", 0)
        total_episodes = ckpt.get("total_episodes", 0)
        update_count = ckpt.get("update_count", 0)

    # Mixed opponent: frozen copy of counter used as self-play opponent for half envs
    mixed_opp_policy = None
    if use_mixed_opp:
        mixed_opp_net = CombatNetwork()
        mixed_opp_policy = PolicyWrapper(mixed_opp_net).to(device)
        mixed_opp_net.load_state_dict(counter_net.state_dict())
        mixed_opp_policy.eval()
        for p in mixed_opp_policy.parameters():
            p.requires_grad = False

    # Population-based: load pool of frozen opponents
    population_pool = []
    if use_population:
        pop_dir = Path(args.population_dir)
        for pt_file in sorted(pop_dir.glob("*.pt")):
            pop_net = CombatNetwork()
            pop_wrapper = PolicyWrapper(pop_net).to(device)
            pop_ckpt = torch.load(pt_file, map_location=device, weights_only=False)
            sd = pop_ckpt.get("network_state_dict", pop_ckpt.get("counter_state_dict"))
            if sd:
                pop_net.load_state_dict(sd)
            pop_wrapper.eval()
            for p_param in pop_wrapper.parameters():
                p_param.requires_grad = False
            population_pool.append((pt_file.stem, pop_wrapper))
        # Also add rule-based opponents to the pool
        population_pool.append(("rule_charger", rule_charger))
        population_pool.append(("rule_charger_smart", rule_charger_smart))
        print(f"Population pool loaded: {len(population_pool)} opponents")
        for name, _ in population_pool:
            print(f"  - {name}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Pre-allocate rollout buffers
    R = args.rollout_steps
    N = args.n_envs

    # Charger (player A) buffer — when training charger or dual-net
    dual_net = not use_rule_opp and not use_frozen_charger and not train_charger_mode
    if use_asymmetric:
        dual_net = True  # force dual-net for asymmetric training
    if train_charger_mode or dual_net:
        ch_obs = np.zeros((R, N, OBS_DIM), dtype=np.float32)
        ch_actions = np.zeros((R, N, NUM_ACTIONS), dtype=np.float32)
        ch_log_probs = np.zeros((R, N), dtype=np.float32)
        ch_rewards = np.zeros((R, N), dtype=np.float32)
        ch_values = np.zeros((R, N), dtype=np.float32)
        ch_dones = np.zeros((R, N), dtype=np.float32)
        ch_advantages = np.zeros((R, N), dtype=np.float32)
        ch_returns = np.zeros((R, N), dtype=np.float32)

    # Counter (player B) buffer — when training counter
    if not train_charger_mode:
        ct_obs = np.zeros((R, N, OBS_DIM), dtype=np.float32)
        ct_actions = np.zeros((R, N, NUM_ACTIONS), dtype=np.float32)
        ct_log_probs = np.zeros((R, N), dtype=np.float32)
        ct_rewards = np.zeros((R, N), dtype=np.float32)
        ct_values = np.zeros((R, N), dtype=np.float32)
        ct_dones = np.zeros((R, N), dtype=np.float32)
        ct_advantages = np.zeros((R, N), dtype=np.float32)
        ct_returns = np.zeros((R, N), dtype=np.float32)

    obs = sim.reset_all()
    if use_noisy_opp:
        init_noisy_charger(sim)
    ep_rewards = []
    ep_lengths = []
    ep_max_combos = []  # counter's max combo per episode
    ep_trades = []  # trades per episode
    current_ep_rewards = np.zeros(N)
    current_ep_lengths = np.zeros(N, dtype=np.int32)
    current_max_combo = np.zeros(N, dtype=np.int32)
    current_trades = np.zeros(N, dtype=np.int32)

    if train_charger_mode:
        opp_label = "frozen-counter"
        role_label = "charger"
    elif use_frozen_charger:
        opp_label = "frozen-charger"
        role_label = "counter"
    elif use_noisy_opp:
        opp_label = "noisy-charger"
        role_label = "hybrid-fighter"
    elif use_population:
        opp_label = f"population({len(population_pool)})"
        role_label = "counter"
    elif use_mixed_opp:
        opp_label = "mixed(rule+self-play)"
        role_label = "counter"
    elif use_asymmetric:
        opp_label = "neural-asymmetric"
        role_label = "dual(aggressive-A/defensive-B)"
    elif use_rule_opp:
        opp_label = "rule-charger"
        role_label = "counter"
    else:
        opp_label = "neural-charger"
        role_label = "counter"
    print(f"Training for {args.steps} steps ({role_label} vs {opp_label})")
    print(f"  rollout={R}, envs={N}, batch_per_update={R*N}")
    start_time = time.time()

    while total_steps < args.steps:
        # ── Collect rollout ──
        if charger is not None:
            charger.eval()
        counter.eval()

        # Population: assign random opponents to each env for this rollout
        if use_population and population_pool:
            pop_indices = np.random.randint(0, len(population_pool), size=N)
            pop_assignments = [population_pool[i] for i in pop_indices]

        for step in range(R):
            # ── Player A (charger) actions ──
            if train_charger_mode:
                a_obs_tensor = torch.from_numpy(obs).to(device)
                with torch.no_grad():
                    a_action, a_log_prob, _, a_value, _ = charger.get_action_and_value(a_obs_tensor)
                a_action_np = a_action.cpu().numpy().astype(np.int8)
            elif use_population:
                a_action_np = np.zeros((N, NUM_ACTIONS), dtype=np.int8)
                # Group envs by opponent type
                rule_envs = []
                neural_groups = {}  # wrapper id -> (wrapper, list of env indices)
                for i, (name, opp) in enumerate(pop_assignments):
                    if callable(opp) and not isinstance(opp, nn.Module):
                        rule_envs.append((i, opp))
                    else:
                        if id(opp) not in neural_groups:
                            neural_groups[id(opp)] = (opp, [])
                        neural_groups[id(opp)][1].append(i)
                # Rule opponents
                for i, opp_fn in rule_envs:
                    a_action_np[i] = opp_fn(sim)[i]
                # Neural opponents (batch per model)
                for _, (opp_model, env_ids) in neural_groups.items():
                    env_ids = np.array(env_ids)
                    opp_obs = torch.from_numpy(obs[env_ids]).to(device)
                    with torch.no_grad():
                        opp_acts = opp_model.sample_actions(opp_obs).cpu().numpy().astype(np.int8)
                    a_action_np[env_ids] = opp_acts
            elif use_rule_opp:
                a_action_np = rule_charger_noisy(sim) if use_noisy_opp else (rule_charger_smart(sim) if use_smart_opp else rule_charger(sim))
            elif use_mixed_opp:
                # Mixed: first half envs = rule charger, second half = frozen self-play
                a_action_np = rule_charger(sim)  # start with rule actions for all
                neural_mask = np.arange(N) >= (N // 2)
                if neural_mask.any():
                    a_obs_tensor = torch.from_numpy(obs).to(device)
                    with torch.no_grad():
                        neural_actions = mixed_opp_policy.sample_actions(a_obs_tensor)
                    a_action_np[neural_mask] = neural_actions.cpu().numpy().astype(np.int8)[neural_mask]
            elif use_frozen_charger:
                a_obs_tensor = torch.from_numpy(obs).to(device)
                with torch.no_grad():
                    a_action_np = charger.sample_actions(a_obs_tensor).cpu().numpy().astype(np.int8)
            else:
                a_obs_tensor = torch.from_numpy(obs).to(device)
                with torch.no_grad():
                    a_action, a_log_prob, _, a_value, _ = charger.get_action_and_value(a_obs_tensor)
                a_action_np = a_action.cpu().numpy().astype(np.int8)

            # ── Player B (counter) actions ──
            b_obs_np = sim.get_b_obs()
            if train_charger_mode:
                b_obs_tensor = torch.from_numpy(b_obs_np).to(device)
                with torch.no_grad():
                    b_action_np = counter.sample_actions(b_obs_tensor).cpu().numpy().astype(np.int8)
            else:
                b_obs_tensor = torch.from_numpy(b_obs_np).to(device)
                with torch.no_grad():
                    b_action, b_log_prob, _, b_value, _ = counter.get_action_and_value(b_obs_tensor)
                b_action_np = b_action.cpu().numpy().astype(np.int8)

            # Step sim — returns a_rewards in "rewards", b_rewards in infos
            next_obs, a_rewards, dones, infos = sim.step(a_action_np, b_action_np)
            b_rewards = infos["b_rewards"]

            # ── Store rollouts ──
            if train_charger_mode:
                ch_obs[step] = obs
                ch_actions[step] = a_action_np.astype(np.float32)
                ch_log_probs[step] = a_log_prob.cpu().numpy()
                ch_rewards[step] = a_rewards
                ch_values[step] = a_value.cpu().numpy()
                ch_dones[step] = dones.astype(np.float32)
            else:
                if dual_net:
                    ch_obs[step] = obs
                    ch_actions[step] = a_action_np.astype(np.float32)
                    ch_log_probs[step] = a_log_prob.cpu().numpy()
                    ch_rewards[step] = a_rewards
                    ch_values[step] = a_value.cpu().numpy()
                    ch_dones[step] = dones.astype(np.float32)
                ct_obs[step] = b_obs_np
                ct_actions[step] = b_action_np.astype(np.float32)
                ct_log_probs[step] = b_log_prob.cpu().numpy()
                ct_rewards[step] = b_rewards
                ct_values[step] = b_value.cpu().numpy()
                ct_dones[step] = dones.astype(np.float32)

            # ── Track episode stats ──
            if train_charger_mode:
                current_ep_rewards += a_rewards
                current_max_combo = np.maximum(current_max_combo, sim.a_combo_streak)
            else:
                current_ep_rewards += b_rewards
                current_max_combo = np.maximum(current_max_combo, sim.b_combo_streak)
            current_ep_lengths += 1
            both_hit = infos["a_sprint_hits"] & infos["b_sprint_hits"]
            current_trades += both_hit.astype(np.int32)

            done_indices = []
            for i in range(N):
                if dones[i]:
                    ep_rewards.append(float(current_ep_rewards[i]))
                    ep_lengths.append(int(current_ep_lengths[i]))
                    ep_max_combos.append(int(current_max_combo[i]))
                    ep_trades.append(int(current_trades[i]))
                    current_ep_rewards[i] = 0.0
                    current_ep_lengths[i] = 0
                    current_max_combo[i] = 0
                    current_trades[i] = 0
                    total_episodes += 1
                    done_indices.append(i)
            if use_noisy_opp and done_indices:
                reset_noisy_charger(sim, np.array(done_indices))

            obs = next_obs
            total_steps += N

        # GAE for both networks
        def compute_gae(buf_r, buf_v, buf_d, policy_net, obs_np):
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_np).to(device)
                last_val, _ = policy_net.get_value(obs_t)
                last_val = last_val.cpu().numpy()
            adv = np.zeros((R, N), dtype=np.float32)
            ret = np.zeros((R, N), dtype=np.float32)
            last_gae = np.zeros(N, dtype=np.float32)
            for t in reversed(range(R)):
                next_val = last_val if t == R - 1 else buf_v[t + 1]
                non_terminal = 1.0 - buf_d[t]
                delta = buf_r[t] + args.gamma * next_val * non_terminal - buf_v[t]
                last_gae = delta + args.gamma * args.gae_lambda * non_terminal * last_gae
                adv[t] = last_gae
            ret[:] = adv + buf_v
            return adv, ret

        if train_charger_mode:
            ch_advantages, ch_returns = compute_gae(ch_rewards, ch_values, ch_dones, charger, obs)
        else:
            if dual_net:
                ch_advantages, ch_returns = compute_gae(ch_rewards, ch_values, ch_dones, charger, obs)
            b_obs_final = sim.get_b_obs()
            ct_advantages, ct_returns = compute_gae(ct_rewards, ct_values, ct_dones, counter, b_obs_final)

        # ── PPO update helper ──
        def ppo_update(policy_net, opt, buf_o, buf_a, buf_lp, buf_adv, buf_ret):
            policy_net.train()
            n_samples = R * N
            flat_obs = buf_o.reshape(n_samples, OBS_DIM)
            flat_actions = buf_a.reshape(n_samples, NUM_ACTIONS)
            flat_log_probs = buf_lp.reshape(n_samples)
            flat_advantages = buf_adv.reshape(n_samples)
            flat_returns = buf_ret.reshape(n_samples)
            indices = np.arange(n_samples)

            all_pg, all_vf, all_ent, all_kl = [], [], [], []

            for epoch in range(args.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, n_samples, args.minibatch_size):
                    end = min(start + args.minibatch_size, n_samples)
                    mb = indices[start:end]

                    mb_obs = torch.from_numpy(flat_obs[mb]).to(device)
                    mb_act = torch.from_numpy(flat_actions[mb]).to(device)
                    mb_old_lp = torch.from_numpy(flat_log_probs[mb]).to(device)
                    mb_adv = torch.from_numpy(flat_advantages[mb]).to(device)
                    mb_ret = torch.from_numpy(flat_returns[mb]).to(device)

                    if len(mb_adv) > 1:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    _, new_lp, entropy, new_val, _ = policy_net.get_action_and_value(
                        mb_obs, action=mb_act
                    )

                    log_ratio = new_lp - mb_old_lp
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - args.clip_range, 1 + args.clip_range) * mb_adv
                    pg_loss = -torch.min(surr1, surr2).mean()
                    vf_loss = F.mse_loss(new_val, mb_ret)
                    ent_loss = -entropy.mean()

                    loss = pg_loss + args.value_coef * vf_loss + args.entropy_coef * ent_loss

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), args.max_grad_norm)
                    opt.step()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        all_pg.append(pg_loss.item())
                        all_vf.append(vf_loss.item())
                        all_ent.append(-ent_loss.item())
                        all_kl.append(approx_kl)

                if all_kl and np.mean(all_kl[-max(1, n_samples // args.minibatch_size):]) > 0.02:
                    break

            return all_pg, all_vf, all_ent, all_kl

        # Update networks
        if train_charger_mode:
            all_pg, all_vf, all_ent, all_kl = ppo_update(
                charger, charger_opt,
                ch_obs, ch_actions, ch_log_probs,
                ch_advantages, ch_returns)
        else:
            if dual_net:
                ppo_update(charger, charger_opt,
                           ch_obs, ch_actions, ch_log_probs,
                           ch_advantages, ch_returns)
            all_pg, all_vf, all_ent, all_kl = ppo_update(
                counter, counter_opt,
                ct_obs, ct_actions, ct_log_probs,
                ct_advantages, ct_returns)

        update_count += 1

        # ── Update mixed opponent frozen copy periodically ──
        if use_mixed_opp and update_count % args.opponent_update_interval == 0:
            mixed_opp_net.load_state_dict(counter_net.state_dict())

        # ── Logging ──
        if update_count % args.log_interval == 0 and ep_rewards:
            elapsed = time.time() - start_time
            sps = total_steps / max(elapsed, 1)
            recent = ep_rewards[-200:]
            mean_r = np.mean(recent)
            mean_l = np.mean(ep_lengths[-200:]) if ep_lengths else 0
            win_rate = sum(1 for r in recent if r > 0) / len(recent)
            mean_combo = np.mean(ep_max_combos[-200:]) if ep_max_combos else 0
            mean_trades = np.mean(ep_trades[-200:]) if ep_trades else 0

            print(f"step={total_steps:>8d} | ep={total_episodes:>6d} | "
                  f"r={mean_r:>7.3f} | len={mean_l:>6.1f} | "
                  f"win={win_rate:>5.1%} | "
                  f"combo={mean_combo:>4.1f} | trades={mean_trades:>4.1f} | "
                  f"pg={np.mean(all_pg):.4f} | vf={np.mean(all_vf):.4f} | "
                  f"ent={np.mean(all_ent):.3f} | kl={np.mean(all_kl):.4f} | "
                  f"sps={sps:.0f}")

            # Reward component breakdown (hybrid mode only)
            if use_noisy_opp:
                rb = sim.get_reward_breakdown(reset=True)
                total_rb = sum(abs(v) for v in rb.values())
                if total_rb > 0:
                    parts = []
                    for k, v in sorted(rb.items(), key=lambda x: -abs(x[1])):
                        pct = abs(v) / total_rb * 100
                        if pct >= 1.0:  # only show components >= 1%
                            parts.append(f"{k}={v:+.0f}({pct:.0f}%)")
                    print(f"  rewards: {' | '.join(parts)}")

            if mean_r > best_reward:
                best_reward = mean_r

        # ── Fight replay (every 10 updates) ──
        if update_count % 10 == 0 and ep_rewards:
            if train_charger_mode:
                charger.eval()
                _replay_fight(charger, device, fight_num=update_count,
                              opponent=counter, train_role="charger")
            else:
                counter.eval()
                if use_population and population_pool:
                    # Pick a random opponent from the pool for replay
                    replay_name, replay_opp = population_pool[np.random.randint(len(population_pool))]
                    print(f"  Replay opponent: {replay_name}")
                    _replay_fight(counter, device, fight_num=update_count,
                                  opponent=replay_opp)
                elif use_rule_opp:
                    opp_fn = rule_charger_noisy if use_noisy_opp else (rule_charger_smart if use_smart_opp else rule_charger)
                    _replay_fight(counter, device, fight_num=update_count,
                                  opponent=opp_fn)
                else:
                    charger.eval()
                    _replay_fight(counter, device, fight_num=update_count,
                                  opponent=charger)

        # ── Checkpoint ──
        if total_steps % args.checkpoint_interval < R * N:
            path = ckpt_dir / f"fast_{total_steps}.pt"
            if train_charger_mode:
                _save_charger(charger, charger_opt, total_steps, total_episodes, update_count, path)
                _save_charger(charger, charger_opt, total_steps, total_episodes, update_count,
                              ckpt_dir / "fast_latest.pt")
            elif use_rule_opp or use_frozen_charger:
                _save(counter, counter_opt, total_steps, total_episodes, update_count, path)
                _save(counter, counter_opt, total_steps, total_episodes, update_count,
                      ckpt_dir / "fast_latest.pt")
            else:
                _save_dual(charger, charger_opt, counter, counter_opt,
                           total_steps, total_episodes, update_count, path)
                _save_dual(charger, charger_opt, counter, counter_opt,
                           total_steps, total_episodes, update_count,
                           ckpt_dir / "fast_latest.pt")

    # Final save
    final_path = ckpt_dir / "fast_final.pt"
    if train_charger_mode:
        _save_charger(charger, charger_opt, total_steps, total_episodes, update_count, final_path)
    elif use_rule_opp or use_frozen_charger:
        _save(counter, counter_opt, total_steps, total_episodes, update_count, final_path)
    else:
        _save_dual(charger, charger_opt, counter, counter_opt,
                   total_steps, total_episodes, update_count, final_path)
    print(f"\nTraining complete. {total_steps} steps, {total_episodes} episodes.")
    print(f"Best mean reward: {best_reward:.3f}")

    if args.export:
        _export(charger if train_charger_mode else counter, args.export, device)


def _save(wrapper, optimizer, steps, episodes, updates, path):
    torch.save({
        "network_state_dict": wrapper.network.state_dict(),
        "counter_state_dict": wrapper.network.state_dict(),  # for resume compat
        "counter_opt_state": optimizer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_steps": steps,
        "total_episodes": episodes,
        "update_count": updates,
    }, path)


def _save_charger(wrapper, optimizer, steps, episodes, updates, path):
    torch.save({
        "charger_state_dict": wrapper.network.state_dict(),
        "network_state_dict": wrapper.network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_steps": steps,
        "total_episodes": episodes,
        "update_count": updates,
    }, path)


def _save_dual(charger, charger_opt, counter, counter_opt, steps, episodes, updates, path):
    torch.save({
        "charger_state_dict": charger.network.state_dict(),
        "charger_opt_state": charger_opt.state_dict(),
        "counter_state_dict": counter.network.state_dict(),
        "counter_opt_state": counter_opt.state_dict(),
        "network_state_dict": counter.network.state_dict(),  # legacy compat for export
        "total_steps": steps,
        "total_episodes": episodes,
        "update_count": updates,
    }, path)


def _replay_fight(policy, device, fight_num=0, opponent=None, train_role="counter"):
    """Run a single 1v1 fight.
    train_role="counter": policy=player B (counter), opponent=player A.
    train_role="charger": policy=player A (charger), opponent=player B.
    opponent can be a PolicyWrapper (neural) or a callable rule function."""
    sim = VecPvPSim(1, episode_length=600, seed=int(time.time()) + fight_num)
    obs = sim.reset_all()
    use_rule_opp = callable(opponent) and not isinstance(opponent, nn.Module)

    a_max_combo = 0
    b_max_combo = 0
    a_total_hits = 0
    b_total_hits = 0
    a_sprint_hits_total = 0
    b_sprint_hits_total = 0
    a_trades = 0
    a_gapples_used = 0
    b_gapples_used = 0
    a_whiffs = 0
    b_whiffs = 0
    a_pots_used = 0
    b_pots_used = 0
    a_pearls_used = 0
    b_pearls_used = 0
    a_enemy_splashed = 0
    b_enemy_splashed = 0
    a_hit_dists = []
    b_hit_dists = []
    a_hit_airborne = []  # was target airborne when hit?
    b_hit_airborne = []

    print(f"\n{'='*80}")
    print(f"  FIGHT REPLAY #{fight_num}")
    print(f"{'='*80}")
    print(f"  A pos=(0.0, 0.0)  B pos=({sim.bx[0]:.1f}, {sim.bz[0]:.1f})  dist={sim._prev_dist[0]:.1f}")
    print(f"  Loadout: armor={sim.armor} sharp={sim.sharpness}  dmg={sim.base_dmg_after_armor:.1f} crit={sim.crit_dmg_after_armor:.1f}  blocked={sim.base_dmg_blocked:.1f}/{sim.crit_dmg_blocked:.1f}")
    print(f"  Gapples: {sim.a_gapple_count[0]} each  |  eat_time={GAP_EAT_TICKS}t  cd={GAP_COOLDOWN_TICKS}t")
    print(f"  Pots: {sim.a_pot_count[0]} each (heal={POT_HEAL_AMOUNT})  |  Pearls: {sim.a_pearl_count[0]} each (cd={PEARL_COOLDOWN_TICKS}t)")
    print(f"{'='*80}")

    for tick in range(600):
        # Snapshot EVERYTHING before step
        pre = {
            "a_x": sim.ax[0], "a_z": sim.az[0], "a_y": sim.ay[0],
            "a_vx": sim.avx[0], "a_vz": sim.avz[0], "a_vy": sim.avy[0],
            "a_yaw": sim.ayaw[0], "a_ground": sim.a_on_ground[0],
            "a_sprint": sim.a_sprinting[0], "a_hp": sim.a_health[0],
            "a_absorb": sim.a_absorption[0], "a_iframes": sim.a_hurt_time[0],
            "a_fall": sim.a_fall_dist[0], "a_eating": sim.a_eating_ticks[0],
            "a_eat_cd": sim.a_eat_cooldown[0], "a_gaps": sim.a_gapple_count[0],
            "a_regen": sim.a_regen_ticks[0], "a_combo": sim.a_combo_streak[0],
            "a_pots": sim.a_pot_count[0], "a_pot_lock": sim.a_pot_lockout[0],
            "a_pearls": sim.a_pearl_count[0], "a_pearl_cd": sim.a_pearl_cooldown[0],
            "b_x": sim.bx[0], "b_z": sim.bz[0], "b_y": sim.by[0],
            "b_vx": sim.bvx[0], "b_vz": sim.bvz[0], "b_vy": sim.bvy[0],
            "b_yaw": sim.byaw[0], "b_ground": sim.b_on_ground[0],
            "b_sprint": sim.b_sprinting[0], "b_hp": sim.b_health[0],
            "b_absorb": sim.b_absorption[0], "b_iframes": sim.b_hurt_time[0],
            "b_fall": sim.b_fall_dist[0], "b_eating": sim.b_eating_ticks[0],
            "b_eat_cd": sim.b_eat_cooldown[0], "b_gaps": sim.b_gapple_count[0],
            "b_regen": sim.b_regen_ticks[0], "b_combo": sim.b_combo_streak[0],
            "b_pots": sim.b_pot_count[0], "b_pot_lock": sim.b_pot_lockout[0],
            "b_pearls": sim.b_pearl_count[0], "b_pearl_cd": sim.b_pearl_cooldown[0],
        }

        # Get actions
        if train_role == "charger":
            # policy = player A (charger), opponent = player B (counter)
            obs_t = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                a_act = policy.sample_actions(obs_t).cpu().numpy().astype(np.int8)
            b_obs = sim.get_b_obs()
            b_obs_t = torch.from_numpy(b_obs).to(device)
            with torch.no_grad():
                b_act = opponent.sample_actions(b_obs_t).cpu().numpy().astype(np.int8)
        else:
            # policy = player B (counter), opponent = player A
            if use_rule_opp:
                a_act = opponent(sim)
            else:
                obs_t = torch.from_numpy(obs).to(device)
                opp = opponent if opponent is not None else policy
                with torch.no_grad():
                    a_act = opp.sample_actions(obs_t).cpu().numpy().astype(np.int8)
            b_obs = sim.get_b_obs()
            b_obs_t = torch.from_numpy(b_obs).to(device)
            with torch.no_grad():
                b_act = policy.sample_actions(b_obs_t).cpu().numpy().astype(np.int8)

        # Raw action bits (before sim forces sprint/cancels)
        a_raw = {
            "W": bool(a_act[0,0]), "S": bool(a_act[0,1]),
            "A": bool(a_act[0,2]), "D": bool(a_act[0,3]),
            "jump": bool(a_act[0,4]), "atk": bool(a_act[0,7]),
            "block": bool(a_act[0,8]), "eat": bool(a_act[0,9]),
            "pot": bool(a_act[0,10]), "pearl": bool(a_act[0,11]),
            "face": bool(a_act[0,26]),
        }
        b_raw = {
            "W": bool(b_act[0,0]), "S": bool(b_act[0,1]),
            "A": bool(b_act[0,2]), "D": bool(b_act[0,3]),
            "jump": bool(b_act[0,4]), "atk": bool(b_act[0,7]),
            "block": bool(b_act[0,8]), "eat": bool(b_act[0,9]),
            "pot": bool(b_act[0,10]), "pearl": bool(b_act[0,11]),
            "face": bool(b_act[0,26]),
        }

        obs, rewards, dones, infos = sim.step(a_act, b_act)

        a_dmg = infos["a_dmg_dealt"][0]
        b_dmg = infos["b_dmg_dealt"][0]
        a_sh = infos["a_sprint_hits"][0]
        b_sh = infos["b_sprint_hits"][0]
        a_hdist = infos["a_hit_dist"][0]
        b_hdist = infos["b_hit_dist"][0]
        reward = rewards[0]

        # Post-step state
        dist = np.sqrt((sim.ax[0]-sim.bx[0])**2 + (sim.az[0]-sim.bz[0])**2)
        a_speed = np.sqrt(sim.avx[0]**2 + sim.avz[0]**2)
        b_speed = np.sqrt(sim.bvx[0]**2 + sim.bvz[0]**2)

        # Track stats
        if a_dmg > 0:
            a_total_hits += 1
            a_hit_dists.append(a_hdist)
            a_hit_airborne.append(not pre["b_ground"])
        if b_dmg > 0:
            b_total_hits += 1
            b_hit_dists.append(b_hdist)
            b_hit_airborne.append(not pre["a_ground"])
        if a_sh: a_sprint_hits_total += 1
        if b_sh: b_sprint_hits_total += 1
        if a_dmg > 0 and b_dmg > 0: a_trades += 1
        a_max_combo = max(a_max_combo, sim.a_combo_streak[0])
        b_max_combo = max(b_max_combo, sim.b_combo_streak[0])
        if a_raw["atk"] and dist > DEFAULT_ATTACK_REACH: a_whiffs += 1
        if b_raw["atk"] and dist > DEFAULT_ATTACK_REACH: b_whiffs += 1
        if pre["a_gaps"] > sim.a_gapple_count[0]: a_gapples_used += 1
        if pre["b_gaps"] > sim.b_gapple_count[0]: b_gapples_used += 1
        a_pot_used = pre["a_pots"] > sim.a_pot_count[0]
        b_pot_used = pre["b_pots"] > sim.b_pot_count[0]
        a_pearl_used = pre["a_pearls"] > sim.a_pearl_count[0]
        b_pearl_used = pre["b_pearls"] > sim.b_pearl_count[0]
        if a_pot_used: a_pots_used += 1
        if b_pot_used: b_pots_used += 1
        if a_pearl_used: a_pearls_used += 1
        if b_pearl_used: b_pearls_used += 1
        if infos.get("a_pot_splash_enemy", np.zeros(1))[0]: a_enemy_splashed += 1
        if infos.get("b_pot_splash_enemy", np.zeros(1))[0]: b_enemy_splashed += 1

        # Determine what's interesting this tick
        has_combat = a_dmg > 0 or b_dmg > 0
        has_eat_start = (pre["a_eating"] == 0 and sim.a_eating_ticks[0] > 0) or \
                        (pre["b_eating"] == 0 and sim.b_eating_ticks[0] > 0)
        has_eat_finish = (pre["a_gaps"] > sim.a_gapple_count[0]) or \
                         (pre["b_gaps"] > sim.b_gapple_count[0])
        has_pot_pearl = a_pot_used or b_pot_used or a_pearl_used or b_pearl_used
        is_periodic = tick % 50 == 0

        if has_combat or has_eat_start or has_eat_finish or has_pot_pearl or is_periodic:
            # Build compact action strings
            def fmt_keys(raw):
                keys = []
                if raw["W"]: keys.append("W")
                if raw["S"]: keys.append("S")
                if raw["A"]: keys.append("A")
                if raw["D"]: keys.append("D")
                if raw["jump"]: keys.append("J")
                if raw["atk"]: keys.append("ATK")
                if raw["block"]: keys.append("BLK")
                if raw["eat"]: keys.append("EAT")
                if raw["pot"]: keys.append("POT")
                if raw["pearl"]: keys.append("PRL")
                if not raw["face"]: keys.append("!FACE")
                return "+".join(keys) if keys else "---"

            # Build state flags
            def fmt_state(prefix, sim_obj, pre_dict):
                flags = []
                if sim_obj.a_sprinting[0] if prefix == "a" else sim_obj.b_sprinting[0]:
                    flags.append("SPRINT")
                blocking = sim_obj.a_blocking[0] if prefix == "a" else sim_obj.b_blocking[0]
                if blocking: flags.append("BLOCK")
                ground = sim_obj.a_on_ground[0] if prefix == "a" else sim_obj.b_on_ground[0]
                if not ground: flags.append("AIR")
                eating = sim_obj.a_eating_ticks[0] if prefix == "a" else sim_obj.b_eating_ticks[0]
                if eating > 0: flags.append(f"EATING({eating})")
                regen = sim_obj.a_regen_ticks[0] if prefix == "a" else sim_obj.b_regen_ticks[0]
                if regen > 0: flags.append(f"REGEN")
                iframes = sim_obj.a_hurt_time[0] if prefix == "a" else sim_obj.b_hurt_time[0]
                if iframes > 0: flags.append(f"i={iframes}")
                absorb = sim_obj.a_absorption[0] if prefix == "a" else sim_obj.b_absorption[0]
                if absorb > 0: flags.append(f"abs={absorb:.0f}")
                return " ".join(flags) if flags else ""

            # Combat events
            events = []
            if a_dmg > 0:
                crit = a_dmg > sim.base_dmg_after_armor * 1.1
                blocked = a_dmg < sim.base_dmg_after_armor * 0.9
                htype = "CRIT" if crit else ("S-HIT" if a_sh else "hit")
                bstr = " (BLOCKED)" if blocked else ""
                combo = sim.a_combo_streak[0]
                cstr = f" COMBO x{combo}" if combo > 1 else ""
                tair = " tAIR" if not pre["b_ground"] else ""
                events.append(f"A->{htype} {a_dmg:.1f}@{a_hdist:.2f}{tair}{bstr}{cstr}")
            if b_dmg > 0:
                crit = b_dmg > sim.base_dmg_after_armor * 1.1
                blocked = b_dmg < sim.base_dmg_after_armor * 0.9
                htype = "CRIT" if crit else ("S-HIT" if b_sh else "hit")
                bstr = " (BLOCKED)" if blocked else ""
                combo = sim.b_combo_streak[0]
                cstr = f" COMBO x{combo}" if combo > 1 else ""
                tair = " tAIR" if not pre["a_ground"] else ""
                events.append(f"B->{htype} {b_dmg:.1f}@{b_hdist:.2f}{tair}{bstr}{cstr}")
            if a_dmg > 0 and b_dmg > 0:
                events.append("TRADE!")
            if has_eat_start:
                if pre["a_eating"] == 0 and sim.a_eating_ticks[0] > 0:
                    events.append(f"A starts eating (hp={sim.a_health[0]:.0f})")
                if pre["b_eating"] == 0 and sim.b_eating_ticks[0] > 0:
                    events.append(f"B starts eating (hp={sim.b_health[0]:.0f})")
            if has_eat_finish:
                if pre["a_gaps"] > sim.a_gapple_count[0]:
                    events.append(f"A finishes gapple! +abs")
                if pre["b_gaps"] > sim.b_gapple_count[0]:
                    events.append(f"B finishes gapple! +abs")
            # Pot/pearl events
            if a_pot_used:
                splash = " SPLASHED ENEMY!" if infos.get("a_pot_splash_enemy", np.zeros(1))[0] else ""
                events.append(f"A HEAL POT (hp={sim.a_health[0]:.0f}, pots={sim.a_pot_count[0]}){splash}")
            if b_pot_used:
                splash = " SPLASHED ENEMY!" if infos.get("b_pot_splash_enemy", np.zeros(1))[0] else ""
                events.append(f"B HEAL POT (hp={sim.b_health[0]:.0f}, pots={sim.b_pot_count[0]}){splash}")
            if a_pearl_used:
                events.append(f"A PEARL! (hp={sim.a_health[0]:.0f}, pearls={sim.a_pearl_count[0]})")
            if b_pearl_used:
                events.append(f"B PEARL! (hp={sim.b_health[0]:.0f}, pearls={sim.b_pearl_count[0]})")

            a_st = fmt_state("a", sim, pre)
            b_st = fmt_state("b", sim, pre)

            print(f"  t={tick:>3d} | "
                  f"A {fmt_keys(a_raw):>12s} hp={sim.a_health[0]:>5.1f} spd={a_speed:.2f} yaw={sim.ayaw[0]:>6.1f} {a_st} | "
                  f"B {fmt_keys(b_raw):>12s} hp={sim.b_health[0]:>5.1f} spd={b_speed:.2f} yaw={sim.byaw[0]:>6.1f} {b_st} | "
                  f"d={dist:>4.1f} r={reward:>+5.1f} | "
                  f"{'  '.join(events)}")

        if dones[0]:
            winner = "A" if sim.b_health[0] <= 0 else ("B" if sim.a_health[0] <= 0 else "DRAW")
            print(f"\n  >>> {winner} WINS at tick {tick}")
            break
    else:
        print(f"\n  >>> TIMEOUT at tick 600")

    # Summary
    print(f"  -- FIGHT SUMMARY --")
    print(f"  A: hp={sim.a_health[0]:.1f} hits={a_total_hits}({a_sprint_hits_total} sprint) "
          f"max_combo={a_max_combo} whiffs={a_whiffs} gapples={a_gapples_used} "
          f"pots={a_pots_used}(splashed={a_enemy_splashed}) pearls={a_pearls_used}")
    print(f"  B: hp={sim.b_health[0]:.1f} hits={b_total_hits}({b_sprint_hits_total} sprint) "
          f"max_combo={b_max_combo} whiffs={b_whiffs} gapples={b_gapples_used} "
          f"pots={b_pots_used}(splashed={b_enemy_splashed}) pearls={b_pearls_used}")
    print(f"  Trades: {a_trades}")
    # Reach analysis
    if a_hit_dists:
        a_air_dists = [d for d, air in zip(a_hit_dists, a_hit_airborne) if air]
        a_gnd_dists = [d for d, air in zip(a_hit_dists, a_hit_airborne) if not air]
        print(f"  A reach: avg={np.mean(a_hit_dists):.2f} min={np.min(a_hit_dists):.2f} max={np.max(a_hit_dists):.2f}"
              f"  |  grounded={len(a_gnd_dists)}(avg={np.mean(a_gnd_dists):.2f})" if a_gnd_dists else ""
              f"  airborne={len(a_air_dists)}(avg={np.mean(a_air_dists):.2f})" if a_air_dists else "")
    if b_hit_dists:
        b_air_dists = [d for d, air in zip(b_hit_dists, b_hit_airborne) if air]
        b_gnd_dists = [d for d, air in zip(b_hit_dists, b_hit_airborne) if not air]
        print(f"  B reach: avg={np.mean(b_hit_dists):.2f} min={np.min(b_hit_dists):.2f} max={np.max(b_hit_dists):.2f}"
              f"  |  grounded={len(b_gnd_dists)}(avg={np.mean(b_gnd_dists):.2f})" if b_gnd_dists else ""
              f"  airborne={len(b_air_dists)}(avg={np.mean(b_air_dists):.2f})" if b_air_dists else "")
    print()


def _export(wrapper, path, device):
    wrapper.eval()

    class Inf(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.network = net

        def forward(self, ss, ef, em, cc, sig, env, h):
            obs = {"self_state": ss, "entity_features": ef, "entity_mask": em,
                   "combat_ctx": cc, "sigil_state": sig, "env_state": env}
            logits, value, nh = self.network(obs, h)
            return torch.sigmoid(logits), value, nh

    inf = Inf(wrapper.network).to(device).eval()
    inputs = (
        torch.zeros(1, 38, device=device),
        torch.zeros(1, 8, 24, device=device),
        torch.zeros(1, 8, device=device),
        torch.zeros(1, 26, device=device),
        torch.zeros(1, 48, device=device),
        torch.zeros(1, 8, device=device),
        torch.zeros(1, 1, GRU_HIDDEN_DIM, device=device),
    )
    scripted = torch.jit.trace(inf, inputs, check_trace=False)
    scripted.save(path)
    print(f"Exported TorchScript model to {path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
