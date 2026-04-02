# MinimalAI — Core Idea

AI-controlled combat bots for competitive Minecraft PvP, trained through reinforcement learning and hand-tuned rule systems.

## Why

Most Minecraft bots use simple if-else logic and fall apart against real players. MinimalAI embeds real ML inference (PyTorch via DJL) server-side, running per-tick neural network decisions alongside a hand-coded combat engine. The bots learn advanced PvP techniques — W-tap combos, spacing, heal timing, ability usage — that actually compete with humans.

## The Vision

A **Legion** of 20 AI bots fighting as a coordinated hive-mind unit against teams of real players. Roles (healer, frontline, assassin), coordinated plays (focus fire, pearl-onto-healer for AoE buffs, bait-and-switch), real-time tactical adaptation — no human commander needed.

## How It Works

1. **Observe** — Each tick, collect game state: HP, position, velocity, nearby entities, cooldowns, active buffs
2. **Decide** — Rule engine (state machine) or neural net picks actions: movement, attacks, items, ability timing
3. **Act** — Action executor translates decisions into NMS server operations on fake players
4. **Train** — Python-side vectorized physics sim runs millions of fights → PPO → export TorchScript model → deploy to server

## Stack

- **Server**: Paper 1.21.10 (Kitara combat fork), Java 21
- **AI Runtime**: Deep Java Library (DJL) + PyTorch TorchScript
- **Training**: Python, NumPy vectorized sim, PPO, CUDA (RTX 5080)
- **Combat System**: Kitara 1.8-style (no cooldown), Arcane Sigils (custom abilities)
