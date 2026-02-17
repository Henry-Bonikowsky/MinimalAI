"""Arcane Sigils implementation for combat simulator.

12 sigils from Pharaoh + Seasonal sets. Each modeled with its mechanical
combat effect (particles/sounds/messages skipped for sim).

Sigil Slot Mapping:
  0: pharaoh_curse     (Helmet, DEFENSE auto)
  1: sandstorm         (Chestplate, ATTACK auto)
  2: royal_bolster     (Leggings, ABILITY)
  3: quick_sand        (Boots, ABILITY)
  4: rulers_hand       (Sword, ATTACK auto) - ALWAYS EQUIPPED
  5: royal_guard       (Axe, ABILITY) - ALWAYS EQUIPPED
  6: ancient_crown     (Helmet, PASSIVE)
  7: kings_brace       (Chestplate, DEFENSE auto + ABILITY)
  8: cleopatra         (Leggings, ABILITY)
  9: divine_intervention (Sword, DEFENSE auto) - ALWAYS EQUIPPED
  10: niles_grace      (Axe, ABILITY) - ALWAYS EQUIPPED
  11: sandwalker       (Boots, PASSIVE)

Weapon sigils (4, 5, 9, 10) are ALWAYS equipped from both sets.
Armor sigils are per-piece randomized (Pharaoh OR Seasonal per slot).
"""

import numpy as np
from .entities import Agent, SigilSlot, StatusEffect, NUM_SIGIL_SLOTS


# Tier scaling: tier 1-5 multiplies the base value
def _tier_scale(base: float, tier: int, scale: float = 0.15) -> float:
    """Scale a value by tier. tier 1 = base, tier 5 = base * (1 + 4*scale)."""
    return base * (1.0 + (tier - 1) * scale)


# --- Sigil definitions with base stats ---

SIGIL_DEFS = {
    "pharaoh_curse": {
        "slot": 0, "piece": "helmet", "set": "pharaoh",
        "activation": "auto_defense",
        "base_chance": 0.15, "base_stun_ticks": 20, "cooldown": 200,  # 10s
    },
    "sandstorm": {
        "slot": 1, "piece": "chestplate", "set": "pharaoh",
        "activation": "auto_attack",
        "base_chance": 0.20, "radius": 5.0, "slow_mult": 0.25, "cooldown": 0,  # no cd
    },
    "royal_bolster": {
        "slot": 2, "piece": "leggings", "set": "pharaoh",
        "activation": "ability",
        "base_absorption": 4.0, "base_dr": 0.15, "duration": 100, "cooldown": 400,  # 20s
    },
    "quick_sand": {
        "slot": 3, "piece": "boots", "set": "pharaoh",
        "activation": "ability",
        "duration": 120, "cooldown": 500,  # 25s
    },
    "rulers_hand": {
        "slot": 4, "piece": "sword", "set": "pharaoh",
        "activation": "auto_attack",
        "base_chance": 0.25, "base_damage_boost": 0.30, "cooldown": 0,
    },
    "royal_guard": {
        "slot": 5, "piece": "axe", "set": "pharaoh",
        "activation": "ability",
        "mummy_hp": 50.0, "mummy_damage": 6.0, "mummy_duration": 200, "cooldown": 2400,  # 120s
    },
    "ancient_crown": {
        "slot": 6, "piece": "helmet", "set": "seasonal",
        "activation": "passive",
        "base_immunity": 0.25, "cooldown": 0,
    },
    "kings_brace": {
        "slot": 7, "piece": "chestplate", "set": "seasonal",
        "activation": "auto_defense_ability",
        "base_dr_per_charge": 0.003, "max_charges": 100,
        "burst_duration": 100, "cooldown": 600,  # 30s for burst
    },
    "cleopatra": {
        "slot": 8, "piece": "leggings", "set": "seasonal",
        "activation": "ability",
        "base_damage_amp": 0.20, "duration": 100, "cooldown": 3600,  # 180s
    },
    "divine_intervention": {
        "slot": 9, "piece": "sword", "set": "seasonal",
        "activation": "auto_defense",
        "base_chance": 0.10, "invuln_hits": 3, "cooldown": 400,  # 20s
    },
    "niles_grace": {
        "slot": 10, "piece": "axe", "set": "seasonal",
        "activation": "ability",
        "regen_amp": 2, "resistance_amp": 2, "duration": 100, "cooldown": 3600,  # 180s
    },
    "sandwalker": {
        "slot": 11, "piece": "boots", "set": "seasonal",
        "activation": "passive",
        "base_speed_amp": 0, "cooldown": 0,  # Speed I always on
    },
}


def randomize_loadout(agent: Agent, rng: np.random.Generator):
    """Randomize sigil loadout for an episode.

    Weapon sigils (4, 5, 9, 10) are ALWAYS equipped.
    Armor sigils: for each slot (helmet, chest, legs, boots),
    pick either the Pharaoh or Seasonal sigil. Tier randomized 1-5.
    """
    # Reset all slots
    for slot in agent.sigil_slots:
        slot.equipped = False
        slot.cooldown_remaining = 0
        slot.tier = 1
        slot.sigil_type = ""
        slot.activation_type = ""
        slot.last_used_tick = -1000

    # Weapon sigils: ALWAYS equipped
    weapon_slots = [4, 5, 9, 10]
    for idx in weapon_slots:
        for name, defn in SIGIL_DEFS.items():
            if defn["slot"] == idx:
                slot = agent.sigil_slots[idx]
                slot.equipped = True
                slot.sigil_type = name
                slot.tier = rng.integers(1, 6)  # 1-5
                slot.cooldown_max = defn["cooldown"]
                slot.activation_type = defn["activation"]
                break

    # Armor slots: pick Pharaoh OR Seasonal per piece
    armor_pairs = [
        (0, 6),   # helmet: pharaoh_curse vs ancient_crown
        (1, 7),   # chestplate: sandstorm vs kings_brace
        (2, 8),   # leggings: royal_bolster vs cleopatra
        (3, 11),  # boots: quick_sand vs sandwalker
    ]

    for pharaoh_idx, seasonal_idx in armor_pairs:
        chosen = pharaoh_idx if rng.random() < 0.5 else seasonal_idx
        for name, defn in SIGIL_DEFS.items():
            if defn["slot"] == chosen:
                slot = agent.sigil_slots[chosen]
                slot.equipped = True
                slot.sigil_type = name
                slot.tier = rng.integers(1, 6)
                slot.cooldown_max = defn["cooldown"]
                slot.activation_type = defn["activation"]
                break

    # Apply passive effects immediately
    _apply_passives(agent)


def _apply_passives(agent: Agent):
    """Apply passive sigil effects (ancient_crown, sandwalker)."""
    for slot in agent.sigil_slots:
        if not slot.equipped:
            continue
        if slot.sigil_type == "ancient_crown":
            agent.negative_effect_immunity = _tier_scale(
                SIGIL_DEFS["ancient_crown"]["base_immunity"], slot.tier
            )
        elif slot.sigil_type == "sandwalker":
            agent.add_effect("speed", amplifier=0, duration=999999)


class SigilEngine:
    """Processes sigil triggers and ability activations."""

    def __init__(self, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()

    def on_attack_hit(self, attacker: Agent, target: Agent, damage: float,
                      current_tick: int) -> dict:
        """Process auto-attack sigils when attacker lands a hit.

        Returns dict of modifications: {"bonus_damage": float, "effects": list}
        """
        result = {"bonus_damage": 0.0, "effects": []}

        for slot in attacker.sigil_slots:
            if not slot.equipped:
                continue

            # Sandstorm: chance to apply slow + mark on nearby enemies
            if slot.sigil_type == "sandstorm":
                defn = SIGIL_DEFS["sandstorm"]
                chance = _tier_scale(defn["base_chance"], slot.tier)
                if self.rng.random() < chance:
                    target.add_effect("slowness", amplifier=0,
                                      duration=60)  # 3s slow
                    target.pharaoh_marks.add(attacker.agent_id)
                    result["effects"].append("sandstorm_applied")

            # Ruler's Hand: bonus damage if target has Pharaoh's Mark
            if (slot.sigil_type == "rulers_hand" and
                    attacker.weapon == 1):  # SWORD
                defn = SIGIL_DEFS["rulers_hand"]
                chance = _tier_scale(defn["base_chance"], slot.tier)
                if (attacker.agent_id in target.pharaoh_marks and
                        self.rng.random() < chance):
                    boost = _tier_scale(defn["base_damage_boost"], slot.tier)
                    result["bonus_damage"] += damage * boost
                    target.add_effect("wither", amplifier=0, duration=100)
                    result["effects"].append("rulers_hand_proc")

        return result

    def on_take_damage(self, defender: Agent, attacker: Agent, damage: float,
                       current_tick: int) -> dict:
        """Process auto-defense sigils when defender takes damage.

        Returns dict: {"damage_reduction": float, "effects": list}
        """
        result = {"damage_reduction": 0.0, "effects": []}

        # Check invuln hits FIRST (from previous divine intervention)
        if defender.invuln_hits > 0:
            result["damage_reduction"] += damage  # negate all damage
            defender.invuln_hits -= 1
            result["effects"].append("invuln_hit_consumed")
            return result  # skip further sigil processing when invuln

        for slot in defender.sigil_slots:
            if not slot.equipped:
                continue

            # Pharaoh's Curse: chance to stun attacker
            if (slot.sigil_type == "pharaoh_curse" and
                    slot.cooldown_remaining == 0):
                defn = SIGIL_DEFS["pharaoh_curse"]
                chance = _tier_scale(defn["base_chance"], slot.tier)
                if self.rng.random() < chance:
                    stun_ticks = int(_tier_scale(defn["base_stun_ticks"], slot.tier))
                    attacker.add_effect("stun", amplifier=0, duration=stun_ticks)
                    slot.cooldown_remaining = slot.cooldown_max
                    slot.last_used_tick = current_tick
                    result["effects"].append("pharaoh_curse_stun")

            # King's Brace: gain charge on hit taken
            if slot.sigil_type == "kings_brace":
                defn = SIGIL_DEFS["kings_brace"]
                if defender.kings_brace_charges < defn["max_charges"]:
                    defender.kings_brace_charges += 1
                    dr = defender.kings_brace_charges * _tier_scale(
                        defn["base_dr_per_charge"], slot.tier
                    )
                    defender.sigil_damage_reduction = min(0.5, dr)  # cap at 50%

            # Divine Intervention: chance for invuln when blocking
            if (slot.sigil_type == "divine_intervention" and
                    defender.is_blocking and
                    slot.cooldown_remaining == 0):
                defn = SIGIL_DEFS["divine_intervention"]
                chance = _tier_scale(defn["base_chance"], slot.tier)
                if self.rng.random() < chance:
                    defender.invuln_hits = defn["invuln_hits"]
                    slot.cooldown_remaining = slot.cooldown_max
                    slot.last_used_tick = current_tick
                    result["effects"].append("divine_intervention_proc")

        return result

    def activate_ability(self, agent: Agent, slot_index: int,
                         targets: list, current_tick: int) -> bool:
        """Manually activate an ability sigil. Returns True if activated."""
        if slot_index < 0 or slot_index >= NUM_SIGIL_SLOTS:
            return False

        slot = agent.sigil_slots[slot_index]
        if not slot.equipped or slot.cooldown_remaining > 0:
            return False

        # Only ability-type sigils can be manually activated
        if "ability" not in slot.activation_type:
            return False

        name = slot.sigil_type

        if name == "royal_bolster":
            defn = SIGIL_DEFS["royal_bolster"]
            absorption = _tier_scale(defn["base_absorption"], slot.tier)
            dr = _tier_scale(defn["base_dr"], slot.tier)
            agent.absorption = min(20.0, agent.absorption + absorption)
            agent.add_effect("resistance", amplifier=0, duration=defn["duration"])
            agent.sigil_damage_reduction = min(0.5, agent.sigil_damage_reduction + dr)

        elif name == "quick_sand":
            defn = SIGIL_DEFS["quick_sand"]
            agent.no_knockback_ticks = defn["duration"]
            # Slow nearby enemies
            for target in targets:
                if target.agent_id != agent.agent_id and target.is_alive:
                    dist = np.sqrt((target.x - agent.x)**2 + (target.z - agent.z)**2)
                    if dist < 6.0:
                        target.add_effect("slowness", amplifier=1, duration=defn["duration"])

        elif name == "royal_guard":
            defn = SIGIL_DEFS["royal_guard"]
            # Spawn 2 mummies (simplified as damage-over-time to nearest enemy)
            for _ in range(2):
                agent.mummies.append({
                    "hp": _tier_scale(defn["mummy_hp"], slot.tier),
                    "damage": _tier_scale(defn["mummy_damage"], slot.tier),
                    "ticks_remaining": defn["mummy_duration"],
                    "target_id": agent.target_id,
                })

        elif name == "kings_brace" and agent.kings_brace_charges >= 100:
            defn = SIGIL_DEFS["kings_brace"]
            agent.kings_brace_charges = 0
            agent.sigil_damage_reduction = 0.0
            agent.add_effect("resistance", amplifier=2, duration=defn["burst_duration"])

        elif name == "cleopatra":
            defn = SIGIL_DEFS["cleopatra"]
            amp = _tier_scale(defn["base_damage_amp"], slot.tier)
            # Strip all defensive buffs from targets + apply damage amp
            for target in targets:
                if target.agent_id != agent.agent_id and target.is_alive:
                    dist = np.sqrt((target.x - agent.x)**2 + (target.z - agent.z)**2)
                    if dist < 10.0:  # 10 block range
                        target.clear_defensive_buffs()
                        # Mark with damage vulnerability
                        target.sigil_damage_amp = amp
                        target.add_effect("weakness", amplifier=1, duration=defn["duration"])

        elif name == "niles_grace":
            defn = SIGIL_DEFS["niles_grace"]
            # Regen III + Resistance III to self and allies
            agent.add_effect("regen", amplifier=defn["regen_amp"], duration=defn["duration"])
            agent.add_effect("resistance", amplifier=defn["resistance_amp"], duration=defn["duration"])
            for target in targets:
                if (target.agent_id != agent.agent_id and
                        target.alliance == agent.alliance and target.is_alive):
                    dist = np.sqrt((target.x - agent.x)**2 + (target.z - agent.z)**2)
                    if dist < 10.0:
                        target.add_effect("regen", amplifier=defn["regen_amp"], duration=defn["duration"])
                        target.add_effect("resistance", amplifier=defn["resistance_amp"], duration=defn["duration"])

        else:
            return False

        slot.cooldown_remaining = slot.cooldown_max
        slot.last_used_tick = current_tick
        return True

    def tick_mummies(self, agent: Agent, all_agents: list, current_tick: int):
        """Tick mummy entities spawned by Royal Guard."""
        from .entities import HURT_IMMUNE_TICKS
        remaining = []
        for mummy in agent.mummies:
            mummy["ticks_remaining"] -= 1
            if mummy["ticks_remaining"] <= 0 or mummy["hp"] <= 0:
                continue
            # Deal damage to target every 20 ticks (1 second)
            if mummy["ticks_remaining"] % 20 == 0:
                for target in all_agents:
                    if target.agent_id == mummy["target_id"] and target.is_alive:
                        # Respect hurt immunity
                        if target.hurt_immune_ticks > 0:
                            break
                        # Apply armor reduction like a normal hit
                        raw = mummy["damage"]
                        armor_red = min(20, max(target.armor / 5,
                            target.armor - raw / (2 + target.armor_toughness / 4))) / 25
                        prot_red = min(20, target.protection_epf) / 25.0
                        actual = raw * (1.0 - armor_red) * (1.0 - prot_red)
                        actual = max(0.0, actual)
                        if target.absorption > 0:
                            absorbed = min(target.absorption, actual)
                            target.absorption -= absorbed
                            actual -= absorbed
                        target.health = max(0.0, target.health - actual)
                        target.damage_taken_this_tick += actual
                        target.hurt_immune_ticks = HURT_IMMUNE_TICKS
                        if target.health <= 0:
                            target.is_alive = False
                            agent.kills += 1
                        break
            remaining.append(mummy)
        agent.mummies = remaining

    def tick_sigil_cooldowns(self, agent: Agent):
        """Tick all sigil cooldowns down by 1."""
        for slot in agent.sigil_slots:
            if slot.cooldown_remaining > 0:
                slot.cooldown_remaining -= 1
        # Tick no-KB timer
        if agent.no_knockback_ticks > 0:
            agent.no_knockback_ticks -= 1
        # Decay sigil damage amp on targets over time
        if agent.sigil_damage_amp > 0:
            agent.sigil_damage_amp = max(0.0, agent.sigil_damage_amp - 0.002)

    def get_ability_mask(self, agent: Agent) -> np.ndarray:
        """Return a 12-dim mask: 1 if the sigil slot can be manually activated."""
        mask = np.zeros(NUM_SIGIL_SLOTS, dtype=np.float32)
        for i, slot in enumerate(agent.sigil_slots):
            if (slot.equipped and
                    "ability" in slot.activation_type and
                    slot.cooldown_remaining == 0):
                # Kings brace ability requires 100 charges
                if slot.sigil_type == "kings_brace" and agent.kings_brace_charges < 100:
                    continue
                mask[i] = 1.0
        return mask
