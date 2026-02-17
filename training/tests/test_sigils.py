"""Tests for Arcane Sigils implementations."""

import numpy as np
import pytest
from combat_sim.entities import Agent, Alliance, WeaponType, MAX_HEALTH, NUM_SIGIL_SLOTS
from combat_sim.sigils import SigilEngine, randomize_loadout, SIGIL_DEFS, _tier_scale


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sigil_engine(rng):
    return SigilEngine(rng=rng)


@pytest.fixture
def player(rng):
    agent = Agent(agent_id=0, alliance=Alliance.ALLY)
    agent.reset(x=0.0, z=0.0, facing=0.0)
    randomize_loadout(agent, rng)
    return agent


@pytest.fixture
def enemy(rng):
    agent = Agent(agent_id=1, alliance=Alliance.ENEMY)
    agent.reset(x=3.0, z=0.0, facing=np.pi)
    randomize_loadout(agent, rng)
    return agent


class TestLoadout:
    def test_weapon_sigils_always_equipped(self, player):
        """Weapon sigils (4, 5, 9, 10) should always be equipped."""
        for idx in [4, 5, 9, 10]:
            assert player.sigil_slots[idx].equipped

    def test_armor_slots_have_one_each(self, rng):
        """Each armor slot should have exactly one sigil (Pharaoh or Seasonal)."""
        agent = Agent(agent_id=0)
        agent.reset()
        randomize_loadout(agent, rng)

        # Helmet: slot 0 or 6
        assert agent.sigil_slots[0].equipped or agent.sigil_slots[6].equipped
        assert not (agent.sigil_slots[0].equipped and agent.sigil_slots[6].equipped)

        # Chestplate: slot 1 or 7
        assert agent.sigil_slots[1].equipped or agent.sigil_slots[7].equipped

        # Leggings: slot 2 or 8
        assert agent.sigil_slots[2].equipped or agent.sigil_slots[8].equipped

        # Boots: slot 3 or 11
        assert agent.sigil_slots[3].equipped or agent.sigil_slots[11].equipped

    def test_tier_randomized(self, rng):
        """Tiers should be randomized 1-5."""
        agent = Agent(agent_id=0)
        agent.reset()
        tiers = set()
        for _ in range(50):
            randomize_loadout(agent, rng)
            for slot in agent.sigil_slots:
                if slot.equipped:
                    tiers.add(slot.tier)
        assert min(tiers) >= 1
        assert max(tiers) <= 5

    def test_total_equipped_count(self, player):
        """Should have 4 weapon + 4 armor = 8 sigils equipped."""
        count = sum(1 for s in player.sigil_slots if s.equipped)
        assert count == 8


class TestTierScaling:
    def test_tier_1_is_base(self):
        assert _tier_scale(1.0, 1) == pytest.approx(1.0)

    def test_tier_5_is_higher(self):
        assert _tier_scale(1.0, 5) > 1.0

    def test_tier_5_value(self):
        # base * (1 + (5-1) * 0.15) = base * 1.6
        assert _tier_scale(1.0, 5) == pytest.approx(1.6)


class TestPharaohCurse:
    def test_stun_on_hit_received(self, sigil_engine, enemy):
        """Pharaoh's Curse should have a chance to stun attacker."""
        # Force equip pharaoh_curse on enemy
        enemy.sigil_slots[0].equipped = True
        enemy.sigil_slots[0].sigil_type = "pharaoh_curse"
        enemy.sigil_slots[0].cooldown_remaining = 0
        enemy.sigil_slots[0].cooldown_max = 200
        enemy.sigil_slots[0].tier = 5  # max chance

        attacker = Agent(agent_id=99, alliance=Alliance.ALLY)
        attacker.reset()

        # Try many times (stochastic)
        stunned = False
        for _ in range(100):
            enemy.sigil_slots[0].cooldown_remaining = 0
            result = sigil_engine.on_take_damage(enemy, attacker, 5.0, current_tick=100)
            if "pharaoh_curse_stun" in result["effects"]:
                stunned = True
                break

        assert stunned
        assert attacker.has_effect("stun")


class TestRulerHand:
    def test_bonus_damage_on_marked_target(self, sigil_engine):
        """Ruler's Hand should proc on targets with Pharaoh's Mark."""
        attacker = Agent(agent_id=0, alliance=Alliance.ALLY)
        attacker.reset()
        attacker.weapon = WeaponType.SWORD
        # Force equip rulers_hand
        attacker.sigil_slots[4].equipped = True
        attacker.sigil_slots[4].sigil_type = "rulers_hand"
        attacker.sigil_slots[4].tier = 5

        target = Agent(agent_id=1, alliance=Alliance.ENEMY)
        target.reset()
        target.pharaoh_marks.add(attacker.agent_id)  # target is marked

        procced = False
        for _ in range(100):
            result = sigil_engine.on_attack_hit(attacker, target, 10.0, current_tick=100)
            if result["bonus_damage"] > 0:
                procced = True
                break

        assert procced
        assert target.has_effect("wither")


class TestDivineIntervention:
    def test_invuln_while_blocking(self, sigil_engine):
        """Divine Intervention should grant invuln hits when blocking."""
        defender = Agent(agent_id=0, alliance=Alliance.ALLY)
        defender.reset()
        defender.is_blocking = True
        defender.sigil_slots[9].equipped = True
        defender.sigil_slots[9].sigil_type = "divine_intervention"
        defender.sigil_slots[9].cooldown_remaining = 0
        defender.sigil_slots[9].cooldown_max = 400
        defender.sigil_slots[9].tier = 5

        attacker = Agent(agent_id=1, alliance=Alliance.ENEMY)
        attacker.reset()

        procced = False
        for _ in range(200):
            defender.sigil_slots[9].cooldown_remaining = 0
            defender.invuln_hits = 0
            result = sigil_engine.on_take_damage(defender, attacker, 10.0, current_tick=100)
            if "divine_intervention_proc" in result["effects"]:
                procced = True
                assert defender.invuln_hits == 3
                break

        assert procced


class TestAbilities:
    def test_royal_bolster(self, sigil_engine, rng):
        """Royal Bolster should grant absorption + DR."""
        agent = Agent(agent_id=0)
        agent.reset()
        randomize_loadout(agent, rng)

        # Force equip
        agent.sigil_slots[2].equipped = True
        agent.sigil_slots[2].sigil_type = "royal_bolster"
        agent.sigil_slots[2].cooldown_remaining = 0
        agent.sigil_slots[2].cooldown_max = 400
        agent.sigil_slots[2].tier = 3
        agent.sigil_slots[2].activation_type = "ability"

        result = sigil_engine.activate_ability(agent, 2, [], current_tick=100)
        assert result
        assert agent.absorption > 0
        assert agent.has_effect("resistance")

    def test_quick_sand(self, sigil_engine, rng):
        """Quick Sand should grant no-KB and slow enemies."""
        agent = Agent(agent_id=0)
        agent.reset(x=0, z=0)
        agent.sigil_slots[3].equipped = True
        agent.sigil_slots[3].sigil_type = "quick_sand"
        agent.sigil_slots[3].cooldown_remaining = 0
        agent.sigil_slots[3].cooldown_max = 500
        agent.sigil_slots[3].tier = 3
        agent.sigil_slots[3].activation_type = "ability"

        nearby_enemy = Agent(agent_id=1, alliance=Alliance.ENEMY)
        nearby_enemy.reset(x=3.0, z=0.0)

        result = sigil_engine.activate_ability(agent, 3, [nearby_enemy], current_tick=100)
        assert result
        assert agent.no_knockback_ticks > 0
        assert nearby_enemy.has_effect("slowness")

    def test_cleopatra_strips_buffs(self, sigil_engine, rng):
        """Cleopatra should strip defensive buffs from enemies."""
        agent = Agent(agent_id=0)
        agent.reset(x=0, z=0)
        agent.sigil_slots[8].equipped = True
        agent.sigil_slots[8].sigil_type = "cleopatra"
        agent.sigil_slots[8].cooldown_remaining = 0
        agent.sigil_slots[8].cooldown_max = 3600
        agent.sigil_slots[8].tier = 3
        agent.sigil_slots[8].activation_type = "ability"

        target = Agent(agent_id=1, alliance=Alliance.ENEMY)
        target.reset(x=5.0, z=0.0)
        target.add_effect("resistance", amplifier=2, duration=200)
        target.absorption = 10.0

        result = sigil_engine.activate_ability(agent, 8, [target], current_tick=100)
        assert result
        assert target.absorption == 0.0
        assert not target.has_effect("resistance")
        assert target.has_effect("weakness")

    def test_niles_grace(self, sigil_engine, rng):
        """Nile's Grace should grant regen + resistance to self and allies."""
        agent = Agent(agent_id=0, alliance=Alliance.ALLY)
        agent.reset(x=0, z=0)
        agent.sigil_slots[10].equipped = True
        agent.sigil_slots[10].sigil_type = "niles_grace"
        agent.sigil_slots[10].cooldown_remaining = 0
        agent.sigil_slots[10].cooldown_max = 3600
        agent.sigil_slots[10].tier = 3
        agent.sigil_slots[10].activation_type = "ability"

        ally = Agent(agent_id=2, alliance=Alliance.ALLY)
        ally.reset(x=3.0, z=0.0)

        result = sigil_engine.activate_ability(agent, 10, [ally], current_tick=100)
        assert result
        assert agent.has_effect("regen")
        assert agent.has_effect("resistance")
        assert ally.has_effect("regen")
        assert ally.has_effect("resistance")

    def test_royal_guard_spawns_mummies(self, sigil_engine, rng):
        """Royal Guard should spawn 2 mummies."""
        agent = Agent(agent_id=0)
        agent.reset()
        agent.target_id = 1
        agent.sigil_slots[5].equipped = True
        agent.sigil_slots[5].sigil_type = "royal_guard"
        agent.sigil_slots[5].cooldown_remaining = 0
        agent.sigil_slots[5].cooldown_max = 2400
        agent.sigil_slots[5].tier = 3
        agent.sigil_slots[5].activation_type = "ability"

        result = sigil_engine.activate_ability(agent, 5, [], current_tick=100)
        assert result
        assert len(agent.mummies) == 2

    def test_kings_brace_charge_and_burst(self, sigil_engine, rng):
        """King's Brace: gain charges, then burst at 100."""
        agent = Agent(agent_id=0)
        agent.reset()
        agent.sigil_slots[7].equipped = True
        agent.sigil_slots[7].sigil_type = "kings_brace"
        agent.sigil_slots[7].cooldown_remaining = 0
        agent.sigil_slots[7].cooldown_max = 600
        agent.sigil_slots[7].tier = 3
        agent.sigil_slots[7].activation_type = "auto_defense_ability"

        attacker = Agent(agent_id=1)
        attacker.reset()

        # Gain charges by taking hits
        for i in range(100):
            sigil_engine.on_take_damage(agent, attacker, 1.0, current_tick=i)

        assert agent.kings_brace_charges == 100

        # Activate burst
        result = sigil_engine.activate_ability(agent, 7, [], current_tick=200)
        assert result
        assert agent.kings_brace_charges == 0
        assert agent.has_effect("resistance")


class TestPassives:
    def test_ancient_crown_immunity(self, rng):
        """Ancient Crown should give negative effect immunity."""
        agent = Agent(agent_id=0)
        agent.reset()
        agent.sigil_slots[6].equipped = True
        agent.sigil_slots[6].sigil_type = "ancient_crown"
        agent.sigil_slots[6].tier = 5

        # Manually apply passive
        from combat_sim.sigils import _apply_passives
        _apply_passives(agent)

        assert agent.negative_effect_immunity > 0

    def test_sandwalker_speed(self, rng):
        """Sandwalker should give permanent speed boost."""
        agent = Agent(agent_id=0)
        agent.reset()
        agent.sigil_slots[11].equipped = True
        agent.sigil_slots[11].sigil_type = "sandwalker"
        agent.sigil_slots[11].tier = 1

        from combat_sim.sigils import _apply_passives
        _apply_passives(agent)

        assert agent.has_effect("speed")


class TestAbilityMask:
    def test_ability_mask_shape(self, sigil_engine, player):
        mask = sigil_engine.get_ability_mask(player)
        assert mask.shape == (NUM_SIGIL_SLOTS,)

    def test_passive_sigils_not_in_mask(self, sigil_engine, rng):
        """Passive sigils should not appear in ability mask."""
        agent = Agent(agent_id=0)
        agent.reset()
        randomize_loadout(agent, rng)

        mask = sigil_engine.get_ability_mask(agent)
        # Slot 6 (ancient_crown) and 11 (sandwalker) are passive
        if agent.sigil_slots[6].equipped:
            assert mask[6] == 0.0
        if agent.sigil_slots[11].equipped:
            assert mask[11] == 0.0

    def test_auto_sigils_not_in_mask(self, sigil_engine, rng):
        """Auto-trigger sigils should not appear in ability mask."""
        agent = Agent(agent_id=0)
        agent.reset()
        randomize_loadout(agent, rng)

        mask = sigil_engine.get_ability_mask(agent)
        # Slot 0 (pharaoh_curse) is auto_defense
        if agent.sigil_slots[0].equipped:
            assert mask[0] == 0.0
        # Slot 4 (rulers_hand) is auto_attack
        assert mask[4] == 0.0

    def test_cooldown_blocks_activation(self, sigil_engine, rng):
        """Sigils on cooldown should be masked."""
        agent = Agent(agent_id=0)
        agent.reset()
        randomize_loadout(agent, rng)

        # Put all on cooldown
        for slot in agent.sigil_slots:
            slot.cooldown_remaining = 100

        mask = sigil_engine.get_ability_mask(agent)
        assert mask.sum() == 0.0


class TestMummies:
    def test_mummies_deal_damage(self, sigil_engine):
        """Mummies should deal damage to their target."""
        owner = Agent(agent_id=0, alliance=Alliance.ALLY)
        owner.reset()
        owner.mummies = [{
            "hp": 50.0,
            "damage": 6.0,
            "ticks_remaining": 21,  # After decrement becomes 20 (divisible by 20)
            "target_id": 1,
        }]

        target = Agent(agent_id=1, alliance=Alliance.ENEMY)
        target.reset()
        target.health = MAX_HEALTH

        sigil_engine.tick_mummies(owner, [owner, target], current_tick=100)
        assert target.health < MAX_HEALTH

    def test_mummies_expire(self, sigil_engine):
        """Mummies should expire after duration."""
        owner = Agent(agent_id=0)
        owner.reset()
        owner.mummies = [{
            "hp": 50.0, "damage": 6.0, "ticks_remaining": 1, "target_id": 1,
        }]

        sigil_engine.tick_mummies(owner, [owner], current_tick=100)
        assert len(owner.mummies) == 0
