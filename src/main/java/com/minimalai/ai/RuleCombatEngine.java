package com.minimalai.ai;

import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.phys.Vec3;

import java.util.concurrent.ThreadLocalRandom;

import static com.minimalai.ai.ActionSpace.*;

/**
 * Rule-based combat engine implementing W-tap combos, retreating, healing,
 * and pearl usage. State machine with 5 phases.
 *
 * Returns int[35] compatible with ActionExecutor.execute().
 * Sigil bits (14-25) are left at 0 for the neural net to fill in hybrid mode.
 */
public class RuleCombatEngine {

    public enum CombatPhase {
        ENGAGE,       // sprint toward target, attack in reach
        COMBO,        // W-tap: release FWD 1 tick, re-sprint
        RETREAT,      // face away, heal, block
        PEARL_ESCAPE, // low HP, losing — pearl away
        PEARL_AGGRO   // healthy, target far — pearl toward
    }

    // Difficulty parameters
    private final int difficulty; // 1-5
    private final int reactionDelay;
    private final float aimJitter;
    private final double attackReach;
    private final float healThreshold;
    private final float wtapConsistency;
    private final float strafeFrequency;

    // State
    private CombatPhase currentPhase = CombatPhase.ENGAGE;
    private int comboTick = 0;       // ticks since last landed hit
    private int strafeCooldown = 0;  // ticks until next strafe direction switch
    private boolean strafeLeft = false;
    private int reactionBuffer = 0;  // delayed ticks before reacting to state changes
    private int lastHitTick = -100;  // tick when we last hit the target
    private int tickCounter = 0;
    private float lastTargetHealth = -1;
    private int retreatTicks = 0;    // how long we've been retreating
    private int pearlCooldownTicks = 0;
    private boolean wtapReleaseTick = false; // true = release forward this tick

    public RuleCombatEngine(int difficulty) {
        this.difficulty = Math.max(1, Math.min(5, difficulty));

        // Interpolate parameters based on difficulty (1=easiest, 5=hardest)
        float t = (this.difficulty - 1) / 4.0f;
        this.reactionDelay = Math.round(lerp(8, 0, t));
        this.aimJitter = lerp(15, 0, t);
        this.attackReach = lerp(2.0f, 3.0f, t);
        this.healThreshold = lerp(8, 14, t);
        this.wtapConsistency = lerp(0.50f, 1.0f, t);
        this.strafeFrequency = lerp(0.10f, 0.50f, t);
    }

    private static float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }

    public double getAttackReach() {
        return attackReach;
    }

    public CombatPhase getCurrentPhase() {
        return currentPhase;
    }

    /**
     * Generate a full 35-element action vector for this tick.
     * Sigil bits (14-25) are always 0 — hybrid mode fills those.
     */
    public int[] generateActions(ServerPlayer bot, LivingEntity target) {
        int[] actions = new int[NUM_ACTIONS];
        tickCounter++;

        if (target == null || !target.isAlive()) {
            // No target — stand still
            return actions;
        }

        ThreadLocalRandom rng = ThreadLocalRandom.current();
        double dist = bot.distanceTo(target);
        float botHp = bot.getHealth() + bot.getAbsorptionAmount();
        float targetHp = target.getHealth() + target.getAbsorptionAmount();

        // Track if we landed a hit (target health dropped)
        boolean landedHit = false;
        if (lastTargetHealth >= 0 && targetHp < lastTargetHealth - 0.1f) {
            landedHit = true;
            lastHitTick = tickCounter;
            comboTick = 0;
        } else {
            comboTick++;
        }
        lastTargetHealth = targetHp;

        // Tick cooldowns
        if (strafeCooldown > 0) strafeCooldown--;
        if (pearlCooldownTicks > 0) pearlCooldownTicks--;
        if (reactionBuffer > 0) {
            reactionBuffer--;
            // During reaction delay, continue previous phase without updating
        }

        // --- Phase transitions (with reaction delay) ---
        CombatPhase desiredPhase = evaluateDesiredPhase(botHp, targetHp, dist);
        if (desiredPhase != currentPhase && reactionBuffer <= 0) {
            currentPhase = desiredPhase;
            reactionBuffer = reactionDelay;
            retreatTicks = 0;
        }

        // --- Execute current phase ---
        switch (currentPhase) {
            case ENGAGE -> executeEngage(actions, bot, target, dist, rng);
            case COMBO -> executeCombo(actions, bot, target, dist, rng);
            case RETREAT -> executeRetreat(actions, bot, target, dist, botHp, rng);
            case PEARL_ESCAPE -> executePearlEscape(actions, bot, target, dist, botHp, rng);
            case PEARL_AGGRO -> executePearlAggro(actions, bot, target, dist, rng);
        }

        // Transition COMBO → ENGAGE if combo window expired
        if (currentPhase == CombatPhase.COMBO && comboTick > 20) {
            currentPhase = CombatPhase.ENGAGE;
        }

        // Sprint is always on (server behavior) except during W-tap release
        if (!wtapReleaseTick) {
            actions[ACT_SPRINT] = 1;
        }

        // Apply strafing (all phases except retreat)
        if (currentPhase != CombatPhase.RETREAT) {
            applyStrafe(actions, rng);
        }

        // Always face target when engaging/comboing
        if (currentPhase == CombatPhase.ENGAGE || currentPhase == CombatPhase.COMBO
                || currentPhase == CombatPhase.PEARL_AGGRO) {
            actions[ACT_ENGAGE] = 1;
        }

        // --- Sigil ability timing ---
        // Slots: [0]=Brace, [1]=Cleopatra, [2]=QuickSand, [3]=Grace
        // ActionExecutor maps these to ArcaneSigils bind slots
        applySigilTiming(actions, botHp, targetHp, dist, rng);

        return actions;
    }

    // Track sigil internal cooldowns (ActionExecutor handles server cooldowns,
    // but we need our own to avoid spamming activate every tick)
    private int braceCooldown = 0;
    private int cleoCooldown = 0;
    private int sandCooldown = 0;
    private int graceCooldown = 0;
    private boolean targetHadBuffs = false; // for Cleopatra counter-play

    private void applySigilTiming(int[] actions, float botHp, float targetHp,
                                   double dist, ThreadLocalRandom rng) {
        // Tick internal cooldowns
        if (braceCooldown > 0) braceCooldown--;
        if (cleoCooldown > 0) cleoCooldown--;
        if (sandCooldown > 0) sandCooldown--;
        if (graceCooldown > 0) graceCooldown--;

        float t = (difficulty - 1) / 4.0f;

        // --- King's Brace (slot 0): burst 80% DR ---
        // Use proactively when taking damage, not just when critical
        // Higher difficulty = smarter timing
        if (braceCooldown <= 0) {
            float braceThreshold = lerp(10, 18, t); // T1: HP<10, T5: HP<18
            if (botHp < braceThreshold) {
                actions[ACT_SIGIL_0] = 1;
                braceCooldown = 600; // 30s server cooldown
            }
        }

        // --- Cleopatra (slot 1): strip buffs + dmg amp ---
        // Best when target just used a defensive ability
        if (cleoCooldown <= 0) {
            // Counter-play: strip target's Brace/Grace immediately
            boolean targetDefending = targetHp < 14 && (targetHp > lastTargetHealth - 1);
            boolean targetHealthy = targetHp > 12;
            float cleoChance = lerp(0.3f, 0.9f, t);

            if (rng.nextFloat() < cleoChance) {
                if (targetDefending || (targetHealthy && dist < 5)) {
                    actions[ACT_SIGIL_0 + 1] = 1;
                    cleoCooldown = 180; // 9s
                }
            }
        }

        // --- Quick Sand (slot 2): slow + pull ---
        // Best during combo or when chasing
        if (sandCooldown <= 0) {
            boolean inCombat = dist < 6;
            boolean chasing = currentPhase == CombatPhase.ENGAGE && dist > 3;
            boolean comboing = currentPhase == CombatPhase.COMBO;
            float sandChance = lerp(0.4f, 1.0f, t);

            if (rng.nextFloat() < sandChance && (comboing || chasing || inCombat)) {
                actions[ACT_SIGIL_0 + 2] = 1;
                sandCooldown = 140; // 7s
            }
        }

        // --- Nile's Grace (slot 3): regen + DR ---
        // Use when damaged, especially during retreat
        if (graceCooldown <= 0) {
            float graceThreshold = lerp(12, 20, t); // T1: HP<12, T5: HP<20
            boolean retreating = currentPhase == CombatPhase.RETREAT;

            if (botHp < graceThreshold || (retreating && botHp < graceThreshold + 4)) {
                actions[ACT_SIGIL_0 + 3] = 1;
                graceCooldown = 180; // 9s
            }
        }
    }

    private CombatPhase evaluateDesiredPhase(float botHp, float targetHp, double dist) {
        boolean losing = botHp < targetHp - 4;

        // Pearl escape: very low HP and losing
        if (botHp < 5 && losing && pearlCooldownTicks <= 0) {
            return CombatPhase.PEARL_ESCAPE;
        }

        // Retreat: HP below heal threshold
        if (botHp < healThreshold) {
            return CombatPhase.RETREAT;
        }

        // Pearl aggro: healthy, target is far
        if (dist > 8 && botHp > 12 && pearlCooldownTicks <= 0) {
            return CombatPhase.PEARL_AGGRO;
        }

        // Combo: recently landed a hit (within 15 ticks)
        if (comboTick < 15) {
            return CombatPhase.COMBO;
        }

        return CombatPhase.ENGAGE;
    }

    // --- Phase execution ---

    private void executeEngage(int[] actions, ServerPlayer bot, LivingEntity target,
                               double dist, ThreadLocalRandom rng) {
        // Sprint toward target
        actions[ACT_FORWARD] = 1;

        // Attack when in reach
        if (dist <= attackReach) {
            actions[ACT_ATTACK] = 1;
        }

        // Jump occasionally for crits (more at higher difficulty)
        if (dist < 4 && dist > 2 && bot.onGround() && rng.nextFloat() < 0.12f * difficulty) {
            actions[ACT_JUMP] = 1;
        }

        wtapReleaseTick = false;
    }

    private void executeCombo(int[] actions, ServerPlayer bot, LivingEntity target,
                              double dist, ThreadLocalRandom rng) {
        // W-tap combo: alternate forward on/off
        boolean doWtap = rng.nextFloat() < wtapConsistency;

        if (doWtap && comboTick == 1) {
            // Release tick: stop forward, stop sprint (sprint resets)
            actions[ACT_FORWARD] = 0;
            actions[ACT_SPRINT] = 0;
            wtapReleaseTick = true;
        } else {
            // Re-engage: forward + sprint
            actions[ACT_FORWARD] = 1;
            actions[ACT_SPRINT] = 1;
            wtapReleaseTick = false;
        }

        // Always attack in reach during combo
        if (dist <= attackReach) {
            actions[ACT_ATTACK] = 1;
        }

        // Crit jumps during combo at high difficulty
        if (dist < 3.5 && bot.onGround() && rng.nextFloat() < 0.08f * difficulty
                && comboTick > 3) {
            actions[ACT_JUMP] = 1;
        }
    }

    private void executeRetreat(int[] actions, ServerPlayer bot, LivingEntity target,
                                double dist, float botHp, ThreadLocalRandom rng) {
        retreatTicks++;

        // Face away from target
        actions[ACT_FACE_AWAY] = 1;
        actions[ACT_FORWARD] = 1; // moving forward while facing away = running away
        actions[ACT_SPRINT] = 1;
        wtapReleaseTick = false;

        // Heal: eat golden apple
        if (botHp < healThreshold) {
            actions[ACT_EAT_GAP] = 1;
        }

        // Emergency pot if very low
        if (botHp < 8) {
            actions[ACT_LOOK_DOWN_SELF] = 1;
            actions[ACT_FACE_AWAY] = 0; // override face away to look down for self-pot
            actions[ACT_THROW_POT] = 1;
        }

        // Block if enemy is close while retreating
        if (dist < 3.5) {
            actions[ACT_BLOCK] = 1;
        }

        // Re-engage after healing enough or retreat timeout
        if (botHp > healThreshold + 4 || retreatTicks > 60) {
            currentPhase = CombatPhase.ENGAGE;
            retreatTicks = 0;
        }
    }

    private void executePearlEscape(int[] actions, ServerPlayer bot, LivingEntity target,
                                    double dist, float botHp, ThreadLocalRandom rng) {
        // Face away and pearl
        actions[ACT_FACE_AWAY] = 1;
        actions[ACT_FORWARD] = 1;
        actions[ACT_SPRINT] = 1;
        actions[ACT_THROW_PEARL] = 1;
        pearlCooldownTicks = 200; // 10s cooldown
        wtapReleaseTick = false;

        // Block while escaping
        if (dist < 4) {
            actions[ACT_BLOCK] = 1;
        }

        // After pearl, switch to retreat
        currentPhase = CombatPhase.RETREAT;
    }

    private void executePearlAggro(int[] actions, ServerPlayer bot, LivingEntity target,
                                   double dist, ThreadLocalRandom rng) {
        // Face target and pearl toward them
        actions[ACT_ENGAGE] = 1;
        actions[ACT_FORWARD] = 1;
        actions[ACT_SPRINT] = 1;
        actions[ACT_THROW_PEARL] = 1;
        pearlCooldownTicks = 200;
        wtapReleaseTick = false;

        // After pearl, switch to engage
        currentPhase = CombatPhase.ENGAGE;
    }

    private void applyStrafe(int[] actions, ThreadLocalRandom rng) {
        if (strafeCooldown <= 0 && rng.nextFloat() < strafeFrequency) {
            strafeLeft = !strafeLeft;
            strafeCooldown = 5 + rng.nextInt(15); // 5-20 tick strafe duration
        }

        if (strafeCooldown > 0) {
            if (strafeLeft) {
                actions[ACT_STRAFE_LEFT] = 1;
            } else {
                actions[ACT_STRAFE_RIGHT] = 1;
            }
        }
    }

    /**
     * Reset state for a new fight/episode.
     */
    public void reset() {
        currentPhase = CombatPhase.ENGAGE;
        comboTick = 100;
        strafeCooldown = 0;
        reactionBuffer = 0;
        lastHitTick = -100;
        tickCounter = 0;
        lastTargetHealth = -1;
        retreatTicks = 0;
        pearlCooldownTicks = 0;
        wtapReleaseTick = false;
        braceCooldown = 0;
        cleoCooldown = 0;
        sandCooldown = 0;
        graceCooldown = 0;
    }
}
