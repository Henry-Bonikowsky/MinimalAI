package com.minimalai.training;

import net.minecraft.core.BlockPos;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.Registries;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.effect.MobEffect;
import net.minecraft.world.effect.MobEffectInstance;
import net.minecraft.world.effect.MobEffects;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.enchantment.Enchantments;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.phys.Vec3;
import org.jetbrains.annotations.Nullable;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

/**
 * Records per-tick physics + combat state to CSV for system identification
 * and simulation training.
 *
 * Each tick captures full state BEFORE and AFTER action execution,
 * including bot state, target state, combat events, and environment.
 * This provides enough data to build a complete Minecraft PvP simulator.
 */
public class PhysicsRecorder implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger("MinimalAI");

    private static final String HEADER =
            // Identity
            "tick,bot," +
            // Pre-action bot state
            "pre_x,pre_y,pre_z,pre_vx,pre_vy,pre_vz," +
            "pre_yaw,pre_pitch,pre_onGround,pre_sprinting,pre_sneaking," +
            "pre_health,pre_max_health,pre_absorption,pre_food," +
            "pre_hurt_time,pre_attack_cooldown," +
            "pre_armor,pre_armor_toughness,pre_held_item," +
            // Actions (movement + combat-relevant)
            "act_fwd,act_back,act_left,act_right," +
            "act_jump,act_sneak,act_sprint," +
            "act_attack,act_block,act_eat_gap,act_sprint_reset," +
            // Post-action bot state
            "post_x,post_y,post_z,post_vx,post_vy,post_vz," +
            "post_yaw,post_pitch,post_onGround,post_sprinting,post_sneaking," +
            "post_health,post_max_health,post_absorption," +
            "post_hurt_time," +
            // Combat events this tick
            "dmg_dealt,dmg_taken," +
            "hit_vel_x,hit_vel_y,hit_vel_z," +
            // Target state (full snapshot at recording time)
            "tgt_x,tgt_y,tgt_z,tgt_vx,tgt_vy,tgt_vz," +
            "tgt_yaw,tgt_pitch,tgt_onGround,tgt_sprinting," +
            "tgt_health,tgt_max_health,tgt_absorption,tgt_hurt_time," +
            "tgt_armor,tgt_armor_toughness,tgt_held_item," +
            "tgt_attack_cooldown," +
            // Relative to target
            "target_dist,target_dx,target_dy,target_dz," +
            // Environment
            "block_friction," +
            // Enhanced physics columns (appended for backward compat)
            "pre_horizontal_collision,pre_vertical_collision,pre_fall_distance," +
            "pre_speed_amplifier,pre_strength_amplifier,pre_sharpness_level," +
            "tgt_sharpness_level,hit_attacker_sprinting," +
            "post_horizontal_collision,post_fall_distance";

    private final BufferedWriter writer;
    private final Path filePath;
    private int tickCounter = 0;
    private boolean closed = false;

    // Per-tick damage accumulators (set by external callbacks)
    private float dmgDealt = 0f;
    private float dmgTaken = 0f;
    // Velocity at the moment damage was received (before knockback is applied)
    private double hitVelX = 0, hitVelY = 0, hitVelZ = 0;
    private boolean wasHitThisTick = false;
    private boolean attackerSprinting = false;

    // --- 1-tick buffer for correct post-state capture ---
    // doTick() (physics) runs between BotBrain.tick() calls, so next tick's
    // pre-state is the true post-state of the current tick.
    private boolean hasPrevTick = false;
    private StateSnapshot bufferedPre;
    private int[] bufferedActions;
    private float bufferedBlockFriction;
    private String bufferedBotName;
    private @Nullable TargetSnapshot bufferedTargetSnap;
    private double bufferedRelDist, bufferedRelDx, bufferedRelDy, bufferedRelDz;
    private boolean bufferedHasTarget;

    public PhysicsRecorder(Path outputFile) throws IOException {
        this.filePath = outputFile;
        Files.createDirectories(outputFile.getParent());
        this.writer = Files.newBufferedWriter(outputFile);
        writer.write(HEADER);
        writer.newLine();
        LOG.info("[PhysicsRecorder] Recording to " + outputFile);
    }

    /**
     * Full snapshot of a player's physics + combat state at one instant.
     */
    public record StateSnapshot(
            double x, double y, double z,
            double vx, double vy, double vz,
            float yaw, float pitch,
            boolean onGround, boolean sprinting, boolean sneaking,
            float health, float maxHealth, float absorption,
            int foodLevel,
            int hurtTime,
            float attackCooldown,
            int armor, float armorToughness,
            int heldItemType,  // 0=sword, 1=axe, 2=other
            // Enhanced physics fields
            boolean horizontalCollision, boolean verticalCollision,
            float fallDistance,
            int speedAmplifier, int strengthAmplifier,
            int sharpnessLevel
    ) {
        public static StateSnapshot capture(ServerPlayer bot) {
            Vec3 vel = bot.getDeltaMovement();
            return new StateSnapshot(
                    bot.getX(), bot.getY(), bot.getZ(),
                    vel.x, vel.y, vel.z,
                    bot.getYRot(), bot.getXRot(),
                    bot.onGround(), bot.isSprinting(), bot.isShiftKeyDown(),
                    bot.getHealth(), (float) bot.getMaxHealth(), bot.getAbsorptionAmount(),
                    bot.getFoodData().getFoodLevel(),
                    bot.hurtTime,
                    bot.getAttackStrengthScale(0.5f),
                    bot.getArmorValue(), getArmorToughness(bot),
                    classifyWeapon(bot.getMainHandItem()),
                    bot.horizontalCollision, bot.verticalCollision,
                    (float) bot.fallDistance,
                    getEffectAmplifier(bot, MobEffects.SPEED),
                    getEffectAmplifier(bot, MobEffects.STRENGTH),
                    getSharpnessLevel(bot.getMainHandItem(), bot)
            );
        }
    }

    /**
     * Snapshot of a target entity's state.
     */
    public record TargetSnapshot(
            double x, double y, double z,
            double vx, double vy, double vz,
            float yaw, float pitch,
            boolean onGround, boolean sprinting,
            float health, float maxHealth, float absorption,
            int hurtTime,
            int armor, float armorToughness,
            int heldItemType,
            float attackCooldown,
            int sharpnessLevel
    ) {
        public static TargetSnapshot capture(LivingEntity target) {
            Vec3 vel = target.getDeltaMovement();
            float cooldown = (target instanceof Player p) ? p.getAttackStrengthScale(0.5f) : 1.0f;
            return new TargetSnapshot(
                    target.getX(), target.getY(), target.getZ(),
                    vel.x, vel.y, vel.z,
                    target.getYRot(), target.getXRot(),
                    target.onGround(), target.isSprinting(),
                    target.getHealth(), (float) target.getMaxHealth(), target.getAbsorptionAmount(),
                    target.hurtTime,
                    target.getArmorValue(), getArmorToughness(target),
                    classifyWeapon(target.getMainHandItem()),
                    cooldown,
                    getSharpnessLevel(target.getMainHandItem(), target)
            );
        }
    }

    /**
     * Record one tick using a 1-tick buffer for correct post-state capture.
     *
     * doTick() (which processes xxa/zza into actual movement) runs between
     * BotBrain.tick() calls, so next tick's pre-state is the true post-state
     * of the current tick. This method buffers each tick and writes the
     * previous tick's row when the next tick arrives.
     *
     * Combat accumulators (dmgDealt, dmgTaken, hitVel) collect events that
     * fire between ticks (from doTick processing attacks), so they correctly
     * belong to the row being written.
     *
     * @param botName       bot identifier
     * @param currentState  bot state captured NOW (before this tick's actions)
     * @param actions       the action vector being applied this tick
     * @param target        combat target (nullable)
     * @param botPlayer     the bot's ServerPlayer
     * @param blockFriction friction of block below bot
     */
    public void recordTick(String botName, StateSnapshot currentState, int[] actions,
                           @Nullable LivingEntity target, ServerPlayer botPlayer,
                           float blockFriction) {
        if (closed) return;

        if (hasPrevTick) {
            // Write previous tick's row: post-state = this tick's pre-state
            // Combat accumulators contain events from between previous and current tick
            writeRow(bufferedBotName, bufferedPre, currentState, bufferedActions,
                     bufferedHasTarget, bufferedTargetSnap,
                     bufferedRelDist, bufferedRelDx, bufferedRelDy, bufferedRelDz,
                     bufferedBlockFriction);
        }

        // Buffer current tick data
        bufferedPre = currentState;
        bufferedActions = actions.clone();
        bufferedBlockFriction = blockFriction;
        bufferedBotName = botName;

        // Capture target snapshot NOW (at pre-state time)
        if (target != null && target.isAlive()) {
            bufferedTargetSnap = TargetSnapshot.capture(target);
            double dx = target.getX() - botPlayer.getX();
            double dy = target.getY() - botPlayer.getY();
            double dz = target.getZ() - botPlayer.getZ();
            bufferedRelDist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            bufferedRelDx = dx;
            bufferedRelDy = dy;
            bufferedRelDz = dz;
            bufferedHasTarget = true;
        } else {
            bufferedTargetSnap = null;
            bufferedRelDist = bufferedRelDx = bufferedRelDy = bufferedRelDz = 0;
            bufferedHasTarget = false;
        }

        hasPrevTick = true;
    }

    /**
     * Flush the last buffered tick. Call before close() to avoid losing
     * the final tick. Captures one more snapshot as the post-state.
     */
    public void flush(ServerPlayer bot) {
        if (!hasPrevTick || closed) return;
        StateSnapshot finalState = StateSnapshot.capture(bot);
        writeRow(bufferedBotName, bufferedPre, finalState, bufferedActions,
                 bufferedHasTarget, bufferedTargetSnap,
                 bufferedRelDist, bufferedRelDx, bufferedRelDy, bufferedRelDz,
                 bufferedBlockFriction);
        hasPrevTick = false;
    }

    private void writeRow(String botName, StateSnapshot pre, StateSnapshot post,
                          int[] actions, boolean hasTarget, @Nullable TargetSnapshot tgt,
                          double relDist, double relDx, double relDy, double relDz,
                          float blockFriction) {
        try {
            StringBuilder sb = new StringBuilder(1024);

            // Identity
            sb.append(tickCounter++).append(',').append(botName).append(',');

            // Pre-action bot state
            appendBotState(sb, pre);

            // Actions (movement + combat-relevant subset)
            sb.append(actions[0]).append(',');  // fwd
            sb.append(actions[1]).append(',');  // back
            sb.append(actions[2]).append(',');  // strafe left
            sb.append(actions[3]).append(',');  // strafe right
            sb.append(actions[4]).append(',');  // jump
            sb.append(actions[5]).append(',');  // sneak
            sb.append(actions[6]).append(',');  // sprint
            sb.append(actions[7]).append(',');  // attack
            sb.append(actions[8]).append(',');  // block
            sb.append(actions[9]).append(',');  // eat gap
            sb.append(actions[12]).append(','); // sprint reset

            // Post-action bot state (now from NEXT tick's pre-state = true post-physics)
            appendPostState(sb, post);

            // Combat events (from accumulators: events between prev and current tick)
            sb.append(fmt(dmgDealt)).append(',');
            sb.append(fmt(dmgTaken)).append(',');
            if (wasHitThisTick) {
                sb.append(fmt(hitVelX)).append(',');
                sb.append(fmt(hitVelY)).append(',');
                sb.append(fmt(hitVelZ)).append(',');
            } else {
                sb.append(",,,");
            }

            // Target state (captured at pre-state time)
            if (hasTarget && tgt != null) {
                appendTargetState(sb, tgt);
                sb.append(fmt(relDist)).append(',');
                sb.append(fmt(relDx)).append(',');
                sb.append(fmt(relDy)).append(',');
                sb.append(fmt(relDz)).append(',');
            } else {
                // 22 empty target fields + 4 relative fields
                sb.append(",".repeat(22));
            }

            // Environment: block friction
            sb.append(fmt(blockFriction)).append(',');

            // Enhanced physics columns
            sb.append(pre.horizontalCollision() ? 1 : 0).append(',');
            sb.append(pre.verticalCollision() ? 1 : 0).append(',');
            sb.append(fmt(pre.fallDistance())).append(',');
            sb.append(pre.speedAmplifier()).append(',');
            sb.append(pre.strengthAmplifier()).append(',');
            sb.append(pre.sharpnessLevel()).append(',');
            // Target sharpness
            if (hasTarget && tgt != null) {
                sb.append(tgt.sharpnessLevel());
            }
            sb.append(',');
            // Hit attacker sprinting (from combat accumulator)
            sb.append(wasHitThisTick && attackerSprinting ? 1 : 0).append(',');
            // Post-state collision + fall distance
            sb.append(post.horizontalCollision() ? 1 : 0).append(',');
            sb.append(fmt(post.fallDistance()));

            // Reset combat accumulators
            dmgDealt = 0f;
            dmgTaken = 0f;
            hitVelX = hitVelY = hitVelZ = 0;
            wasHitThisTick = false;
            attackerSprinting = false;

            writer.write(sb.toString());
            writer.newLine();

        } catch (IOException e) {
            LOG.warning("[PhysicsRecorder] Write error: " + e.getMessage());
        }
    }

    /** Called from damage event listeners when bot deals damage. */
    public void onDamageDealt(float amount) {
        dmgDealt += amount;
    }

    /**
     * Called from damage event listeners when bot takes damage.
     * Velocity here is BEFORE knockback is applied (event fires before vanilla KB).
     *
     * @param attackerWasSprinting whether the attacker was sprinting at hit time (affects KB)
     */
    public void onDamageTaken(float amount, double velX, double velY, double velZ,
                              boolean attackerWasSprinting) {
        dmgTaken += amount;
        hitVelX = velX;
        hitVelY = velY;
        hitVelZ = velZ;
        wasHitThisTick = true;
        this.attackerSprinting = attackerWasSprinting;
    }

    public int getTickCount() {
        return tickCounter;
    }

    public Path getFilePath() {
        return filePath;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            writer.flush();
            writer.close();
            LOG.info("[PhysicsRecorder] Saved " + tickCounter + " ticks to " + filePath);
        } catch (IOException e) {
            LOG.warning("[PhysicsRecorder] Error closing: " + e.getMessage());
        }
    }

    // ------------------------------------------------------------------
    //  CSV formatting helpers
    // ------------------------------------------------------------------

    private void appendBotState(StringBuilder sb, StateSnapshot s) {
        sb.append(fmt(s.x)).append(',');
        sb.append(fmt(s.y)).append(',');
        sb.append(fmt(s.z)).append(',');
        sb.append(fmt(s.vx)).append(',');
        sb.append(fmt(s.vy)).append(',');
        sb.append(fmt(s.vz)).append(',');
        sb.append(fmt(s.yaw)).append(',');
        sb.append(fmt(s.pitch)).append(',');
        sb.append(s.onGround ? 1 : 0).append(',');
        sb.append(s.sprinting ? 1 : 0).append(',');
        sb.append(s.sneaking ? 1 : 0).append(',');
        sb.append(fmt(s.health)).append(',');
        sb.append(fmt(s.maxHealth)).append(',');
        sb.append(fmt(s.absorption)).append(',');
        sb.append(s.foodLevel).append(',');
        sb.append(s.hurtTime).append(',');
        sb.append(fmt(s.attackCooldown)).append(',');
        sb.append(s.armor).append(',');
        sb.append(fmt(s.armorToughness)).append(',');
        sb.append(s.heldItemType).append(',');
    }

    private void appendPostState(StringBuilder sb, StateSnapshot s) {
        sb.append(fmt(s.x)).append(',');
        sb.append(fmt(s.y)).append(',');
        sb.append(fmt(s.z)).append(',');
        sb.append(fmt(s.vx)).append(',');
        sb.append(fmt(s.vy)).append(',');
        sb.append(fmt(s.vz)).append(',');
        sb.append(fmt(s.yaw)).append(',');
        sb.append(fmt(s.pitch)).append(',');
        sb.append(s.onGround ? 1 : 0).append(',');
        sb.append(s.sprinting ? 1 : 0).append(',');
        sb.append(s.sneaking ? 1 : 0).append(',');
        sb.append(fmt(s.health)).append(',');
        sb.append(fmt(s.maxHealth)).append(',');
        sb.append(fmt(s.absorption)).append(',');
        sb.append(s.hurtTime).append(',');
    }

    private void appendTargetState(StringBuilder sb, TargetSnapshot t) {
        sb.append(fmt(t.x)).append(',');
        sb.append(fmt(t.y)).append(',');
        sb.append(fmt(t.z)).append(',');
        sb.append(fmt(t.vx)).append(',');
        sb.append(fmt(t.vy)).append(',');
        sb.append(fmt(t.vz)).append(',');
        sb.append(fmt(t.yaw)).append(',');
        sb.append(fmt(t.pitch)).append(',');
        sb.append(t.onGround ? 1 : 0).append(',');
        sb.append(t.sprinting ? 1 : 0).append(',');
        sb.append(fmt(t.health)).append(',');
        sb.append(fmt(t.maxHealth)).append(',');
        sb.append(fmt(t.absorption)).append(',');
        sb.append(t.hurtTime).append(',');
        sb.append(t.armor).append(',');
        sb.append(fmt(t.armorToughness)).append(',');
        sb.append(t.heldItemType).append(',');
        sb.append(fmt(t.attackCooldown)).append(',');
    }

    private static String fmt(double v) {
        return String.format("%.6f", v);
    }

    // ------------------------------------------------------------------
    //  Static helpers
    // ------------------------------------------------------------------

    /** Classify held item: 0=sword, 1=axe, 2=other */
    private static int classifyWeapon(ItemStack stack) {
        if (stack.isEmpty()) return 2;
        String path = net.minecraft.core.registries.BuiltInRegistries.ITEM.getKey(stack.getItem()).getPath();
        if (path.endsWith("_sword") || path.equals("sword")) return 0;
        if (path.endsWith("_axe") || path.equals("axe")) return 1;
        return 2;
    }

    /** Get armor toughness attribute safely. */
    private static float getArmorToughness(LivingEntity entity) {
        var inst = entity.getAttribute(Attributes.ARMOR_TOUGHNESS);
        return inst != null ? (float) inst.getValue() : 0f;
    }

    /** Get the amplifier of a mob effect on an entity, or -1 if not present. */
    private static int getEffectAmplifier(LivingEntity entity, Holder<MobEffect> effect) {
        MobEffectInstance inst = entity.getEffect(effect);
        return inst != null ? inst.getAmplifier() : -1;
    }

    /** Get the sharpness enchantment level on a weapon, or 0 if absent. */
    private static int getSharpnessLevel(ItemStack stack, LivingEntity entity) {
        if (stack.isEmpty()) return 0;
        try {
            var registry = entity.level().registryAccess().lookupOrThrow(Registries.ENCHANTMENT);
            var holder = registry.getOrThrow(Enchantments.SHARPNESS);
            return stack.getEnchantments().getLevel(holder);
        } catch (Exception e) {
            return 0;
        }
    }

    /**
     * Get the friction of the block below a player.
     * Default is 0.6, ice is 0.98, slime is 0.8, etc.
     */
    public static float getBlockFriction(ServerPlayer bot) {
        BlockPos below = bot.getBlockPosBelowThatAffectsMyMovement();
        Block block = bot.level().getBlockState(below).getBlock();
        return block.getFriction();
    }
}
