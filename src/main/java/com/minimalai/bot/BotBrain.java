package com.minimalai.bot;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.minimalai.ai.*;
import com.minimalai.integration.ArcaneSigilsAPI;
import com.minimalai.training.ExperienceBuffer;
import com.minimalai.training.PhysicsRecorder;
import com.minimalai.training.RewardComputer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.phys.AABB;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Logger;

/**
 * Per-bot tick loop: observe → infer → mask → sample → act → collect experience.
 *
 * Each BotBrain owns its own DJL Predictor (not thread-safe) and GRU hidden state.
 */
public class BotBrain {

    private static final Logger LOG = Logger.getLogger("MinimalAI");
    private static final double NEARBY_RANGE = 30.0;

    private final String name;
    private final ServerPlayer bot;
    private final Predictor<NDList, NDList> predictor;
    private final NDManager ndManager;
    private final ObservationBuilder obsBuilder;
    private final ActionExecutor actionExecutor;
    private final ModelManager modelManager;
    private final @Nullable ArcaneSigilsAPI sigilsApi;

    // GRU hidden state - zeroed on episode reset
    private float[] hidden = new float[ObservationSpace.HIDDEN_DIM];

    // Current target entity
    private @Nullable LivingEntity target;

    // Training components (null if training disabled)
    private @Nullable ExperienceBuffer experienceBuffer;
    private @Nullable RewardComputer rewardComputer;
    private boolean trainingEnabled = false;

    // Self-play: pause ticking when "dead", ally UUIDs to avoid targeting teammates
    private boolean paused = false;
    private Set<UUID> allyUUIDs = Set.of();

    // Arena boundary enforcement: teleport back if bot strays too far
    private double spawnX, spawnY, spawnZ;
    private double arenaRadius = 30.0; // max distance from spawn before teleport
    private boolean hasSpawnAnchor = false;

    // Engagement: force approach if idle too long (prevents sim-to-real stalemates)
    private int idleTicks = 0;
    private int forceEngageTicks = 0; // countdown of forced engagement
    private float lastCombatHealth = -1;
    private static final int IDLE_THRESHOLD = 60; // 3 seconds without health change
    private static final int FORCE_ENGAGE_DURATION = 100; // force approach for 5 seconds

    // Physics recording mode (random actions, no model needed)
    private @Nullable PhysicsRecorder physicsRecorder;
    private boolean recordingMode = false;
    private float prevBotHealth = -1; // for direct damage-taken tracking

    // Random action probabilities for recording mode
    private static final float[] RANDOM_PROBS = {
        0.6f,  // forward (biased toward moving)
        0.1f,  // backward
        0.2f,  // strafe left
        0.2f,  // strafe right
        0.15f, // jump
        0.05f, // sneak
        0.4f,  // sprint (biased toward sprinting)
        0.3f,  // attack
        0.05f, // block
        0.02f, // eat gap
        0.02f, // throw pot
        0.02f, // throw pearl
        0.1f,  // sprint reset
        0.0f,  // swap weapon (always masked)
        0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, // sigils 0-5
        0.02f, 0.02f, 0.02f, 0.02f, 0.02f, 0.02f, // sigils 6-11
        0.5f,  // face target
        0.1f,  // face away
        0.05f, // look down self
        0.2f,  // face movement
        0.1f, 0.05f, 0.03f, 0.02f, 0.01f // target selection 0-4
    };

    public BotBrain(String name,
                    ServerPlayer bot,
                    ModelManager modelManager,
                    String modelName,
                    ObservationBuilder obsBuilder,
                    ActionExecutor actionExecutor,
                    @Nullable ArcaneSigilsAPI sigilsApi) {
        this.name = name;
        this.bot = bot;
        this.modelManager = modelManager;
        this.predictor = modelManager.createPredictor(modelName);
        this.ndManager = NDManager.newBaseManager();
        this.obsBuilder = obsBuilder;
        this.actionExecutor = actionExecutor;
        this.sigilsApi = sigilsApi;
    }

    /**
     * Execute one tick of the bot's brain: observe → infer → act.
     * In recording mode, uses random actions and logs physics data.
     * Called from the server tick scheduler.
     */
    // Respawn cooldown (ticks to wait after death before respawning)
    private int respawnCooldown = 0;
    private static final int RESPAWN_DELAY_TICKS = 20; // 1 second

    public void tick() {
        if (paused) return;

        // Auto-respawn: when dead or dying, wait a cooldown then reset
        if (bot.isDeadOrDying() || bot.isRemoved()) {
            if (respawnCooldown <= 0) {
                respawnCooldown = RESPAWN_DELAY_TICKS;
            }
            // Prevent entity removal while waiting to respawn (vanilla removes after 20 ticks)
            bot.deathTime = 0;
            respawnCooldown--;
            if (respawnCooldown <= 0) {
                respawnBot();
            }
            return;
        }

        // Ensure bot is hittable (not globally invulnerable)
        bot.setInvulnerable(false);
        bot.setClientLoaded(true);
        // Note: invulnerableTime is NOT cleared here — it provides i-frame protection
        // between hits and is naturally decremented by LivingEntity.tick() each tick

        // Arena boundary enforcement: teleport back if too far from spawn
        if (hasSpawnAnchor) {
            double dx = bot.getX() - spawnX;
            double dz = bot.getZ() - spawnZ;
            double horizDist = Math.sqrt(dx * dx + dz * dz);
            if (horizDist > arenaRadius || bot.getY() < spawnY - 5) {
                bot.snapTo(spawnX, spawnY, spawnZ, bot.getYRot(), bot.getXRot());
                bot.setDeltaMovement(net.minecraft.world.phys.Vec3.ZERO);
                bot.fallDistance = 0;
                target = null; // re-acquire target after teleport
            }
        }

        // Recording mode: random actions + physics logging, no model needed
        if (recordingMode && physicsRecorder != null) {
            tickRecording();
            return;
        }

        // Advance per-tick observation state (combat accumulators, tick counter)
        obsBuilder.onTick(name);

        try {
            // Find nearest target if we don't have one or it's dead
            if (target == null || !target.isAlive()) {
                target = findNearestEnemy();
            }

            // No enemies nearby — stand still, don't run inference
            if (target == null) {
                bot.xxa = 0;
                bot.zza = 0;
                bot.setSprinting(false);
                return;
            }

            // 1. Build observations
            ServerPlayer targetPlayer = (target instanceof ServerPlayer sp) ? sp : null;
            ObservationBuilder.Observation obs = obsBuilder.build(bot, targetPlayer);

            // 2. Run model inference
            // Save pre-inference hidden state for experience recording
            float[] preHidden = hidden.clone();
            ModelManager.InferenceResult result = modelManager.infer(
                    predictor, ndManager,
                    obs.selfState(),
                    obs.entityFeatures(),
                    obs.entityMask(),
                    obs.combatCtx(),
                    obs.sigilState(),
                    obs.envState(),
                    hidden
            );

            // Update hidden state
            hidden = result.newHidden;

            // 3. Build action mask
            List<LivingEntity> nearby = getNearbyEntities();
            float[] actionMask = actionExecutor.buildActionMask(bot, nearby.size());

            // 4. Sample actions from probabilities with mask
            int[] actions = sampleActions(result.actionProbs, actionMask);

            // 4.5. Anti-idle: force approach when stalemate detected
            float currentHealth = bot.getHealth() + bot.getAbsorptionAmount();
            float targetHealth = target instanceof LivingEntity le ? le.getHealth() + le.getAbsorptionAmount() : 0;
            float combinedHealth = currentHealth + targetHealth;
            if (lastCombatHealth < 0) lastCombatHealth = combinedHealth;
            if (Math.abs(combinedHealth - lastCombatHealth) > 0.5f) {
                idleTicks = 0;
                lastCombatHealth = combinedHealth;
            } else {
                idleTicks++;
            }
            if (idleTicks > IDLE_THRESHOLD && target != null) {
                forceEngageTicks = FORCE_ENGAGE_DURATION;
                idleTicks = 0;
            }
            if (forceEngageTicks > 0) {
                forceEngageTicks--;
                // Override actions: sprint toward target and attack
                actions[ActionSpace.ACT_FORWARD] = 1;
                actions[ActionSpace.ACT_BACKWARD] = 0;
                actions[ActionSpace.ACT_SPRINT] = 1;
                actions[ActionSpace.ACT_FACE_TARGET] = 1;
                actions[ActionSpace.ACT_ATTACK] = 1;
                actions[ActionSpace.ACT_JUMP] = bot.onGround() && ThreadLocalRandom.current().nextFloat() < 0.15f ? 1 : 0;
            }

            // 5. Execute actions (with reward callbacks if training)
            LivingEntity newTarget = actionExecutor.execute(
                    bot, actions, target, nearby, rewardComputer, name);
            if (newTarget != null) {
                target = newTarget;
            }

            // 6. Collect experience if training
            if (trainingEnabled && experienceBuffer != null && rewardComputer != null) {
                float reward = rewardComputer.consumeReward(name);
                boolean done = !bot.isAlive();

                experienceBuffer.add(new ExperienceBuffer.Experience(
                        obs.selfState(),
                        obs.entityFeatures(),
                        obs.entityMask(),
                        obs.combatCtx(),
                        obs.sigilState(),
                        obs.envState(),
                        actions,
                        result.actionProbs,
                        result.value,
                        reward,
                        done,
                        preHidden // use pre-inference hidden, not post-update
                ));
            }

            // Update potential shaping
            if (rewardComputer != null && target != null) {
                float dist = (float) bot.distanceTo(target);
                rewardComputer.updatePotentialShaping(
                        name,
                        bot.getHealth(),
                        target instanceof LivingEntity le ? le.getHealth() : 20f,
                        dist
                );
            }

        } catch (Exception e) {
            LOG.warning("BotBrain tick error for '" + name + "': " + e.getMessage());
        }
    }

    /**
     * Sample multi-binary actions from probabilities, applying action mask.
     */
    private int[] sampleActions(float[] actionProbs, float[] actionMask) {
        int[] actions = new int[ActionSpace.NUM_ACTIONS];
        ThreadLocalRandom rng = ThreadLocalRandom.current();
        for (int i = 0; i < ActionSpace.NUM_ACTIONS; i++) {
            if (actionMask[i] > 0f && rng.nextFloat() < actionProbs[i]) {
                actions[i] = 1;
            }
        }
        return actions;
    }

    /**
     * Find the nearest living enemy entity within range.
     */
    private @Nullable LivingEntity findNearestEnemy() {
        List<LivingEntity> nearby = getNearbyEntities();
        // Filter out allies — they're visible in observations but not valid targets
        for (LivingEntity e : nearby) {
            if (!allyUUIDs.contains(e.getUUID())) {
                return e;
            }
        }
        return null;
    }

    /**
     * Get nearby living entities sorted by distance.
     */
    private List<LivingEntity> getNearbyEntities() {
        AABB box = bot.getBoundingBox().inflate(NEARBY_RANGE);
        return bot.level().getEntitiesOfClass(LivingEntity.class, box, e -> e != bot && e.isAlive())
                .stream()
                .sorted(Comparator.comparingDouble(e -> e.distanceToSqr(bot)))
                .limit(ObservationSpace.MAX_ENTITIES)
                .toList();
    }

    /**
     * Reset for a new episode - zero hidden state, clear target.
     */
    public void resetEpisode() {
        hidden = new float[ObservationSpace.HIDDEN_DIM];
        target = null;
        idleTicks = 0;
        forceEngageTicks = 0;
        lastCombatHealth = -1;
    }

    /**
     * Respawn the bot: restore health, teleport to spawn, reset episode state.
     * Resets NMS death state so the entity is considered alive again.
     */
    private void respawnBot() {
        // Reset NMS death state (dead is protected, use reflection)
        try {
            java.lang.reflect.Field deadField = net.minecraft.world.entity.LivingEntity.class.getDeclaredField("dead");
            deadField.setAccessible(true);
            deadField.setBoolean(bot, false);
        } catch (Exception ignored) {}
        bot.deathTime = 0;

        // Restore health
        bot.setHealth(bot.getMaxHealth());
        bot.setAbsorptionAmount(0);

        // Clear damage/invulnerability state
        bot.invulnerableTime = 0;
        bot.hurtTime = 0;
        bot.hurtDuration = 0;
        bot.fallDistance = 0;
        bot.setDeltaMovement(net.minecraft.world.phys.Vec3.ZERO);

        // Teleport to spawn anchor
        if (hasSpawnAnchor) {
            bot.snapTo(spawnX, spawnY, spawnZ, bot.getYRot(), bot.getXRot());
        }

        // Reapply kit: golden apples are consumed during fights
        org.bukkit.inventory.PlayerInventory inv = bot.getBukkitEntity().getInventory();
        inv.clear();
        inv.setItem(0, new org.bukkit.inventory.ItemStack(org.bukkit.Material.DIAMOND_SWORD, 1));
        inv.setItem(1, new org.bukkit.inventory.ItemStack(org.bukkit.Material.GOLDEN_APPLE, 64));
        inv.setHelmet(new org.bukkit.inventory.ItemStack(org.bukkit.Material.DIAMOND_HELMET, 1));
        inv.setChestplate(new org.bukkit.inventory.ItemStack(org.bukkit.Material.DIAMOND_CHESTPLATE, 1));
        inv.setLeggings(new org.bukkit.inventory.ItemStack(org.bukkit.Material.DIAMOND_LEGGINGS, 1));
        inv.setBoots(new org.bukkit.inventory.ItemStack(org.bukkit.Material.DIAMOND_BOOTS, 1));
        inv.setHeldItemSlot(0);

        // Reset AI state for new episode
        resetEpisode();
    }

    public void setTrainingComponents(ExperienceBuffer buffer, RewardComputer rewards) {
        this.experienceBuffer = buffer;
        this.rewardComputer = rewards;
    }

    public void setTrainingEnabled(boolean enabled) {
        this.trainingEnabled = enabled;
    }

    public String getName() {
        return name;
    }

    public ServerPlayer getServerPlayer() {
        return bot;
    }

    public @Nullable LivingEntity getTarget() {
        return target;
    }

    public void setTarget(@Nullable LivingEntity target) {
        this.target = target;
    }

    public void setPaused(boolean paused) {
        this.paused = paused;
    }

    public boolean isPaused() {
        return paused;
    }

    public void setAllies(Set<UUID> allies) {
        this.allyUUIDs = allies;
    }

    /**
     * Set the bot's spawn anchor point. If the bot moves more than arenaRadius
     * blocks away (horizontal) or falls below spawnY - 5, it gets teleported back.
     */
    public void setSpawnAnchor(double x, double y, double z, double radius) {
        this.spawnX = x;
        this.spawnY = y;
        this.spawnZ = z;
        this.arenaRadius = radius;
        this.hasSpawnAnchor = true;
    }

    // ----------------------------------------------------------------
    //  Physics recording mode
    // ----------------------------------------------------------------

    /**
     * Enable recording mode: random actions, no model inference, logs physics.
     */
    public void setRecordingMode(PhysicsRecorder recorder) {
        this.physicsRecorder = recorder;
        this.recordingMode = true;
    }

    public boolean isRecordingMode() {
        return recordingMode;
    }

    public @Nullable PhysicsRecorder getPhysicsRecorder() {
        return physicsRecorder;
    }

    /**
     * Recording tick: random actions + physics data logging.
     * No model inference needed — actions are sampled from fixed probabilities.
     *
     * Uses buffered recording: this tick's pre-state becomes the previous
     * tick's post-state (since doTick/physics runs between tick() calls).
     */
    private void tickRecording() {
        try {
            // Find target for combat recording
            if (target == null || !target.isAlive()) {
                target = findNearestEnemy();
            }

            // Capture current state (before this tick's actions execute)
            PhysicsRecorder.StateSnapshot currentState = PhysicsRecorder.StateSnapshot.capture(bot);
            float blockFriction = PhysicsRecorder.getBlockFriction(bot);

            // Direct damage-taken tracking via health comparison
            float curHealth = bot.getHealth() + bot.getAbsorptionAmount();
            if (prevBotHealth >= 0 && curHealth < prevBotHealth) {
                float taken = prevBotHealth - curHealth;
                net.minecraft.world.phys.Vec3 vel = bot.getDeltaMovement();
                physicsRecorder.onDamageTaken(taken, vel.x, vel.y, vel.z, false);
            }
            prevBotHealth = curHealth;

            // Build action mask and sample random actions
            List<LivingEntity> nearby = getNearbyEntities();
            float[] actionMask = actionExecutor.buildActionMask(bot, nearby.size());
            int[] actions = sampleActions(RANDOM_PROBS, actionMask);

            // Execute actions (sets xxa/zza for doTick to process between ticks)
            LivingEntity newTarget = actionExecutor.execute(bot, actions, target, nearby);
            if (newTarget != null) {
                target = newTarget;
            }

            // Record tick (buffered: this tick's pre becomes last tick's post)
            physicsRecorder.recordTick(name, currentState, actions, target, bot, blockFriction);

        } catch (Exception e) {
            LOG.warning("Recording tick error for '" + name + "': " + e.getMessage());
        }
    }

    public void close() {
        try {
            if (physicsRecorder != null) {
                physicsRecorder.flush(bot);
                physicsRecorder.close();
            }
            predictor.close();
            ndManager.close();
        } catch (Exception e) {
            LOG.warning("Error closing BotBrain '" + name + "': " + e.getMessage());
        }
    }
}
