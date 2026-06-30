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
import org.bukkit.entity.Player;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.logging.Logger;

/**
 * Per-bot tick loop: observe → infer → act → collect experience.
 *
 * Supports three modes:
 * - NEURAL: full neural net inference (original behavior)
 * - RULE: rule-based combat engine, no model needed
 * - HYBRID: rule combat + neural sigil timing
 */
public class BotBrain {

    private static final Logger LOG = Logger.getLogger("MinimalAI");
    private static final double NEARBY_RANGE = 30.0;

    public enum BrainMode { NEURAL, RULE, HYBRID }

    private final String name;
    private final ServerPlayer bot;
    private final BrainMode mode;
    private final ObservationBuilder obsBuilder;
    private final ActionExecutor actionExecutor;
    private final @Nullable ArcaneSigilsAPI sigilsApi;

    // NEURAL mode: full model inference
    private final @Nullable Predictor<NDList, NDList> predictor;
    private final @Nullable NDManager ndManager;
    private final @Nullable ModelManager modelManager;

    // RULE/HYBRID mode: rule-based combat
    private @Nullable RuleCombatEngine ruleEngine;

    // HYBRID mode: separate sigil net
    private @Nullable Predictor<NDList, NDList> sigilPredictor;
    private @Nullable NDManager sigilNdManager;
    private @Nullable ModelManager sigilModelManager;

    // GRU hidden state - zeroed on episode reset (NEURAL mode only)
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
    private double arenaRadius = 30.0;
    private boolean hasSpawnAnchor = false;

    // Engagement: force approach if idle too long (prevents sim-to-real stalemates)
    private int idleTicks = 0;
    private int forceEngageTicks = 0;
    private float lastCombatHealth = -1;
    private static final int IDLE_THRESHOLD = 60;
    private static final int FORCE_ENGAGE_DURATION = 100;

    // Engage auto-attack reach (per-bot skill level, default 3.0 = max range)
    private double attackReach = 3.0;

    // Death callback (fired once when bot dies, triggers despawn)
    private @Nullable Runnable onDeath;
    private boolean deathFired = false;

    // Physics recording mode (random actions, no model needed)
    private @Nullable PhysicsRecorder physicsRecorder;
    private boolean recordingMode = false;
    private float prevBotHealth = -1;

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

    // Kill statistics
    private int deaths = 0;
    private int ticksAlive = 0;
    private static final int STATS_LOG_INTERVAL = 6000;

    // Sigil observation constants
    private static final int SIGIL_OBS_DIM = 16;
    private static final int NUM_ACTIVE_SIGILS = 4;

    // Active ability slots: neural net output [0-3] → ArcaneSigils bind slot
    // Based on config.yml registration order:
    //   0=ancient_crown(passive), 1=kings_brace, 2=cleopatra, 3=quick_sand,
    //   4=divine_intervention(passive), 5=niles_grace
    // Net output: [0]=brace, [1]=cleopatra, [2]=quicksand, [3]=grace
    private static final int[] ACTIVE_ABILITY_SLOTS = {1, 2, 3, 5};

    /**
     * NEURAL mode constructor — full model inference (original behavior).
     */
    public BotBrain(String name,
                    ServerPlayer bot,
                    ModelManager modelManager,
                    String modelName,
                    ObservationBuilder obsBuilder,
                    ActionExecutor actionExecutor,
                    @Nullable ArcaneSigilsAPI sigilsApi) {
        this.name = name;
        this.bot = bot;
        this.mode = BrainMode.NEURAL;
        this.modelManager = modelManager;
        this.predictor = modelManager.createPredictor(modelName);
        this.ndManager = NDManager.newBaseManager();
        this.obsBuilder = obsBuilder;
        this.actionExecutor = actionExecutor;
        this.sigilsApi = sigilsApi;
        this.ruleEngine = null;
        this.sigilPredictor = null;
        this.sigilNdManager = null;
        this.sigilModelManager = null;
    }

    /**
     * RULE or HYBRID mode constructor — rule combat engine, optional sigil net.
     */
    public BotBrain(String name,
                    ServerPlayer bot,
                    RuleCombatEngine ruleEngine,
                    @Nullable ModelManager sigilModelManager,
                    @Nullable String sigilModelName,
                    ObservationBuilder obsBuilder,
                    ActionExecutor actionExecutor,
                    @Nullable ArcaneSigilsAPI sigilsApi) {
        this.name = name;
        this.bot = bot;
        this.ruleEngine = ruleEngine;
        this.obsBuilder = obsBuilder;
        this.actionExecutor = actionExecutor;
        this.sigilsApi = sigilsApi;

        // No main model for rule/hybrid
        this.modelManager = null;
        this.predictor = null;
        this.ndManager = null;

        // Sigil model for hybrid mode
        if (sigilModelManager != null && sigilModelName != null) {
            this.mode = BrainMode.HYBRID;
            this.sigilModelManager = sigilModelManager;
            this.sigilPredictor = sigilModelManager.createPredictor(sigilModelName);
            this.sigilNdManager = NDManager.newBaseManager();
        } else {
            this.mode = BrainMode.RULE;
            this.sigilModelManager = null;
            this.sigilPredictor = null;
            this.sigilNdManager = null;
        }

        // Set attack reach from rule engine difficulty
        this.attackReach = ruleEngine.getAttackReach();
    }

    /**
     * Execute one tick of the bot's brain.
     * Routes to the appropriate tick method based on mode.
     */
    public void tick() {
        if (paused) return;

        // Death detection
        if (bot.isDeadOrDying() || bot.isRemoved()) {
            if (!deathFired) {
                deathFired = true;
                deaths++;
                paused = true;
                LOG.info(name + " died (death #" + deaths + ").");
                if (onDeath != null) onDeath.run();
            }
            return;
        }

        ticksAlive++;
        if (ticksAlive % STATS_LOG_INTERVAL == 0) {
            LOG.info(String.format("[%s] Alive %d min | Deaths: %d | HP: %.1f | Mode: %s | Target: %s",
                name, ticksAlive / 1200, deaths, bot.getHealth(), mode,
                target != null ? target.getScoreboardName() : "none"));
        }

        bot.setInvulnerable(false);
        bot.setClientLoaded(true);

        // Recording mode: random actions + physics logging
        if (recordingMode && physicsRecorder != null) {
            tickRecording();
            return;
        }

        switch (mode) {
            case NEURAL -> tickNeural();
            case RULE -> tickRule();
            case HYBRID -> tickHybrid();
        }
    }

    // ----------------------------------------------------------------
    //  NEURAL mode tick (original behavior)
    // ----------------------------------------------------------------

    private void tickNeural() {
        obsBuilder.onTick(name);

        try {
            updateTarget();
            if (target == null) {
                if (ticksAlive % 100 == 0) LOG.info("[" + name + "] tick=" + ticksAlive + " NO TARGET, nearby=" + getNearbyEntities().size());
                bot.zza = 0; bot.xxa = 0;
                return;
            }

            ServerPlayer targetPlayer = (target instanceof ServerPlayer sp) ? sp : null;
            ObservationBuilder.Observation obs = obsBuilder.build(bot, targetPlayer);

            float[] preHidden = hidden.clone();
            ModelManager.InferenceResult result = modelManager.infer(
                    predictor, ndManager,
                    obs.selfState(), obs.entityFeatures(), obs.entityMask(),
                    obs.combatCtx(), obs.sigilState(), obs.envState(), hidden);

            hidden = result.newHidden;

            List<LivingEntity> nearby = getNearbyEntities();
            float[] actionMask = actionExecutor.buildActionMask(bot, nearby.size());

            // Zero out untrained action bits — must match training USED_BITS
            {
                boolean[] usedBits = new boolean[ActionSpace.NUM_ACTIONS];
                for (int b : new int[]{0, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 26}) usedBits[b] = true;
                for (int i = 0; i < ActionSpace.NUM_ACTIONS; i++) {
                    if (!usedBits[i]) actionMask[i] = 0f;
                }
            }

            int[] actions = sampleActions(result.actionProbs, actionMask);
            actions[ActionSpace.ACT_SPRINT] = 1;
            actions[ActionSpace.ACT_SNEAK] = 0;

            if (actions[ActionSpace.ACT_ENGAGE] == 1 && target != null) {
                double dist = bot.distanceTo(target);
                actions[ActionSpace.ACT_ATTACK] = (dist <= attackReach) ? 1 : 0;
            }

            logActions(actions);
            applyAntiIdle(actions);

            LivingEntity newTarget = actionExecutor.execute(bot, actions, target, nearby, rewardComputer, name);
            if (newTarget != null) target = newTarget;

            collectExperience(obs, actions, result, preHidden);
            updatePotentialShaping();

        } catch (Exception e) {
            LOG.warning("BotBrain tick error for '" + name + "': " + e.getMessage());
        }
    }

    // ----------------------------------------------------------------
    //  RULE mode tick
    // ----------------------------------------------------------------

    private void tickRule() {
        try {
            updateTarget();
            if (target == null) {
                if (ticksAlive % 100 == 0) LOG.info("[" + name + "] tick=" + ticksAlive + " NO TARGET");
                bot.zza = 0; bot.xxa = 0;
                return;
            }

            int[] actions = ruleEngine.generateActions(bot, target);

            logActions(actions);

            List<LivingEntity> nearby = getNearbyEntities();
            LivingEntity newTarget = actionExecutor.execute(bot, actions, target, nearby);
            if (newTarget != null) target = newTarget;

        } catch (Exception e) {
            LOG.warning("BotBrain RULE tick error for '" + name + "': " + e.getMessage());
        }
    }

    // ----------------------------------------------------------------
    //  HYBRID mode tick — rule combat + neural sigil timing
    // ----------------------------------------------------------------

    private void tickHybrid() {
        try {
            updateTarget();
            if (target == null) {
                if (ticksAlive % 100 == 0) LOG.info("[" + name + "] tick=" + ticksAlive + " NO TARGET");
                bot.zza = 0; bot.xxa = 0;
                return;
            }

            // Rule engine generates combat actions (sigil bits left at 0)
            int[] actions = ruleEngine.generateActions(bot, target);

            // Neural net fills sigil bits if available
            if (sigilPredictor != null && sigilsApi != null && sigilsApi.isAvailable()) {
                float[] sigilObs = buildSigilObservation(bot, target);
                float[] sigilMask = buildSigilActionMask(bot);
                float[] sigilProbs = inferSigilNet(sigilObs, sigilMask);

                if (sigilProbs != null) {
                    ThreadLocalRandom rng = ThreadLocalRandom.current();
                    for (int i = 0; i < NUM_ACTIVE_SIGILS; i++) {
                        if (sigilMask[i] > 0f && rng.nextFloat() < sigilProbs[i]) {
                            actions[ActionSpace.ACT_SIGIL_0 + i] = 1;
                        }
                    }
                }
            }

            logActions(actions);

            List<LivingEntity> nearby = getNearbyEntities();
            LivingEntity newTarget = actionExecutor.execute(bot, actions, target, nearby);
            if (newTarget != null) target = newTarget;

        } catch (Exception e) {
            LOG.warning("BotBrain HYBRID tick error for '" + name + "': " + e.getMessage());
        }
    }

    // ----------------------------------------------------------------
    //  Sigil observation + inference (HYBRID mode)
    // ----------------------------------------------------------------

    /**
     * Build a 20-float observation for the sigil neural net.
     */
    private float[] buildSigilObservation(ServerPlayer bot, LivingEntity target) {
        float[] obs = new float[SIGIL_OBS_DIM];
        Player p = bot.getBukkitEntity();

        // [0-3] Health state
        obs[0] = bot.getHealth() / 24f;  // max 24 with extra_padding
        obs[1] = bot.getAbsorptionAmount() / 20f;
        obs[2] = target.getHealth() / 24f;
        obs[3] = target.getAbsorptionAmount() / 20f;

        // [4] Brace charges (normalized to requirement)
        obs[4] = Math.min(sigilsApi.getKingsBraceCharges(p) / 30f, 3f);

        // [5-8] Cooldown progress for 4 active abilities (0=ready, 1=just used)
        for (int i = 0; i < NUM_ACTIVE_SIGILS; i++) {
            obs[5 + i] = (float) sigilsApi.getCooldownProgress(p, ACTIVE_ABILITY_SLOTS[i]);
        }

        // [9-11] Own active buff indicators
        obs[9] = bot.hasEffect(net.minecraft.world.effect.MobEffects.RESISTANCE) ? 1f : 0f;  // brace active
        obs[10] = bot.hasEffect(net.minecraft.world.effect.MobEffects.REGENERATION) ? 1f : 0f;  // grace active
        obs[11] = 0f; // sand timer (approximated — no direct API)

        // [12-14] Target state
        if (target instanceof ServerPlayer tp) {
            Player tp_bukkit = tp.getBukkitEntity();
            obs[12] = sigilsApi.isMarked(tp_bukkit, p) ? 1f : 0f;  // our cleo on them
            obs[13] = tp.hasEffect(net.minecraft.world.effect.MobEffects.RESISTANCE) ? 1f : 0f;  // target has brace
            obs[14] = tp.hasEffect(net.minecraft.world.effect.MobEffects.REGENERATION) ? 1f : 0f;  // target has grace
        }

        // [15] Episode progress
        obs[15] = Math.min(ticksAlive / 1800f, 1f);

        return obs;
    }

    /**
     * Build a 7-float mask for active sigil abilities (1=ready, 0=on cooldown).
     */
    private float[] buildSigilActionMask(ServerPlayer bot) {
        float[] mask = new float[NUM_ACTIVE_SIGILS];
        Player p = bot.getBukkitEntity();

        for (int i = 0; i < NUM_ACTIVE_SIGILS; i++) {
            mask[i] = sigilsApi.isSigilReady(p, ACTIVE_ABILITY_SLOTS[i]) ? 1f : 0f;
        }

        return mask;
    }

    /**
     * Run the sigil neural net: obs(20) + mask(7) → probs(7).
     */
    private @Nullable float[] inferSigilNet(float[] obs, float[] mask) {
        if (sigilPredictor == null || sigilNdManager == null || sigilModelManager == null) return null;

        try {
            return sigilModelManager.inferSigil(sigilPredictor, sigilNdManager, obs, mask);
        } catch (Exception e) {
            LOG.warning("Sigil inference error for '" + name + "': " + e.getMessage());
            return null;
        }
    }

    // ----------------------------------------------------------------
    //  Shared helpers
    // ----------------------------------------------------------------

    private void updateTarget() {
        LivingEntity prevTarget = target;
        if (target == null || !target.isAlive() || target.distanceToSqr(bot) > NEARBY_RANGE * NEARBY_RANGE) {
            target = findNearestEnemy();
        }
        if (target != null && target != prevTarget) {
            hidden = new float[ObservationSpace.HIDDEN_DIM];
            if (ruleEngine != null) ruleEngine.reset();
        }
    }

    private void logActions(int[] actions) {
        if (ticksAlive % 100 == 0 && target != null) {
            LOG.info("[" + name + "] tick=" + ticksAlive + " mode=" + mode
                + " target=" + target.getScoreboardName()
                + " dist=" + String.format("%.1f", bot.distanceTo(target))
                + " hp=" + String.format("%.1f", bot.getHealth() + bot.getAbsorptionAmount())
                + (ruleEngine != null ? " phase=" + ruleEngine.getCurrentPhase() : "")
                + " actions=" + ActionExecutor.describeActions(actions));
        }
    }

    private void applyAntiIdle(int[] actions) {
        if (target == null) return;
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
        if (idleTicks > IDLE_THRESHOLD) {
            forceEngageTicks = FORCE_ENGAGE_DURATION;
            idleTicks = 0;
        }
        if (forceEngageTicks > 0) {
            forceEngageTicks--;
            actions[ActionSpace.ACT_FORWARD] = 1;
            actions[ActionSpace.ACT_BACKWARD] = 0;
            actions[ActionSpace.ACT_SPRINT] = 1;
            actions[ActionSpace.ACT_ATTACK] = 1;
            actions[ActionSpace.ACT_JUMP] = bot.onGround() && ThreadLocalRandom.current().nextFloat() < 0.15f ? 1 : 0;
        }
    }

    private void collectExperience(ObservationBuilder.Observation obs, int[] actions,
                                    ModelManager.InferenceResult result, float[] preHidden) {
        if (trainingEnabled && experienceBuffer != null && rewardComputer != null) {
            float reward = rewardComputer.consumeReward(name);
            boolean done = !bot.isAlive();

            experienceBuffer.add(new ExperienceBuffer.Experience(
                    obs.selfState(), obs.entityFeatures(), obs.entityMask(),
                    obs.combatCtx(), obs.sigilState(), obs.envState(),
                    actions, result.actionProbs, result.value,
                    reward, done, preHidden));
        }
    }

    private void updatePotentialShaping() {
        if (rewardComputer != null && target != null) {
            float dist = (float) bot.distanceTo(target);
            rewardComputer.updatePotentialShaping(name, bot.getHealth(),
                    target instanceof LivingEntity le ? le.getHealth() : 20f, dist);
        }
    }

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

    private @Nullable LivingEntity findNearestEnemy() {
        List<LivingEntity> nearby = getNearbyEntities();
        for (LivingEntity e : nearby) {
            if (!allyUUIDs.contains(e.getUUID())) {
                return e;
            }
        }
        return null;
    }

    private List<LivingEntity> getNearbyEntities() {
        AABB box = bot.getBoundingBox().inflate(NEARBY_RANGE);
        return bot.level().getEntitiesOfClass(LivingEntity.class, box, e -> {
                    if (e == bot || !e.isAlive()) return false;
                    if (e instanceof ServerPlayer sp && (sp.gameMode.isSurvival() == false && sp.gameMode.getGameModeForPlayer() != net.minecraft.world.level.GameType.ADVENTURE))
                        return false;
                    return true;
                })
                .stream()
                .sorted(Comparator.comparingDouble(e -> e.distanceToSqr(bot)))
                .limit(ObservationSpace.MAX_ENTITIES)
                .toList();
    }

    // ----------------------------------------------------------------
    //  Recording mode
    // ----------------------------------------------------------------

    private void tickRecording() {
        try {
            if (target == null || !target.isAlive()) {
                target = findNearestEnemy();
            }

            PhysicsRecorder.StateSnapshot currentState = PhysicsRecorder.StateSnapshot.capture(bot);
            float blockFriction = PhysicsRecorder.getBlockFriction(bot);

            float curHealth = bot.getHealth() + bot.getAbsorptionAmount();
            if (prevBotHealth >= 0 && curHealth < prevBotHealth) {
                float taken = prevBotHealth - curHealth;
                net.minecraft.world.phys.Vec3 vel = bot.getDeltaMovement();
                physicsRecorder.onDamageTaken(taken, vel.x, vel.y, vel.z, false);
            }
            prevBotHealth = curHealth;

            List<LivingEntity> nearby = getNearbyEntities();
            float[] actionMask = actionExecutor.buildActionMask(bot, nearby.size());
            int[] actions = sampleActions(RANDOM_PROBS, actionMask);

            LivingEntity newTarget = actionExecutor.execute(bot, actions, target, nearby);
            if (newTarget != null) target = newTarget;

            physicsRecorder.recordTick(name, currentState, actions, target, bot, blockFriction);

        } catch (Exception e) {
            LOG.warning("Recording tick error for '" + name + "': " + e.getMessage());
        }
    }

    // ----------------------------------------------------------------
    //  Lifecycle
    // ----------------------------------------------------------------

    public void resetEpisode() {
        hidden = new float[ObservationSpace.HIDDEN_DIM];
        target = null;
        idleTicks = 0;
        forceEngageTicks = 0;
        lastCombatHealth = -1;
        if (ruleEngine != null) ruleEngine.reset();
    }

    public void close() {
        try {
            if (physicsRecorder != null) {
                physicsRecorder.flush(bot);
                physicsRecorder.close();
            }
            if (predictor != null) predictor.close();
            if (ndManager != null) ndManager.close();
            if (sigilPredictor != null) sigilPredictor.close();
            if (sigilNdManager != null) sigilNdManager.close();
        } catch (Exception e) {
            LOG.warning("Error closing BotBrain '" + name + "': " + e.getMessage());
        }
    }

    // ----------------------------------------------------------------
    //  Getters / Setters
    // ----------------------------------------------------------------

    public void setTrainingComponents(ExperienceBuffer buffer, RewardComputer rewards) {
        this.experienceBuffer = buffer;
        this.rewardComputer = rewards;
    }

    public void setTrainingEnabled(boolean enabled) { this.trainingEnabled = enabled; }
    public String getName() { return name; }
    public BrainMode getMode() { return mode; }
    public ServerPlayer getServerPlayer() { return bot; }
    public @Nullable LivingEntity getTarget() { return target; }
    public void setTarget(@Nullable LivingEntity target) { this.target = target; }
    public void setPaused(boolean paused) { this.paused = paused; }
    public boolean isPaused() { return paused; }
    public int getDeaths() { return deaths; }
    public void setAllies(Set<UUID> allies) { this.allyUUIDs = allies; }

    public void setAttackReach(double reach) {
        this.attackReach = Math.max(0.5, Math.min(reach, 3.0));
    }

    public double getAttackReach() { return attackReach; }

    public void setOnDeath(@Nullable Runnable onDeath) { this.onDeath = onDeath; }

    public void setSpawnAnchor(double x, double y, double z, double radius) {
        this.spawnX = x;
        this.spawnY = y;
        this.spawnZ = z;
        this.arenaRadius = radius;
        this.hasSpawnAnchor = true;
    }

    public void setRecordingMode(PhysicsRecorder recorder) {
        this.physicsRecorder = recorder;
        this.recordingMode = true;
    }

    public boolean isRecordingMode() { return recordingMode; }
    public @Nullable PhysicsRecorder getPhysicsRecorder() { return physicsRecorder; }
}
