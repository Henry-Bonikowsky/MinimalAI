package com.minimalai.bot;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.minimalai.ai.*;
import com.minimalai.integration.ArcaneSigilsAPI;
import com.minimalai.training.ExperienceBuffer;
import com.minimalai.training.RewardComputer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.phys.AABB;
import org.jetbrains.annotations.Nullable;

import java.util.Comparator;
import java.util.List;
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
     * Called from the server tick scheduler.
     */
    public void tick() {
        try {
            // Find nearest target if we don't have one or it's dead
            if (target == null || !target.isAlive()) {
                target = findNearestEnemy();
            }

            // 1. Build observations
            ServerPlayer targetPlayer = (target instanceof ServerPlayer sp) ? sp : null;
            ObservationBuilder.Observation obs = obsBuilder.build(bot, targetPlayer);

            // 2. Run model inference
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

            // 5. Execute actions
            LivingEntity newTarget = actionExecutor.execute(bot, actions, target, nearby);
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
                        hidden
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
        return nearby.isEmpty() ? null : nearby.get(0);
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

    public void close() {
        try {
            predictor.close();
            ndManager.close();
        } catch (Exception e) {
            LOG.warning("Error closing BotBrain '" + name + "': " + e.getMessage());
        }
    }
}
