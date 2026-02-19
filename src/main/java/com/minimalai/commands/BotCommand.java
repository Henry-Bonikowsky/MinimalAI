package com.minimalai.commands;

import com.minimalai.ai.*;
import com.minimalai.bot.BotBrain;
import com.minimalai.bot.FakePlayerManager;
import com.minimalai.bot.FakePlayerManager.BotContext;
import com.minimalai.bot.KitManager;
import com.minimalai.integration.ArcaneSigilsAPI;
import com.minimalai.training.EpisodeManager;
import com.minimalai.training.ExperienceBuffer;
import com.minimalai.training.RewardComputer;
import org.bukkit.Location;
import org.bukkit.World;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Internal service managing BotBrain lifecycle, ticking, and spawning.
 * Command handling is done by {@link MaiCommand}.
 */
public class BotCommand {

    private static final Logger LOG = Logger.getLogger("MinimalAI");

    private final FakePlayerManager botManager;
    private final ModelManager modelManager;
    private final ObservationBuilder obsBuilder;
    private final ActionExecutor actionExecutor;
    private final @Nullable ArcaneSigilsAPI sigilsApi;
    private final @Nullable ExperienceBuffer experienceBuffer;
    private final @Nullable RewardComputer rewardComputer;
    private final @Nullable EpisodeManager episodeManager;
    private final KitManager kitManager;

    private final Map<String, BotBrain> brains = new ConcurrentHashMap<>();
    private @Nullable MaiCommand maiCommand;

    public BotCommand(FakePlayerManager botManager,
                      ModelManager modelManager,
                      ObservationBuilder obsBuilder,
                      ActionExecutor actionExecutor,
                      @Nullable ArcaneSigilsAPI sigilsApi,
                      @Nullable ExperienceBuffer experienceBuffer,
                      @Nullable RewardComputer rewardComputer,
                      @Nullable EpisodeManager episodeManager,
                      KitManager kitManager) {
        this.botManager = botManager;
        this.modelManager = modelManager;
        this.obsBuilder = obsBuilder;
        this.actionExecutor = actionExecutor;
        this.sigilsApi = sigilsApi;
        this.experienceBuffer = experienceBuffer;
        this.rewardComputer = rewardComputer;
        this.episodeManager = episodeManager;
        this.kitManager = kitManager;
    }

    public void setMaiCommand(MaiCommand cmd) {
        this.maiCommand = cmd;
    }

    // ----------------------------------------------------------------
    //  Spawning
    // ----------------------------------------------------------------

    /**
     * Spawn a bot with a brain. Returns the BotContext or null on failure.
     * Handles: FakePlayer creation, BotBrain setup, kit application, training auto-enable.
     */
    public @Nullable BotContext spawnBot(World world, Location location,
                                         @Nullable String name,
                                         @Nullable String modelName,
                                         @Nullable String kitName) {
        if (modelManager.listModels().isEmpty()) return null;

        BotContext ctx = botManager.spawn(world, location, name);
        if (ctx == null) return null;

        if (kitName != null) {
            kitManager.applyKit(ctx.serverPlayer(), kitName);
        }

        try {
            String model = modelName != null ? modelName : modelManager.listModels().get(0);
            BotBrain brain = new BotBrain(ctx.name(), ctx.serverPlayer(), modelManager, model,
                    obsBuilder, actionExecutor, sigilsApi);

            if (experienceBuffer != null && rewardComputer != null) {
                brain.setTrainingComponents(experienceBuffer, rewardComputer);
                rewardComputer.registerBot(ctx.name());
            }

            if (maiCommand != null && maiCommand.isTrainingEnabled()) {
                brain.setTrainingEnabled(true);
                if (episodeManager != null) {
                    episodeManager.startEpisode(ctx.name());
                }
            }

            brains.put(ctx.name(), brain);
        } catch (Exception e) {
            LOG.warning("Brain creation failed for " + ctx.name() + ": " + e.getMessage());
        }

        return ctx;
    }

    // ----------------------------------------------------------------
    //  Brain management
    // ----------------------------------------------------------------

    public Map<String, BotBrain> getBrains() {
        return Collections.unmodifiableMap(brains);
    }

    public void registerBrain(String name, BotBrain brain) {
        brains.put(name, brain);
    }

    public void removeBrain(String name) {
        BotBrain brain = brains.remove(name);
        if (brain != null) {
            if (rewardComputer != null) rewardComputer.unregisterBot(name);
            brain.close();
        }
    }

    // ----------------------------------------------------------------
    //  Tick loop
    // ----------------------------------------------------------------

    /** Called by MinimalAIPlugin tick loop. */
    public void tickAll() {
        boolean training = maiCommand != null && maiCommand.isTrainingEnabled();
        for (BotBrain brain : brains.values()) {
            brain.tick();
            if (training && episodeManager != null && episodeManager.isActive(brain.getName())) {
                episodeManager.tick(brain.getName());
            }
        }
    }

    // ----------------------------------------------------------------
    //  Cleanup
    // ----------------------------------------------------------------

    /** Called by MinimalAIPlugin on disable. */
    public void closeAll() {
        for (BotBrain brain : brains.values()) {
            if (rewardComputer != null) rewardComputer.unregisterBot(brain.getName());
            brain.close();
        }
        brains.clear();
    }

    // ----------------------------------------------------------------
    //  Getters
    // ----------------------------------------------------------------

    public FakePlayerManager getBotManager() { return botManager; }
    public ModelManager getModelManager() { return modelManager; }
    public KitManager getKitManager() { return kitManager; }
}
