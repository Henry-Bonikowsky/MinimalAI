package com.minimalai.commands;

import com.minimalai.ai.*;
import com.minimalai.bot.BotBrain;
import com.minimalai.bot.BotBrain.BrainMode;
import com.minimalai.bot.FakePlayerManager;
import com.minimalai.bot.FakePlayerManager.BotContext;
import com.minimalai.bot.KitManager;
import com.minimalai.integration.ArcaneSigilsAPI;
import com.minimalai.training.EpisodeManager;
import com.minimalai.training.ExperienceBuffer;
import com.minimalai.training.RewardComputer;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;
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
    private final List<String> defaultSigils;
    private final double defaultAttackReach;

    // Config-driven defaults
    private BrainMode defaultMode = BrainMode.NEURAL;
    private int defaultDifficulty = 3;
    private @Nullable String sigilModelName = null;

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
                      KitManager kitManager,
                      List<String> defaultSigils,
                      double defaultAttackReach) {
        this.botManager = botManager;
        this.modelManager = modelManager;
        this.obsBuilder = obsBuilder;
        this.actionExecutor = actionExecutor;
        this.sigilsApi = sigilsApi;
        this.experienceBuffer = experienceBuffer;
        this.rewardComputer = rewardComputer;
        this.episodeManager = episodeManager;
        this.kitManager = kitManager;
        this.defaultSigils = defaultSigils != null ? defaultSigils : Collections.emptyList();
        this.defaultAttackReach = defaultAttackReach;
    }

    public void setDefaultMode(String mode) {
        try {
            this.defaultMode = BrainMode.valueOf(mode.toUpperCase());
        } catch (IllegalArgumentException e) {
            LOG.warning("Unknown bot mode '" + mode + "', defaulting to NEURAL");
            this.defaultMode = BrainMode.NEURAL;
        }
    }

    public void setDefaultDifficulty(int difficulty) {
        this.defaultDifficulty = Math.max(1, Math.min(5, difficulty));
    }

    public void setSigilModelName(@Nullable String name) {
        this.sigilModelName = name;
    }

    public void setMaiCommand(MaiCommand cmd) {
        this.maiCommand = cmd;
    }

    // ----------------------------------------------------------------
    //  Spawning
    // ----------------------------------------------------------------

    /**
     * Spawn a bare bot with NO brain, NO kit, NO sigils - just the fake player.
     * Use this for debugging to test if the fake player behaves like a regular entity.
     */
    public @Nullable BotContext spawnBare(World world, Location location, @Nullable String name) {
        BotContext ctx = botManager.spawn(world, location, name);
        if (ctx == null) return null;
        LOG.info("Spawned bare bot '" + ctx.name() + "' with no brain/kit/sigils");
        return ctx;
    }

    /**
     * Spawn a bot with a brain. Returns the BotContext or null on failure.
     * Handles: FakePlayer creation, BotBrain setup, kit application, training auto-enable.
     *
     * @param modelOrMode  model name, or "rule"/"hybrid" for rule-based modes
     * @param kitName      kit name (nullable)
     * @param difficulty   difficulty tier 1-5 (only for rule/hybrid, -1 = use default)
     */
    public @Nullable BotContext spawnBot(World world, Location location,
                                         @Nullable String name,
                                         @Nullable String modelOrMode,
                                         @Nullable String kitName,
                                         int difficulty) {
        // Determine the brain mode
        BrainMode mode = defaultMode;
        String modelName = null;
        int diff = difficulty > 0 ? difficulty : defaultDifficulty;

        if (modelOrMode != null) {
            switch (modelOrMode.toLowerCase()) {
                case "rule" -> mode = BrainMode.RULE;
                case "hybrid" -> mode = BrainMode.HYBRID;
                default -> {
                    mode = BrainMode.NEURAL;
                    modelName = modelOrMode;
                }
            }
        }

        // NEURAL mode requires a model
        if (mode == BrainMode.NEURAL && modelManager.listModels().isEmpty()) {
            LOG.warning("No models available for NEURAL mode");
            return null;
        }

        BotContext ctx = botManager.spawn(world, location, name);
        if (ctx == null) return null;

        // Phase 1 (tick+2): position reassert + sigil registration
        final Location loc = location;
        var plugin = org.bukkit.Bukkit.getPluginManager().getPlugin("MinimalAI");
        org.bukkit.Bukkit.getScheduler().runTaskLater(plugin, () -> {
            ctx.serverPlayer().snapTo(loc.getX(), loc.getY(), loc.getZ(), loc.getYaw(), loc.getPitch());

            if (sigilsApi != null && sigilsApi.isAvailable()) {
                List<String> sigils;
                if (kitName != null) {
                    sigils = kitManager.loadSigils(kitName);
                } else {
                    sigils = defaultSigils;
                }
                if (sigils != null && !sigils.isEmpty()) {
                    sigilsApi.registerBotSigils(ctx.serverPlayer().getBukkitEntity(), sigils);
                }
            }
        }, 2L);

        // Phase 2 (tick+10): apply kit AFTER all plugins finish join processing
        if (kitName != null) {
            org.bukkit.Bukkit.getScheduler().runTaskLater(plugin, () -> {
                kitManager.applyKit(ctx.serverPlayer(), kitName);
            }, 10L);
        }

        try {
            BotBrain brain;

            switch (mode) {
                case RULE -> {
                    RuleCombatEngine engine = new RuleCombatEngine(diff);
                    brain = new BotBrain(ctx.name(), ctx.serverPlayer(), engine,
                            null, null, obsBuilder, actionExecutor, sigilsApi);
                    LOG.info("Spawning RULE bot '" + ctx.name() + "' difficulty=" + diff);
                }
                case HYBRID -> {
                    RuleCombatEngine engine = new RuleCombatEngine(diff);
                    // Try to load sigil model
                    ModelManager sigilMgr = null;
                    String sigilModel = this.sigilModelName;
                    if (sigilModel != null && modelManager.listModels().contains(sigilModel)) {
                        sigilMgr = modelManager;
                    }
                    brain = new BotBrain(ctx.name(), ctx.serverPlayer(), engine,
                            sigilMgr, sigilModel, obsBuilder, actionExecutor, sigilsApi);
                    LOG.info("Spawning HYBRID bot '" + ctx.name() + "' difficulty=" + diff
                            + " sigilModel=" + (sigilModel != null ? sigilModel : "none"));
                }
                default -> { // NEURAL
                    String model = modelName != null ? modelName : modelManager.listModels().get(0);
                    brain = new BotBrain(ctx.name(), ctx.serverPlayer(), modelManager, model,
                            obsBuilder, actionExecutor, sigilsApi);
                    brain.setAttackReach(defaultAttackReach);
                    LOG.info("Spawning NEURAL bot '" + ctx.name() + "' model=" + model);
                }
            }

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

            // Wire death callback
            final String botName = ctx.name();
            brain.setOnDeath(() -> {
                org.bukkit.Location deathLoc = ctx.serverPlayer().getBukkitEntity().getLocation();
                int totalDeaths = brain.getDeaths();
                org.bukkit.Bukkit.getScheduler().runTask(plugin, () -> {
                    removeBrain(botName);
                    botManager.despawn(botName);
                    org.bukkit.Bukkit.getPluginManager().callEvent(
                        new com.minimalai.bot.BotDeathEvent(botName, deathLoc, totalDeaths)
                    );
                });
            });

            brain.setSpawnAnchor(
                ctx.serverPlayer().getX(),
                ctx.serverPlayer().getY(),
                ctx.serverPlayer().getZ(),
                15.0
            );

            brains.put(ctx.name(), brain);
        } catch (Exception e) {
            LOG.warning("Brain creation failed for " + ctx.name() + ": " + e.getMessage());
        }

        return ctx;
    }

    /**
     * Backward-compatible spawn (defaults to config mode, no difficulty override).
     */
    public @Nullable BotContext spawnBot(World world, Location location,
                                         @Nullable String name,
                                         @Nullable String modelOrMode,
                                         @Nullable String kitName) {
        return spawnBot(world, location, name, modelOrMode, kitName, -1);
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
            // Unregister virtual sigils
            if (sigilsApi != null && sigilsApi.isAvailable()) {
                Player bukkitPlayer = brain.getServerPlayer().getBukkitEntity();
                sigilsApi.unregisterBotSigils(bukkitPlayer);
            }
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
