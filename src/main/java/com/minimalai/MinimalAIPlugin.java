package com.minimalai;

import com.minimalai.ai.*;
import com.minimalai.bot.BotBrain;
import com.minimalai.bot.FakePlayerManager;
import com.minimalai.bot.KitManager;
import com.minimalai.commands.BotCommand;
import com.minimalai.commands.MaiCommand;
import com.minimalai.integration.ArcaneSigilsAPI;
import com.minimalai.integration.ArcaneSigilsBridge;
import com.minimalai.training.*;
import org.bukkit.Bukkit;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitRunnable;

import java.nio.file.Path;
import java.util.List;

/**
 * MinimalAI Paper plugin entry point.
 *
 * Spawns NMS fake player bots, runs TorchScript model inference via DJL,
 * and supports live RL training through a TCP connection to a Python server.
 */
public class MinimalAIPlugin extends JavaPlugin {

    private FakePlayerManager botManager;
    private ModelManager modelManager;
    private ObservationBuilder obsBuilder;
    private ActionExecutor actionExecutor;
    private ArcaneSigilsBridge sigilsBridge;
    private RewardComputer rewardComputer;
    private EpisodeManager episodeManager;
    private ExperienceBuffer experienceBuffer;
    private TrainingClient trainingClient;
    private BukkitRunnable tickTask;
    private BotCommand botCmd;
    private MaiCommand maiCmd;

    @Override
    public void onEnable() {
        saveDefaultConfig();

        // Initialize ArcaneSigils bridge
        sigilsBridge = new ArcaneSigilsBridge(getLogger());

        // Initialize bot system
        botManager = new FakePlayerManager(getLogger(), this);
        getServer().getPluginManager().registerEvents(botManager, this);

        // Initialize model manager
        Path modelsDir = getDataFolder().toPath().resolve(
                getConfig().getString("model.directory", "models"));
        modelsDir.toFile().mkdirs();
        modelManager = new ModelManager(modelsDir, getLogger());

        // Load all models async to avoid blocking server startup (DJL init is slow)
        List<String> availableModels = modelManager.listModels();
        if (!availableModels.isEmpty()) {
            Bukkit.getScheduler().runTaskAsynchronously(this, () -> {
                for (String name : availableModels) {
                    try {
                        modelManager.loadModel(name);
                        getLogger().info("Loaded model: " + name);
                    } catch (Exception e) {
                        getLogger().warning("Failed to load model '" + name + "': " + e.getMessage());
                    }
                }
                getLogger().info("Model loading complete: " + availableModels.size() + " models loaded");
            });
        } else {
            getLogger().info("No models found in " + modelsDir);
        }

        // Initialize action executor with sigils API bridge
        ArcaneSigilsAPI sigilsApi = sigilsBridge.isAvailable() ? sigilsBridge : null;

        // Initialize observation builder (with sigils API for direct state queries)
        double arenaX = getConfig().getDouble("arena.center-x", 0.0);
        double arenaZ = getConfig().getDouble("arena.center-z", 0.0);
        obsBuilder = new ObservationBuilder(arenaX, arenaZ, sigilsApi);
        actionExecutor = new ActionExecutor(
                sigilsApi != null ? (player, slot) -> sigilsApi.activateAbility(player, slot) : null,
                sigilsApi != null ? (player, slot) -> sigilsApi.isSigilReady(player, slot) : null
        );

        // Initialize training components (uses default reward scales matching Python sim)
        rewardComputer = new RewardComputer();
        rewardComputer.setObservationBuilder(obsBuilder);
        getServer().getPluginManager().registerEvents(rewardComputer, this);

        int episodeLength = getConfig().getInt("training.episode-length", 600);
        episodeManager = new EpisodeManager(episodeLength);

        int batchSize = getConfig().getInt("training.batch-size", 2048);
        experienceBuffer = new ExperienceBuffer(batchSize);

        // Training client (connects to Python server)
        String trainHost = getConfig().getString("training.server-host", "localhost");
        int trainPort = getConfig().getInt("training.server-port", 9876);
        trainingClient = new TrainingClient(trainHost, trainPort, getLogger(),
                () -> Bukkit.getScheduler().runTask(this, () -> modelManager.hotReload("latest")));

        // Initialize kit manager
        Path kitsDir = getDataFolder().toPath().resolve("kits");
        KitManager kitManager = new KitManager(kitsDir, getLogger());

        // BotCommand is an internal service (brain lifecycle + tick)
        botCmd = new BotCommand(botManager, modelManager, obsBuilder, actionExecutor,
                sigilsApi, experienceBuffer, rewardComputer, episodeManager, kitManager);

        // MaiCommand handles all /mai subcommands + event listening
        maiCmd = new MaiCommand(botCmd, modelManager, kitManager,
                rewardComputer, episodeManager, experienceBuffer, trainingClient);
        botCmd.setMaiCommand(maiCmd);

        // Register the unified command
        getCommand("mai").setExecutor(maiCmd);
        getCommand("mai").setTabCompleter(maiCmd);
        getServer().getPluginManager().registerEvents(maiCmd, this);

        // Episode end handler
        episodeManager.setOnEpisodeEnd(event -> {
            getLogger().info(String.format("Episode ended for %s: reason=%s ticks=%d reward=%.2f kills=%d deaths=%d",
                    event.botName(), event.reason(), event.ticks(), event.totalReward(), event.kills(), event.deaths()));

            maiCmd.recordEpisodeEnd(event.totalReward());

            // Self-play episode timeout → reset round (draw)
            if (event.reason() == EpisodeManager.EpisodeEndReason.TIMEOUT && maiCmd.isSelfPlayActive()) {
                maiCmd.onSelfPlayEpisodeTimeout(event.botName());
                return;
            }

            // Send experience batch if buffer is full
            if (experienceBuffer.isFull() && trainingClient.isConnected()) {
                try {
                    byte[] data = experienceBuffer.toSerializable();
                    trainingClient.sendExperiences(data);
                } catch (Exception e) {
                    getLogger().warning("Failed to send experience batch: " + e.getMessage());
                }
            }

            // Reset bot brain and restart episode
            BotBrain brain = botCmd.getBrains().get(event.botName());
            if (brain != null) {
                brain.resetEpisode();
            }
            if (maiCmd.isTrainingEnabled()) {
                episodeManager.startEpisode(event.botName());
            }
        });

        // Start bot tick loop
        int tickRate = getConfig().getInt("bot.tick-rate", 1);
        tickTask = new BukkitRunnable() {
            @Override
            public void run() {
                botCmd.tickAll();
            }
        };
        tickTask.runTaskTimer(this, 20L, tickRate);

        getLogger().info("MinimalAI enabled - " + modelManager.listModels().size() + " models available");
    }

    @Override
    public void onDisable() {
        if (tickTask != null) tickTask.cancel();
        if (botCmd != null) botCmd.closeAll();
        if (trainingClient != null) trainingClient.disconnect();
        if (botManager != null) botManager.despawnAll();
        if (modelManager != null) modelManager.close();
        getLogger().info("MinimalAI disabled");
    }

    public FakePlayerManager getBotManager() { return botManager; }
    public ModelManager getModelManager() { return modelManager; }
    public ObservationBuilder getObsBuilder() { return obsBuilder; }
    public ActionExecutor getActionExecutor() { return actionExecutor; }
    public RewardComputer getRewardComputer() { return rewardComputer; }
    public EpisodeManager getEpisodeManager() { return episodeManager; }
    public ExperienceBuffer getExperienceBuffer() { return experienceBuffer; }
    public TrainingClient getTrainingClient() { return trainingClient; }
}
