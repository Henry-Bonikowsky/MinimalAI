package com.minimalai.ai;

import ai.djl.ndarray.NDManager;
import com.minimalai.MinimalAI;
import com.minimalai.playback.ActionExecutor;
import com.minimalai.recording.GameStateCollector;
import net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.MinecraftClient;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;

/**
 * Runs a trained neural network to control the player.
 * Executes at render rate for smooth camera movement.
 */
public class AIController {
    private static AIController instance;

    // Rate limiting (same as recording)
    private static final int TARGET_FPS = 90;
    private static final long FRAME_INTERVAL_NS = 1_000_000_000L / TARGET_FPS;

    private final GameStateCollector stateCollector;
    private final ActionExecutor actionExecutor;
    private final Path modelsPath;

    private NDManager manager;
    private SimpleNetwork network;
    private boolean isRunning;
    private long lastFrameTimeNs;

    // Stats
    private int frameCount;
    private int framesSinceLastLog;
    private long lastLogTime;

    public static AIController getInstance() {
        if (instance == null) {
            instance = new AIController();
        }
        return instance;
    }

    private AIController() {
        this.stateCollector = new GameStateCollector();
        this.actionExecutor = new ActionExecutor();
        this.isRunning = false;

        Path gameDir = FabricLoader.getInstance().getGameDir();
        this.modelsPath = gameDir.resolve("minimalai").resolve("models");

        // Register render event
        WorldRenderEvents.END.register(context -> {
            if (isRunning) {
                runFrame();
            }
        });

        MinimalAI.LOGGER.info("AIController initialized");
    }

    /**
     * Start AI mode with the latest trained model.
     */
    public boolean start() {
        if (isRunning) {
            MinimalAI.LOGGER.warn("AI already running");
            return false;
        }

        // Find latest model
        Path modelPath = findLatestModel();
        if (modelPath == null) {
            MinimalAI.LOGGER.warn("No trained model found. Train first with T key.");
            return false;
        }

        return start(modelPath);
    }

    /**
     * Start AI mode with a specific model.
     */
    public boolean start(Path modelPath) {
        if (isRunning) {
            stop();
        }

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null) {
            MinimalAI.LOGGER.warn("Cannot start AI: no player");
            return false;
        }

        try {
            manager = NDManager.newBaseManager();
            network = new SimpleNetwork(manager);
            network.load(modelPath);

            isRunning = true;
            lastFrameTimeNs = System.nanoTime();
            frameCount = 0;
            framesSinceLastLog = 0;
            lastLogTime = System.currentTimeMillis();

            MinimalAI.LOGGER.info("AI started with model: {}", modelPath.getFileName());
            return true;

        } catch (IOException e) {
            MinimalAI.LOGGER.error("Failed to load model", e);
            if (manager != null) {
                manager.close();
                manager = null;
            }
            return false;
        }
    }

    public void stop() {
        if (!isRunning) return;

        isRunning = false;
        actionExecutor.releaseAllKeys();

        MinimalAI.LOGGER.info("AI stopped after {} frames", frameCount);

        if (manager != null) {
            manager.close();
            manager = null;
        }
        network = null;
    }

    private void runFrame() {
        if (network == null) return;

        // Rate limiting
        long nowNs = System.nanoTime();
        if (nowNs - lastFrameTimeNs < FRAME_INTERVAL_NS) {
            return;
        }
        lastFrameTimeNs = nowNs;

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null || client.currentScreen != null) {
            return; // Don't run while in menu
        }

        // Collect current state
        float[] state = stateCollector.collect();

        // Forward pass
        float[][] outputs = network.forward(state);
        float[] actionLogits = outputs[0];
        float[] camera = outputs[1];

        // Convert logits to probabilities and sample actions
        float[] probs = network.getActionProbabilities(actionLogits);
        boolean[] actions = new boolean[probs.length];
        for (int i = 0; i < probs.length; i++) {
            actions[i] = probs[i] > 0.5f;
        }

        // Execute actions
        actionExecutor.applyDiscreteActions(actions);
        actionExecutor.applyCameraMovement(camera[0], camera[1]);

        frameCount++;

        // Log progress
        framesSinceLastLog++;
        long now = System.currentTimeMillis();
        if (now - lastLogTime >= 2000) { // Every 2 seconds
            float fps = framesSinceLastLog * 1000.0f / (now - lastLogTime);

            // Count active actions
            int activeCount = 0;
            for (boolean a : actions) if (a) activeCount++;

            MinimalAI.LOGGER.info("[AI] {} frames | {} FPS | cam({}, {}) | {} actions",
                frameCount,
                String.format("%.1f", fps),
                String.format("%.2f", camera[0]),
                String.format("%.2f", camera[1]),
                activeCount);

            framesSinceLastLog = 0;
            lastLogTime = now;
        }
    }

    private Path findLatestModel() {
        try {
            if (!Files.exists(modelsPath)) {
                return null;
            }

            return Files.list(modelsPath)
                .filter(p -> p.toString().endsWith(".bin"))
                .max(Comparator.comparingLong(p -> {
                    try {
                        return Files.getLastModifiedTime(p).toMillis();
                    } catch (IOException e) {
                        return 0;
                    }
                }))
                .orElse(null);

        } catch (IOException e) {
            MinimalAI.LOGGER.error("Error finding model", e);
            return null;
        }
    }

    public boolean isRunning() {
        return isRunning;
    }

    public int getFrameCount() {
        return frameCount;
    }

    public Path getModelsPath() {
        return modelsPath;
    }

    /**
     * List all saved models.
     */
    public java.util.List<Path> listModels() {
        try {
            if (!Files.exists(modelsPath)) {
                return java.util.Collections.emptyList();
            }
            return Files.list(modelsPath)
                .filter(p -> p.toString().endsWith(".bin"))
                .sorted(Comparator.comparingLong((Path p) -> {
                    try {
                        return Files.getLastModifiedTime(p).toMillis();
                    } catch (IOException e) {
                        return 0;
                    }
                }).reversed())
                .collect(java.util.stream.Collectors.toList());
        } catch (IOException e) {
            MinimalAI.LOGGER.error("Error listing models", e);
            return java.util.Collections.emptyList();
        }
    }

    /**
     * Delete a model file.
     */
    public boolean deleteModel(Path model) {
        try {
            Files.deleteIfExists(model);
            MinimalAI.LOGGER.info("Deleted model: {}", model.getFileName());
            return true;
        } catch (IOException e) {
            MinimalAI.LOGGER.error("Failed to delete model", e);
            return false;
        }
    }

    /**
     * Delete all models.
     */
    public void deleteAllModels() {
        for (Path model : listModels()) {
            deleteModel(model);
        }
    }
}
