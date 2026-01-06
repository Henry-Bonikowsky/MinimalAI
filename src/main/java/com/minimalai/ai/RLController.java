package com.minimalai.ai;

import com.minimalai.MinimalAI;
import com.minimalai.playback.ActionExecutor;
import com.minimalai.recording.GameStateCollector;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents;
import net.minecraft.client.MinecraftClient;

/**
 * Controls RL training loop.
 *
 * Training cycle:
 * 1. Run BC pass (interleaved training)
 * 2. Start episode
 * 3. Run AI with network, collect experiences
 * 4. On episode end: REINFORCE update
 * 5. Reset and repeat
 */
public class RLController {
    private static RLController instance;

    // Timing (same as AIController)
    private static final int TARGET_FPS = 90;
    private static final long FRAME_INTERVAL_NS = 1_000_000_000L / TARGET_FPS;

    private final GameStateCollector stateCollector;
    private final ActionExecutor actionExecutor;
    private final RLTrainer trainer;
    private final EpisodeManager episodeManager;
    private final RewardDetector rewardDetector;

    // State
    private boolean isRunning = false;
    private long lastFrameTimeNs;
    private ExperienceBuffer currentBuffer;

    // Stats
    private int totalFrames = 0;
    private int framesThisEpisode = 0;

    public static RLController getInstance() {
        if (instance == null) {
            instance = new RLController();
        }
        return instance;
    }

    private RLController() {
        this.stateCollector = new GameStateCollector();
        this.actionExecutor = new ActionExecutor();
        this.trainer = RLTrainer.getInstance();
        this.episodeManager = EpisodeManager.getInstance();
        this.rewardDetector = RewardDetector.getInstance();

        // Register render event for high-frequency updates
        WorldRenderEvents.END.register(context -> {
            if (isRunning) {
                runFrame();
            }
        });

        // Register tick event for episode management
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            if (isRunning && episodeManager.isEpisodeActive()) {
                if (!episodeManager.tick()) {
                    // Episode ended (timeout or death)
                    onEpisodeEnd(episodeManager.getLastEndReason() == EpisodeManager.EpisodeEndReason.GOAL_ACHIEVED ? 1.0f : 0.0f);
                }
            }
        });

        // Listen for reward events
        rewardDetector.addListener(event -> {
            if (isRunning && episodeManager.isEpisodeActive()) {
                if (event.type == RewardDetector.RewardType.BLOCK_BROKEN) {
                    // Goal achieved!
                    episodeManager.endEpisodeWithReward(1.0f);
                    onEpisodeEnd(1.0f);
                }
            }
        });

        MinimalAI.LOGGER.info("RLController initialized");
    }

    /**
     * Start RL training.
     * Captures current position as start, initializes trainer.
     */
    public boolean start() {
        if (isRunning) {
            MinimalAI.LOGGER.warn("RL training already running");
            return false;
        }

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null) {
            MinimalAI.LOGGER.warn("Cannot start RL: no player");
            return false;
        }

        // Capture start position
        episodeManager.captureStartPosition();

        // Initialize trainer
        if (!trainer.initialize()) {
            MinimalAI.LOGGER.error("Failed to initialize RLTrainer");
            return false;
        }

        isRunning = true;
        lastFrameTimeNs = System.nanoTime();
        totalFrames = 0;

        // Enable reward detection
        rewardDetector.enable();

        // Run initial BC pass
        trainer.runBCPass();

        // Start first episode
        startNewEpisode();

        MinimalAI.LOGGER.info("=== RL Training Started ===");
        MinimalAI.LOGGER.info("Press End to stop. Goal: mine any block.");
        return true;
    }

    /**
     * Stop RL training.
     */
    public void stop() {
        if (!isRunning) return;

        isRunning = false;
        actionExecutor.releaseAllKeys();
        rewardDetector.disable();

        // End current episode without reward
        if (episodeManager.isEpisodeActive()) {
            episodeManager.endEpisode(EpisodeManager.EpisodeEndReason.STOPPED, 0);
        }

        trainer.stop();

        MinimalAI.LOGGER.info("=== RL Training Stopped ===");
        MinimalAI.LOGGER.info("Total frames: {}, Episodes: {}", totalFrames, episodeManager.getEpisodeCount());
    }

    /**
     * Start a new episode.
     */
    private void startNewEpisode() {
        // Reset to start position
        episodeManager.resetToStart();

        // Clear network frame buffer
        if (trainer.getNetwork() != null) {
            trainer.getNetwork().clearFrameBuffer();
        }

        // Create new experience buffer
        currentBuffer = new ExperienceBuffer();
        framesThisEpisode = 0;

        // Reset reward detector episode stats
        rewardDetector.resetEpisode();

        // Start episode
        episodeManager.startEpisode();
    }

    /**
     * Called when episode ends.
     */
    private void onEpisodeEnd(float reward) {
        // Finalize experience buffer
        currentBuffer.setEpisodeReward(reward);

        // Process episode with trainer
        trainer.processEpisode(currentBuffer);

        // Run BC pass
        trainer.runBCPass();

        // Start new episode after short delay
        MinecraftClient.getInstance().execute(() -> {
            if (isRunning) {
                startNewEpisode();
            }
        });
    }

    /**
     * Run a single frame of RL.
     */
    private void runFrame() {
        SimpleNetworkRL network = trainer.getNetwork();
        if (network == null) return;

        // Rate limiting
        long nowNs = System.nanoTime();
        if (nowNs - lastFrameTimeNs < FRAME_INTERVAL_NS) {
            return;
        }
        lastFrameTimeNs = nowNs;

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null || client.currentScreen != null) {
            return; // Don't run in menu
        }

        if (!episodeManager.isEpisodeActive()) {
            return; // Episode not active (transitioning)
        }

        // Collect state
        float[] state = stateCollector.collect();

        // Forward pass with frame stacking
        SimpleNetworkRL.ForwardResult fwd = network.forwardWithFrame(state);

        // Sample actions
        SimpleNetworkRL.SampleResult sample = network.sampleActions(fwd);

        // Execute actions
        actionExecutor.applyDiscreteActions(sample.actions);
        actionExecutor.applyCameraMovement(sample.camera[0], sample.camera[1]);

        // Store experience (only if frame buffer is ready)
        if (network.isFrameBufferReady()) {
            currentBuffer.addStep(
                network.getStackedInput(),
                sample.actions,
                sample.logProbs,
                sample.camera,
                sample.value
            );
        }

        totalFrames++;
        framesThisEpisode++;

        // Log progress periodically
        if (totalFrames % 180 == 0) { // Every ~2 seconds
            MinimalAI.LOGGER.info("[RL] Episode {}, Frame {}/{}, Blocks mined: {}",
                episodeManager.getEpisodeCount(),
                framesThisEpisode,
                totalFrames,
                rewardDetector.getBlocksMinedThisEpisode());
        }
    }

    // === Getters ===

    public boolean isRunning() {
        return isRunning;
    }

    public int getTotalFrames() {
        return totalFrames;
    }

    public int getFramesThisEpisode() {
        return framesThisEpisode;
    }

    public RLTrainer getTrainer() {
        return trainer;
    }

    public EpisodeManager getEpisodeManager() {
        return episodeManager;
    }

    public RewardDetector getRewardDetector() {
        return rewardDetector;
    }
}
