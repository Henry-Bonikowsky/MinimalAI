package com.minimalai.ai;

import com.minimalai.MinimalAI;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Vec3d;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Manages episode lifecycle for RL training.
 *
 * An episode:
 * - Starts when training begins or after reset
 * - Ends when: goal achieved (reward), timeout, or death
 * - Auto-resets player to starting position
 */
public class EpisodeManager {
    private static EpisodeManager instance;

    // Episode settings
    private int timeoutTicks = 30 * 20;  // 30 seconds at 20 ticks/sec
    private Vec3d startPosition;
    private float startYaw;
    private float startPitch;

    // Episode state
    private boolean episodeActive = false;
    private int currentEpisodeTicks = 0;
    private int episodeCount = 0;
    private EpisodeEndReason lastEndReason = null;

    // Listeners
    private final List<Consumer<EpisodeEndEvent>> endListeners = new ArrayList<>();

    // Player death tracking
    private float lastHealth = 20f;

    public static EpisodeManager getInstance() {
        if (instance == null) {
            instance = new EpisodeManager();
        }
        return instance;
    }

    private EpisodeManager() {
        MinimalAI.LOGGER.info("EpisodeManager initialized");
    }

    /**
     * Set the starting position for auto-reset.
     * Call this before starting RL training.
     */
    public void setStartPosition(Vec3d position, float yaw, float pitch) {
        this.startPosition = position;
        this.startYaw = yaw;
        this.startPitch = pitch;
        MinimalAI.LOGGER.info("Start position set: {} (yaw={}, pitch={})", position, yaw, pitch);
    }

    /**
     * Capture current player position as start position.
     */
    public void captureStartPosition() {
        ClientPlayerEntity player = MinecraftClient.getInstance().player;
        if (player != null) {
            setStartPosition(player.getPos(), player.getYaw(), player.getPitch());
        }
    }

    /**
     * Start a new episode.
     */
    public void startEpisode() {
        episodeActive = true;
        currentEpisodeTicks = 0;
        episodeCount++;
        lastEndReason = null;

        ClientPlayerEntity player = MinecraftClient.getInstance().player;
        if (player != null) {
            lastHealth = player.getHealth();
        }

        MinimalAI.LOGGER.info("[Episode {}] Started", episodeCount);
    }

    /**
     * Called every game tick during an active episode.
     * Returns true if episode should continue, false if it ended.
     */
    public boolean tick() {
        if (!episodeActive) return false;

        currentEpisodeTicks++;

        // Check timeout
        if (currentEpisodeTicks >= timeoutTicks) {
            endEpisode(EpisodeEndReason.TIMEOUT, 0);
            return false;
        }

        // Check death
        ClientPlayerEntity player = MinecraftClient.getInstance().player;
        if (player != null) {
            if (player.isDead() || player.getHealth() <= 0) {
                endEpisode(EpisodeEndReason.DEATH, 0);
                return false;
            }

            // Check for significant damage (optional - can detect combat/fall damage)
            if (player.getHealth() < lastHealth - 5) {
                // Took significant damage but didn't die
                MinimalAI.LOGGER.debug("[Episode {}] Player took damage: {} -> {}",
                    episodeCount, lastHealth, player.getHealth());
            }
            lastHealth = player.getHealth();
        }

        return true;
    }

    /**
     * End the current episode with a reward achieved.
     */
    public void endEpisodeWithReward(float reward) {
        endEpisode(EpisodeEndReason.GOAL_ACHIEVED, reward);
    }

    /**
     * End the current episode.
     */
    public void endEpisode(EpisodeEndReason reason, float reward) {
        if (!episodeActive) return;

        episodeActive = false;
        lastEndReason = reason;

        float duration = currentEpisodeTicks / 20.0f;
        MinimalAI.LOGGER.info("[Episode {}] Ended: {} (reward={}, duration={}s, ticks={})",
            episodeCount, reason, reward, String.format("%.1f", duration), currentEpisodeTicks);

        // Notify listeners
        EpisodeEndEvent event = new EpisodeEndEvent(reason, reward, currentEpisodeTicks, episodeCount);
        for (Consumer<EpisodeEndEvent> listener : endListeners) {
            listener.accept(event);
        }
    }

    /**
     * Reset player to starting position.
     * Uses teleport command.
     */
    public void resetToStart() {
        if (startPosition == null) {
            MinimalAI.LOGGER.warn("No start position set, cannot reset");
            return;
        }

        ClientPlayerEntity player = MinecraftClient.getInstance().player;
        if (player == null) return;

        // Teleport using command (works in singleplayer with cheats, or with OP)
        String command = String.format("tp @s %.2f %.2f %.2f %.2f %.2f",
            startPosition.x, startPosition.y, startPosition.z, startYaw, startPitch);

        try {
            MinecraftClient.getInstance().getNetworkHandler().sendChatCommand(command);
            MinimalAI.LOGGER.info("Reset to start position");
        } catch (Exception e) {
            MinimalAI.LOGGER.warn("Failed to teleport (cheats may be disabled): {}", e.getMessage());
            // Alternative: directly set position (may have desync issues)
            player.setPosition(startPosition);
            player.setYaw(startYaw);
            player.setPitch(startPitch);
        }
    }

    /**
     * Add a listener for episode end events.
     */
    public void addEndListener(Consumer<EpisodeEndEvent> listener) {
        endListeners.add(listener);
    }

    /**
     * Remove a listener.
     */
    public void removeEndListener(Consumer<EpisodeEndEvent> listener) {
        endListeners.remove(listener);
    }

    /**
     * Clear all listeners.
     */
    public void clearListeners() {
        endListeners.clear();
    }

    // === Getters/Setters ===

    public boolean isEpisodeActive() {
        return episodeActive;
    }

    public int getCurrentEpisodeTicks() {
        return currentEpisodeTicks;
    }

    public int getEpisodeCount() {
        return episodeCount;
    }

    public EpisodeEndReason getLastEndReason() {
        return lastEndReason;
    }

    public void setTimeoutTicks(int ticks) {
        this.timeoutTicks = ticks;
    }

    public void setTimeoutSeconds(int seconds) {
        this.timeoutTicks = seconds * 20;
    }

    public int getTimeoutTicks() {
        return timeoutTicks;
    }

    public Vec3d getStartPosition() {
        return startPosition;
    }

    // === Event types ===

    public enum EpisodeEndReason {
        GOAL_ACHIEVED,  // Successfully completed the task (got reward)
        TIMEOUT,        // Ran out of time
        DEATH,          // Player died
        STOPPED         // Manually stopped by user
    }

    public static class EpisodeEndEvent {
        public final EpisodeEndReason reason;
        public final float reward;
        public final int durationTicks;
        public final int episodeNumber;

        public EpisodeEndEvent(EpisodeEndReason reason, float reward, int durationTicks, int episodeNumber) {
            this.reason = reason;
            this.reward = reward;
            this.durationTicks = durationTicks;
            this.episodeNumber = episodeNumber;
        }
    }
}
