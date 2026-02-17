package com.minimalai.training;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import java.util.logging.Logger;

/**
 * Manages per-bot episode lifecycle: start, tick counting, timeout, kill/death tracking, and reset.
 *
 * <p>Each registered bot has independent episode state. The manager fires an
 * {@link EpisodeEndEvent} via a configurable callback whenever an episode ends
 * for any reason (timeout, death, or manual stop).
 */
public class EpisodeManager {

    private static final Logger LOG = Logger.getLogger("MinimalAI");

    // ----------------------------------------------------------------
    //  Episode end metadata
    // ----------------------------------------------------------------

    public enum EpisodeEndReason { TIMEOUT, DEATH, MANUAL }

    public record EpisodeEndEvent(
        String botName,
        EpisodeEndReason reason,
        int ticks,
        float totalReward,
        int kills,
        int deaths
    ) {}

    // ----------------------------------------------------------------
    //  Per-bot episode state
    // ----------------------------------------------------------------

    public static class EpisodeState {
        int episodeNumber;
        int tickCount;
        int maxTicks;
        float totalReward;
        int kills;
        int deaths;
        long startTimeMs;

        EpisodeState(int episodeNumber, int maxTicks) {
            this.episodeNumber = episodeNumber;
            this.tickCount = 0;
            this.maxTicks = maxTicks;
            this.totalReward = 0.0f;
            this.kills = 0;
            this.deaths = 0;
            this.startTimeMs = System.currentTimeMillis();
        }
    }

    // ----------------------------------------------------------------
    //  Fields
    // ----------------------------------------------------------------

    private final Map<String, EpisodeState> episodes = new ConcurrentHashMap<>();
    private final int maxEpisodeTicks;
    private Consumer<EpisodeEndEvent> onEpisodeEnd;

    /**
     * @param maxEpisodeTicks maximum ticks before timeout (default 600 = 30 seconds at 20 TPS)
     */
    public EpisodeManager(int maxEpisodeTicks) {
        this.maxEpisodeTicks = maxEpisodeTicks;
    }

    public EpisodeManager() {
        this(600);
    }

    // ----------------------------------------------------------------
    //  Callback
    // ----------------------------------------------------------------

    public void setOnEpisodeEnd(Consumer<EpisodeEndEvent> callback) {
        this.onEpisodeEnd = callback;
    }

    // ----------------------------------------------------------------
    //  Lifecycle
    // ----------------------------------------------------------------

    /**
     * Start (or restart) an episode for the given bot.
     */
    public void startEpisode(String botName) {
        int nextNumber = 1;
        EpisodeState prev = episodes.get(botName);
        if (prev != null) {
            nextNumber = prev.episodeNumber + 1;
        }

        EpisodeState state = new EpisodeState(nextNumber, maxEpisodeTicks);
        episodes.put(botName, state);
        LOG.info("[EpisodeManager] Episode " + nextNumber + " started for " + botName);
    }

    /**
     * Tick the episode for a bot. Returns {@code true} if the episode should end (timeout).
     */
    public boolean tick(String botName) {
        EpisodeState state = episodes.get(botName);
        if (state == null) return false;

        state.tickCount++;

        if (state.tickCount >= state.maxTicks) {
            endEpisode(botName, EpisodeEndReason.TIMEOUT);
            return true;
        }
        return false;
    }

    /**
     * Record a kill for the bot's current episode.
     */
    public void onKill(String botName) {
        EpisodeState state = episodes.get(botName);
        if (state != null) state.kills++;
    }

    /**
     * Record a death and end the episode.
     */
    public void onDeath(String botName) {
        EpisodeState state = episodes.get(botName);
        if (state != null) {
            state.deaths++;
            endEpisode(botName, EpisodeEndReason.DEATH);
        }
    }

    /**
     * Accumulate reward for the current episode.
     */
    public void addReward(String botName, float reward) {
        EpisodeState state = episodes.get(botName);
        if (state != null) state.totalReward += reward;
    }

    /**
     * Get episode progress as a ratio (0.0 to 1.0).
     */
    public float getProgress(String botName) {
        EpisodeState state = episodes.get(botName);
        if (state == null) return 0.0f;
        return (float) state.tickCount / state.maxTicks;
    }

    /**
     * Get the raw episode state (may be null if no episode active).
     */
    public EpisodeState getState(String botName) {
        return episodes.get(botName);
    }

    /**
     * End the episode for a bot with the given reason. Fires the callback and logs stats.
     */
    public void endEpisode(String botName, EpisodeEndReason reason) {
        EpisodeState state = episodes.remove(botName);
        if (state == null) return;

        long durationMs = System.currentTimeMillis() - state.startTimeMs;
        LOG.info(String.format("[EpisodeManager] Episode %d ended for %s: reason=%s ticks=%d reward=%.2f kills=%d deaths=%d duration=%dms",
            state.episodeNumber, botName, reason, state.tickCount,
            state.totalReward, state.kills, state.deaths, durationMs));

        if (onEpisodeEnd != null) {
            EpisodeEndEvent event = new EpisodeEndEvent(
                botName, reason, state.tickCount,
                state.totalReward, state.kills, state.deaths
            );
            onEpisodeEnd.accept(event);
        }
    }

    /**
     * Check whether the given bot has an active episode.
     */
    public boolean isActive(String botName) {
        return episodes.containsKey(botName);
    }
}
