package com.minimalai.playback;

import com.minimalai.MinimalAI;
import com.minimalai.recording.Recording;
import com.minimalai.recording.RecordingStorage;
import net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents;
import net.minecraft.client.MinecraftClient;

import java.io.IOException;
import java.util.List;

/**
 * Controls playback of recordings at render rate.
 * Supports play, pause, stop, and speed control.
 */
public class PlaybackController {
    private static PlaybackController instance;

    // Playback rate (matches recording rate)
    private static final int TARGET_FPS = 90;
    private static final long FRAME_INTERVAL_NS = 1_000_000_000L / TARGET_FPS;

    private final ActionExecutor actionExecutor;
    private final RecordingStorage storage;

    private Recording currentRecording;
    private List<Recording.Frame> frames;
    private int currentFrameIndex;
    private boolean isPlaying;
    private boolean isPaused;
    private float playbackSpeed;

    // Timing
    private long playbackStartTimeNs;
    private long lastFrameTimeNs;
    private long pausedAtNs;

    // Stats
    private int framesSinceLastLog;
    private long lastLogTime;

    public static PlaybackController getInstance() {
        if (instance == null) {
            instance = new PlaybackController();
        }
        return instance;
    }

    private PlaybackController() {
        this.actionExecutor = new ActionExecutor();
        this.storage = new RecordingStorage();
        this.isPlaying = false;
        this.isPaused = false;
        this.playbackSpeed = 1.0f;

        // Register render event for frame playback
        WorldRenderEvents.END.register(context -> {
            if (isPlaying && !isPaused) {
                playFrame();
            }
        });

        MinimalAI.LOGGER.info("PlaybackController initialized");
    }

    /**
     * Load and start playing a recording by name.
     */
    public boolean play(String recordingName) {
        try {
            Recording recording = storage.load(recordingName);
            return play(recording);
        } catch (IOException e) {
            MinimalAI.LOGGER.error("Failed to load recording: {}", recordingName, e);
            return false;
        }
    }

    /**
     * Start playing a recording.
     */
    public boolean play(Recording recording) {
        if (isPlaying) {
            stop();
        }

        if (recording == null || recording.getFrameCount() == 0) {
            MinimalAI.LOGGER.warn("Cannot play empty recording");
            return false;
        }

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null) {
            MinimalAI.LOGGER.warn("Cannot play: no player");
            return false;
        }

        this.currentRecording = recording;
        this.frames = recording.getFrames();
        this.currentFrameIndex = 0;
        this.playbackStartTimeNs = System.nanoTime();
        this.lastFrameTimeNs = playbackStartTimeNs;
        this.isPlaying = true;
        this.isPaused = false;

        framesSinceLastLog = 0;
        lastLogTime = System.currentTimeMillis();

        MinimalAI.LOGGER.info("Playback started: {} ({} frames, {} FPS)",
            recording.getName(),
            recording.getFrameCount(),
            String.format("%.1f", recording.getAvgFps()));

        return true;
    }

    /**
     * Play the most recent recording.
     */
    public boolean playLatest() {
        var recordings = storage.listRecordings();
        if (recordings.isEmpty()) {
            MinimalAI.LOGGER.warn("No recordings found");
            return false;
        }

        String latestName = recordings.get(0).name;
        return play(latestName);
    }

    public void pause() {
        if (!isPlaying || isPaused) return;

        isPaused = true;
        pausedAtNs = System.nanoTime();
        actionExecutor.releaseAllKeys();

        MinimalAI.LOGGER.info("Playback paused at frame {}/{}",
            currentFrameIndex, frames.size());
    }

    public void resume() {
        if (!isPlaying || !isPaused) return;

        // Adjust start time to account for pause duration
        long pauseDuration = System.nanoTime() - pausedAtNs;
        playbackStartTimeNs += pauseDuration;
        lastFrameTimeNs += pauseDuration;

        isPaused = false;
        MinimalAI.LOGGER.info("Playback resumed at frame {}/{}",
            currentFrameIndex, frames.size());
    }

    public void togglePause() {
        if (isPaused) {
            resume();
        } else {
            pause();
        }
    }

    public void stop() {
        if (!isPlaying) return;

        isPlaying = false;
        isPaused = false;
        actionExecutor.releaseAllKeys();

        MinimalAI.LOGGER.info("Playback stopped at frame {}/{}",
            currentFrameIndex, frames != null ? frames.size() : 0);

        currentRecording = null;
        frames = null;
        currentFrameIndex = 0;
    }

    public void setSpeed(float speed) {
        this.playbackSpeed = Math.max(0.1f, Math.min(4.0f, speed));
        MinimalAI.LOGGER.info("Playback speed: {}x", String.format("%.1f", playbackSpeed));
    }

    private void playFrame() {
        if (frames == null || currentFrameIndex >= frames.size()) {
            // Playback complete
            MinimalAI.LOGGER.info("Playback complete: {} frames played", currentFrameIndex);
            stop();
            return;
        }

        // Rate limiting with speed adjustment
        long nowNs = System.nanoTime();
        long adjustedInterval = (long) (FRAME_INTERVAL_NS / playbackSpeed);
        if (nowNs - lastFrameTimeNs < adjustedInterval) {
            return;
        }
        lastFrameTimeNs = nowNs;

        // Get and execute current frame
        Recording.Frame frame = frames.get(currentFrameIndex);

        // Apply actions
        actionExecutor.applyDiscreteActions(frame.discreteActions);
        actionExecutor.applyCameraMovement(
            frame.continuousActions[0] * playbackSpeed,
            frame.continuousActions[1] * playbackSpeed
        );

        currentFrameIndex++;

        // Log progress every second
        framesSinceLastLog++;
        long now = System.currentTimeMillis();
        if (now - lastLogTime >= 1000) {
            float fps = framesSinceLastLog * 1000.0f / (now - lastLogTime);
            float progress = (float) currentFrameIndex / frames.size() * 100;
            MinimalAI.LOGGER.info("[PLAY] {}/{} ({} %) | {} FPS | {}x speed",
                currentFrameIndex,
                frames.size(),
                String.format("%.0f", progress),
                String.format("%.1f", fps),
                String.format("%.1f", playbackSpeed));

            framesSinceLastLog = 0;
            lastLogTime = now;
        }
    }

    public boolean isPlaying() {
        return isPlaying;
    }

    public boolean isPaused() {
        return isPaused;
    }

    public Recording getCurrentRecording() {
        return currentRecording;
    }

    public int getCurrentFrameIndex() {
        return currentFrameIndex;
    }

    public int getTotalFrames() {
        return frames != null ? frames.size() : 0;
    }

    public float getPlaybackSpeed() {
        return playbackSpeed;
    }

    public RecordingStorage getStorage() {
        return storage;
    }
}
