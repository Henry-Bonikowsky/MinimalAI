package com.minimalai.recording;

import java.util.ArrayList;
import java.util.List;

/**
 * A recording of gameplay frames.
 * Each frame contains game state and player actions captured at render rate.
 */
public class Recording {
    public static final int VERSION = 1;

    private String name;
    private long createdAt;
    private int version;
    private final List<Frame> frames;

    // Metadata
    private long durationMs;
    private float avgFps;

    public Recording(String name) {
        this.name = name;
        this.createdAt = System.currentTimeMillis();
        this.version = VERSION;
        this.frames = new ArrayList<>();
    }

    public void addFrame(Frame frame) {
        frames.add(frame);
    }

    public List<Frame> getFrames() {
        return frames;
    }

    public int getFrameCount() {
        return frames.size();
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public long getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(long createdAt) {
        this.createdAt = createdAt;
    }

    public int getVersion() {
        return version;
    }

    public void setVersion(int version) {
        this.version = version;
    }

    public long getDurationMs() {
        return durationMs;
    }

    public void setDurationMs(long durationMs) {
        this.durationMs = durationMs;
    }

    public float getAvgFps() {
        return avgFps;
    }

    public void setAvgFps(float avgFps) {
        this.avgFps = avgFps;
    }

    public void finalize(long startTime, long endTime) {
        this.durationMs = endTime - startTime;
        if (durationMs > 0) {
            this.avgFps = frames.size() * 1000.0f / durationMs;
        }
    }

    /**
     * A single frame of recorded data.
     */
    public static class Frame {
        public final long timestamp;      // Milliseconds since recording start
        public final float[] gameState;   // 64 floats
        public final boolean[] discreteActions; // 20 booleans
        public final float[] continuousActions; // 2 floats (yaw/pitch delta)

        public Frame(long timestamp, float[] gameState, boolean[] discreteActions, float[] continuousActions) {
            this.timestamp = timestamp;
            this.gameState = gameState;
            this.discreteActions = discreteActions;
            this.continuousActions = continuousActions;
        }
    }
}
