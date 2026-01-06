package com.minimalai.recording;

import com.minimalai.MinimalAI;
import net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents;
import net.minecraft.client.MinecraftClient;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Manages the recording lifecycle.
 * Records at capped rate (90 Hz) for smooth camera capture without excessive data.
 */
public class RecordingManager {
    private static RecordingManager instance;

    // Recording rate cap (90 Hz = ~11.1ms per frame)
    private static final int TARGET_FPS = 90;
    private static final long FRAME_INTERVAL_NS = 1_000_000_000L / TARGET_FPS;

    private final GameStateCollector stateCollector;
    private final ActionRecorder actionRecorder;
    private final RecordingStorage storage;

    private Recording currentRecording;
    private long recordingStartTime;
    private boolean isRecording;

    // Rate limiting
    private long lastFrameTimeNs;

    // Stats
    private int framesSinceLastLog;
    private long lastLogTime;

    public static RecordingManager getInstance() {
        if (instance == null) {
            instance = new RecordingManager();
        }
        return instance;
    }

    private RecordingManager() {
        this.stateCollector = new GameStateCollector();
        this.actionRecorder = new ActionRecorder();
        this.storage = new RecordingStorage();
        this.isRecording = false;

        // Register render event for frame capture
        WorldRenderEvents.END.register(context -> {
            if (isRecording) {
                captureFrame();
            }
        });

        MinimalAI.LOGGER.info("RecordingManager initialized");
    }

    public void startRecording() {
        if (isRecording) {
            MinimalAI.LOGGER.warn("Already recording!");
            return;
        }

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null) {
            MinimalAI.LOGGER.warn("Cannot start recording: no player");
            return;
        }

        // Generate recording name
        String timestamp = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(new Date());
        String name = "recording_" + timestamp;

        currentRecording = new Recording(name);
        recordingStartTime = System.currentTimeMillis();
        lastFrameTimeNs = System.nanoTime();
        actionRecorder.reset();
        isRecording = true;

        framesSinceLastLog = 0;
        lastLogTime = recordingStartTime;

        MinimalAI.LOGGER.info("Recording started: {}", name);
    }

    public void stopRecording() {
        if (!isRecording) {
            MinimalAI.LOGGER.warn("Not recording!");
            return;
        }

        isRecording = false;
        long endTime = System.currentTimeMillis();
        currentRecording.finalize(recordingStartTime, endTime);

        MinimalAI.LOGGER.info("Recording stopped: {} frames, {} FPS, {}",
            currentRecording.getFrameCount(),
            String.format("%.1f", currentRecording.getAvgFps()),
            formatDuration(currentRecording.getDurationMs()));

        // Save recording
        try {
            storage.save(currentRecording);
        } catch (IOException e) {
            MinimalAI.LOGGER.error("Failed to save recording!", e);
        }

        currentRecording = null;
    }

    public void toggleRecording() {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    }

    private void captureFrame() {
        if (currentRecording == null) return;

        // Rate limiting - skip frame if too soon
        long nowNs = System.nanoTime();
        if (nowNs - lastFrameTimeNs < FRAME_INTERVAL_NS) {
            return;
        }
        lastFrameTimeNs = nowNs;

        long timestamp = System.currentTimeMillis() - recordingStartTime;

        float[] gameState = stateCollector.collect();
        boolean[] discreteActions = actionRecorder.captureDiscreteActions();
        float[] continuousActions = actionRecorder.captureContinuousActions();

        Recording.Frame frame = new Recording.Frame(timestamp, gameState, discreteActions, continuousActions);
        currentRecording.addFrame(frame);

        // Log progress every second
        framesSinceLastLog++;
        long now = System.currentTimeMillis();
        if (now - lastLogTime >= 1000) {
            float fps = framesSinceLastLog * 1000.0f / (now - lastLogTime);
            MinimalAI.LOGGER.info("[REC] {} frames | {} FPS | cam({}, {})",
                currentRecording.getFrameCount(),
                String.format("%.1f", fps),
                String.format("%.2f", continuousActions[0]),
                String.format("%.2f", continuousActions[1]));

            // Log a sample of the game state periodically
            if (currentRecording.getFrameCount() % 100 == 0) {
                logGameStateSample(gameState, discreteActions);
            }

            framesSinceLastLog = 0;
            lastLogTime = now;
        }
    }

    private void logGameStateSample(float[] state, boolean[] actions) {
        StringBuilder activeActions = new StringBuilder();
        String[] actionNames = ActionRecorder.getDiscreteActionNames();
        for (int i = 0; i < actions.length; i++) {
            if (actions[i]) {
                if (activeActions.length() > 0) activeActions.append(", ");
                activeActions.append(actionNames[i]);
            }
        }

        MinimalAI.LOGGER.info("[STATE] health={} yaw={} pitch={} onGround={} | actions: [{}]",
            String.format("%.2f", state[0]),  // health
            String.format("%.2f", state[8]),  // yaw
            String.format("%.2f", state[9]),  // pitch
            state[7] > 0.5 ? "Y" : "N",  // onGround
            activeActions.length() > 0 ? activeActions.toString() : "none");
    }

    public boolean isRecording() {
        return isRecording;
    }

    public Recording getCurrentRecording() {
        return currentRecording;
    }

    public RecordingStorage getStorage() {
        return storage;
    }

    public GameStateCollector getStateCollector() {
        return stateCollector;
    }

    private String formatDuration(long ms) {
        long seconds = ms / 1000;
        long minutes = seconds / 60;
        seconds = seconds % 60;
        return String.format("%d:%02d", minutes, seconds);
    }
}
