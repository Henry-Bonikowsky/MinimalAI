package com.minimalai.ai;

import ai.djl.ndarray.NDManager;
import com.minimalai.MinimalAI;
import com.minimalai.recording.Recording;
import com.minimalai.recording.RecordingStorage;
import net.fabricmc.loader.api.FabricLoader;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Behavior cloning trainer.
 * Trains the network to imitate recorded gameplay.
 */
public class Trainer {
    private static Trainer instance;

    // Training parameters
    private float learningRate = 0.001f;
    private int epochs = 50;
    private int batchSize = 32;
    private float cameraWeight = 1.0f;  // Equal weight to actions now that camera is normalized

    private final RecordingStorage storage;
    private final Path modelsPath;

    private SimpleNetwork network;
    private NDManager manager;

    private AtomicBoolean isTraining = new AtomicBoolean(false);
    private AtomicBoolean shouldStop = new AtomicBoolean(false);

    // Training stats
    private int currentEpoch;
    private float lastActionLoss;
    private float lastCameraLoss;
    private float lastActionAccuracy;

    public static Trainer getInstance() {
        if (instance == null) {
            instance = new Trainer();
        }
        return instance;
    }

    private Trainer() {
        this.storage = new RecordingStorage();
        Path gameDir = FabricLoader.getInstance().getGameDir();
        this.modelsPath = gameDir.resolve("minimalai").resolve("models");
        try {
            Files.createDirectories(modelsPath);
        } catch (IOException e) {
            MinimalAI.LOGGER.error("Failed to create models directory", e);
        }
    }

    /**
     * Start training on all recordings in a background thread.
     */
    public void startTraining() {
        if (isTraining.get()) {
            MinimalAI.LOGGER.warn("Already training!");
            return;
        }

        List<RecordingStorage.RecordingInfo> recordingInfos = storage.listRecordings();
        if (recordingInfos.isEmpty()) {
            MinimalAI.LOGGER.warn("No recordings found to train on");
            return;
        }

        isTraining.set(true);
        shouldStop.set(false);

        new Thread(() -> {
            try {
                train(recordingInfos);
            } catch (Exception e) {
                MinimalAI.LOGGER.error("Training failed", e);
            } finally {
                isTraining.set(false);
            }
        }, "MinimalAI-Trainer").start();
    }

    /**
     * Stop training.
     */
    public void stopTraining() {
        if (!isTraining.get()) return;
        shouldStop.set(true);
        MinimalAI.LOGGER.info("Stopping training...");
    }

    private void train(List<RecordingStorage.RecordingInfo> recordingInfos) {
        MinimalAI.LOGGER.info("=== Training Started ===");
        MinimalAI.LOGGER.info("Recordings: {}", recordingInfos.size());
        MinimalAI.LOGGER.info("Epochs: {}, LR: {}, Batch: {}, CamWeight: {}",
            epochs, learningRate, batchSize, cameraWeight);

        // Initialize network
        manager = NDManager.newBaseManager();
        network = new SimpleNetwork(manager);

        // Load all frames from recordings
        List<Recording.Frame> allFrames = new ArrayList<>();
        for (RecordingStorage.RecordingInfo info : recordingInfos) {
            try {
                Recording recording = storage.load(info.name);
                allFrames.addAll(recording.getFrames());
                MinimalAI.LOGGER.info("Loaded {} frames from {}", recording.getFrameCount(), info.name);
            } catch (IOException e) {
                MinimalAI.LOGGER.error("Failed to load recording: {}", info.name, e);
            }
        }

        if (allFrames.isEmpty()) {
            MinimalAI.LOGGER.warn("No frames to train on");
            return;
        }

        MinimalAI.LOGGER.info("Total frames: {}", allFrames.size());

        // Training loop
        long startTime = System.currentTimeMillis();

        // Analyze camera distribution in training data
        if (!allFrames.isEmpty()) {
            float sumYaw = 0, sumPitch = 0;
            float sumYawSq = 0, sumPitchSq = 0;
            float minYaw = Float.MAX_VALUE, maxYaw = Float.MIN_VALUE;
            float minPitch = Float.MAX_VALUE, maxPitch = Float.MIN_VALUE;
            int nonZeroCount = 0;

            for (Recording.Frame f : allFrames) {
                if (f.continuousActions != null && f.continuousActions.length >= 2) {
                    float yaw = f.continuousActions[0];
                    float pitch = f.continuousActions[1];
                    sumYaw += yaw;
                    sumPitch += pitch;
                    sumYawSq += yaw * yaw;
                    sumPitchSq += pitch * pitch;
                    minYaw = Math.min(minYaw, yaw);
                    maxYaw = Math.max(maxYaw, yaw);
                    minPitch = Math.min(minPitch, pitch);
                    maxPitch = Math.max(maxPitch, pitch);
                    if (Math.abs(yaw) > 0.01f || Math.abs(pitch) > 0.01f) nonZeroCount++;
                }
            }

            int n = allFrames.size();
            float meanYaw = sumYaw / n;
            float meanPitch = sumPitch / n;
            float stdYaw = (float) Math.sqrt(sumYawSq / n - meanYaw * meanYaw);
            float stdPitch = (float) Math.sqrt(sumPitchSq / n - meanPitch * meanPitch);

            MinimalAI.LOGGER.info("[DATA] Camera stats ({} frames, {} non-zero):", n, nonZeroCount);
            MinimalAI.LOGGER.info("[DATA]   Yaw:   mean={}, std={}, range=[{}, {}]",
                String.format("%.4f", meanYaw), String.format("%.4f", stdYaw),
                String.format("%.3f", minYaw), String.format("%.3f", maxYaw));
            MinimalAI.LOGGER.info("[DATA]   Pitch: mean={}, std={}, range=[{}, {}]",
                String.format("%.4f", meanPitch), String.format("%.4f", stdPitch),
                String.format("%.3f", minPitch), String.format("%.3f", maxPitch));
        }

        for (int epoch = 0; epoch < epochs && !shouldStop.get(); epoch++) {
            currentEpoch = epoch;
            long epochStart = System.currentTimeMillis();

            // Shuffle frames
            Collections.shuffle(allFrames);

            float epochActionLoss = 0;
            float epochCameraLoss = 0;
            int correct = 0;
            int total = 0;
            int batchCount = 0;

            MinimalAI.LOGGER.info("[Epoch {}/{}] Starting... ({} frames, {} batches expected)",
                epoch + 1, epochs, allFrames.size(), (allFrames.size() + batchSize - 1) / batchSize);

            for (int i = 0; i < allFrames.size() && !shouldStop.get(); i++) {
                Recording.Frame frame = allFrames.get(i);

                // Forward + backward
                float[] losses = network.backward(
                    frame.gameState,
                    booleanToFloat(frame.discreteActions),
                    frame.continuousActions,
                    cameraWeight
                );

                epochActionLoss += losses[0];
                epochCameraLoss += losses[1];

                // Calculate accuracy
                float[][] outputs = network.forward(frame.gameState);
                float[] probs = network.getActionProbabilities(outputs[0]);
                for (int j = 0; j < probs.length; j++) {
                    boolean predicted = probs[j] > 0.5f;
                    boolean actual = frame.discreteActions[j];
                    if (predicted == actual) correct++;
                    total++;
                }

                // Apply gradients every batch
                if ((i + 1) % batchSize == 0 || i == allFrames.size() - 1) {
                    network.applyGradients(learningRate, Math.min(batchSize, (i % batchSize) + 1));
                    batchCount++;

                    // Log progress every 10 batches
                    if (batchCount % 10 == 0) {
                        float batchActionLoss = epochActionLoss / (i + 1);
                        float batchAcc = (float) correct / total * 100;
                        MinimalAI.LOGGER.info("[Epoch {}/{}] Batch {}: frame {}/{} | loss={} | acc={}%",
                            epoch + 1, epochs, batchCount, i + 1, allFrames.size(),
                            String.format("%.4f", batchActionLoss),
                            String.format("%.1f", batchAcc));
                    }
                }
            }

            // Epoch stats
            lastActionLoss = epochActionLoss / allFrames.size();
            lastCameraLoss = epochCameraLoss / allFrames.size();
            lastActionAccuracy = (float) correct / total * 100;
            long epochDuration = System.currentTimeMillis() - epochStart;

            MinimalAI.LOGGER.info("[Epoch {}/{}] DONE in {}ms | action_loss={} | camera_loss={} | accuracy={}%",
                epoch + 1, epochs, epochDuration,
                String.format("%.4f", lastActionLoss),
                String.format("%.4f", lastCameraLoss),
                String.format("%.1f", lastActionAccuracy));
        }

        long duration = System.currentTimeMillis() - startTime;

        if (shouldStop.get()) {
            MinimalAI.LOGGER.info("Training stopped by user");
        } else {
            MinimalAI.LOGGER.info("=== Training Complete ===");
        }

        MinimalAI.LOGGER.info("Duration: {}s", duration / 1000);

        // Save model
        try {
            String modelName = "model_" + System.currentTimeMillis() + ".bin";
            network.save(modelsPath.resolve(modelName));
        } catch (IOException e) {
            MinimalAI.LOGGER.error("Failed to save model", e);
        }

        // Cleanup
        manager.close();
    }

    private float[] booleanToFloat(boolean[] bools) {
        float[] floats = new float[bools.length];
        for (int i = 0; i < bools.length; i++) {
            floats[i] = bools[i] ? 1.0f : 0.0f;
        }
        return floats;
    }

    // === Getters/Setters ===

    public boolean isTraining() {
        return isTraining.get();
    }

    public int getCurrentEpoch() {
        return currentEpoch;
    }

    public int getTotalEpochs() {
        return epochs;
    }

    public float getLastActionLoss() {
        return lastActionLoss;
    }

    public float getLastCameraLoss() {
        return lastCameraLoss;
    }

    public float getLastActionAccuracy() {
        return lastActionAccuracy;
    }

    public void setLearningRate(float lr) {
        this.learningRate = lr;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public void setCameraWeight(float weight) {
        this.cameraWeight = weight;
    }

    public Path getModelsPath() {
        return modelsPath;
    }

    public SimpleNetwork getNetwork() {
        return network;
    }
}
