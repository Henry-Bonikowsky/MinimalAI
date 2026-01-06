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

/**
 * REINFORCE trainer with interleaved Behavior Cloning.
 *
 * Training loop:
 * 1. BC pass: Train on recordings to maintain action vocabulary
 * 2. RL episode: Let AI play, collect experiences
 * 3. REINFORCE update: Update policy based on episode reward
 * 4. Repeat
 */
public class RLTrainer {
    private static RLTrainer instance;

    // Training parameters
    private float rlLearningRate = 0.0001f;  // Lower for RL stability
    private float bcLearningRate = 0.001f;   // Same as original BC
    private float gamma = 0.99f;             // Discount factor
    private int bcBatchSize = 32;
    private int bcStepsPerCycle = 100;       // BC steps between RL episodes
    private float cameraWeight = 1.0f;

    private final RecordingStorage storage;
    private final Path modelsPath;

    private SimpleNetworkRL network;
    private NDManager manager;

    // Training state
    private boolean isTraining = false;
    private int rlEpisodeCount = 0;
    private int bcCycleCount = 0;
    private float lastEpisodeReward = 0;
    private float avgReward = 0;  // Exponential moving average

    // Loaded recordings for BC
    private List<Recording.Frame> bcFrames;

    public static RLTrainer getInstance() {
        if (instance == null) {
            instance = new RLTrainer();
        }
        return instance;
    }

    private RLTrainer() {
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
     * Initialize for RL training.
     * Loads recordings for BC interleaving.
     */
    public boolean initialize() {
        // Load recordings for BC
        List<RecordingStorage.RecordingInfo> recordingInfos = storage.listRecordings();
        if (recordingInfos.isEmpty()) {
            MinimalAI.LOGGER.warn("No recordings found - BC interleaving will be disabled");
            bcFrames = new ArrayList<>();
        } else {
            bcFrames = new ArrayList<>();
            for (RecordingStorage.RecordingInfo info : recordingInfos) {
                try {
                    Recording recording = storage.load(info.name);
                    bcFrames.addAll(recording.getFrames());
                    MinimalAI.LOGGER.info("Loaded {} frames from {} for BC", recording.getFrameCount(), info.name);
                } catch (IOException e) {
                    MinimalAI.LOGGER.error("Failed to load recording: {}", info.name, e);
                }
            }
            MinimalAI.LOGGER.info("Total BC frames: {}", bcFrames.size());
        }

        // Initialize network
        manager = NDManager.newBaseManager();
        network = new SimpleNetworkRL(manager);

        isTraining = true;
        rlEpisodeCount = 0;
        bcCycleCount = 0;

        MinimalAI.LOGGER.info("=== RL Training Initialized ===");
        MinimalAI.LOGGER.info("RL LR: {}, BC LR: {}, Gamma: {}", rlLearningRate, bcLearningRate, gamma);
        MinimalAI.LOGGER.info("BC batch: {}, BC steps/cycle: {}", bcBatchSize, bcStepsPerCycle);

        return true;
    }

    /**
     * Run a BC training pass.
     * Called between RL episodes to prevent catastrophic forgetting.
     */
    public void runBCPass() {
        if (bcFrames.isEmpty()) {
            MinimalAI.LOGGER.debug("No BC frames, skipping BC pass");
            return;
        }

        bcCycleCount++;
        Collections.shuffle(bcFrames);

        float totalLoss = 0;
        int steps = Math.min(bcStepsPerCycle, bcFrames.size());

        for (int i = 0; i < steps; i++) {
            Recording.Frame frame = bcFrames.get(i);

            // Need to convert single frame to stacked format
            // For BC, we'll just replicate the frame 4 times (not ideal but simple)
            float[] stackedInput = new float[SimpleNetworkRL.INPUT_SIZE];
            for (int f = 0; f < SimpleNetworkRL.FRAME_STACK; f++) {
                System.arraycopy(frame.gameState, 0, stackedInput,
                    f * SimpleNetworkRL.FRAME_SIZE, SimpleNetworkRL.FRAME_SIZE);
            }

            // Forward pass
            SimpleNetworkRL.ForwardResult fwd = network.forward(stackedInput);

            // Compute BC loss and backward (using cross-entropy for actions)
            float loss = computeBCLoss(fwd, frame.discreteActions, frame.continuousActions);
            totalLoss += loss;

            // Backward pass for BC
            backwardBC(stackedInput, frame.discreteActions, frame.continuousActions);

            // Apply gradients every batch
            if ((i + 1) % bcBatchSize == 0 || i == steps - 1) {
                network.applyGradients(bcLearningRate, Math.min(bcBatchSize, (i % bcBatchSize) + 1));
            }
        }

        float avgLoss = totalLoss / steps;
        MinimalAI.LOGGER.info("[BC Cycle {}] {} steps, avg loss: {}",
            bcCycleCount, steps, String.format("%.4f", avgLoss));
    }

    /**
     * Process a completed RL episode.
     * Runs REINFORCE update on the collected experiences.
     */
    public void processEpisode(ExperienceBuffer buffer) {
        if (!buffer.isComplete()) {
            MinimalAI.LOGGER.warn("Attempted to process incomplete episode");
            return;
        }

        rlEpisodeCount++;
        lastEpisodeReward = buffer.getEpisodeReward();

        // Update running average
        if (rlEpisodeCount == 1) {
            avgReward = lastEpisodeReward;
        } else {
            avgReward = 0.95f * avgReward + 0.05f * lastEpisodeReward;
        }

        List<ExperienceBuffer.Experience> experiences = buffer.getExperiences();
        if (experiences.isEmpty()) {
            MinimalAI.LOGGER.warn("Episode {} had no experiences", rlEpisodeCount);
            return;
        }

        // Compute advantages
        float[] advantages = buffer.computeAdvantages(gamma);
        float[] normalizedAdvantages = buffer.normalizeAdvantages(advantages);

        // REINFORCE update
        for (int i = 0; i < experiences.size(); i++) {
            ExperienceBuffer.Experience exp = experiences.get(i);
            float advantage = normalizedAdvantages[i];

            // Policy gradient backward
            network.backward(exp.stackedState, exp.actions, advantage, exp.logProbs);

            // Value head backward (target = return)
            float returnValue = (float) Math.pow(gamma, experiences.size() - 1 - i) * buffer.getEpisodeReward();
            network.backwardValue(exp.stackedState, returnValue);
        }

        // Apply gradients
        network.applyGradients(rlLearningRate, experiences.size());

        MinimalAI.LOGGER.info("[RL Episode {}] Reward: {}, Steps: {}, Avg Reward: {}",
            rlEpisodeCount, lastEpisodeReward, experiences.size(), String.format("%.3f", avgReward));
    }

    /**
     * Compute BC loss (cross-entropy for actions + MSE for camera).
     */
    private float computeBCLoss(SimpleNetworkRL.ForwardResult fwd, boolean[] targetActions, float[] targetCamera) {
        float loss = 0;

        // Action cross-entropy
        for (int i = 0; i < SimpleNetworkRL.ACTION_SIZE; i++) {
            float p = Math.max(1e-7f, Math.min(1 - 1e-7f, fwd.actionProbs[i]));
            float y = targetActions[i] ? 1.0f : 0.0f;
            loss -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
        }
        loss /= SimpleNetworkRL.ACTION_SIZE;

        // Camera MSE
        float cameraLoss = 0;
        for (int i = 0; i < SimpleNetworkRL.CAMERA_SIZE; i++) {
            float diff = fwd.camera[i] - targetCamera[i];
            cameraLoss += diff * diff;
        }
        cameraLoss /= SimpleNetworkRL.CAMERA_SIZE;

        return loss + cameraWeight * cameraLoss;
    }

    /**
     * Backward pass for BC (simplified - reuses network's backward logic).
     */
    private void backwardBC(float[] stackedInput, boolean[] targetActions, float[] targetCamera) {
        // Forward pass
        SimpleNetworkRL.ForwardResult fwd = network.forward(stackedInput);

        // Compute gradients manually for BC
        // This is a simplified version - ideally we'd refactor to share code with SimpleNetwork
        float[] actionGrad = new float[SimpleNetworkRL.ACTION_SIZE];
        for (int i = 0; i < SimpleNetworkRL.ACTION_SIZE; i++) {
            float p = fwd.actionProbs[i];
            p = Math.max(1e-7f, Math.min(1 - 1e-7f, p));
            float y = targetActions[i] ? 1.0f : 0.0f;
            actionGrad[i] = p - y;  // BCE gradient
        }

        // Use the network's backward with advantage=1 (effectively just the gradient direction)
        // This is a bit of a hack - we're treating BC as RL with the "correct" action always taken
        network.backward(stackedInput, targetActions, 1.0f, actionGrad);
    }

    /**
     * Save the current model.
     */
    public void saveModel() {
        if (network == null) return;

        try {
            String modelName = "rl_model_" + System.currentTimeMillis() + ".bin";
            network.save(modelsPath.resolve(modelName));
        } catch (IOException e) {
            MinimalAI.LOGGER.error("Failed to save RL model", e);
        }
    }

    /**
     * Stop training and clean up.
     */
    public void stop() {
        if (!isTraining) return;

        isTraining = false;
        saveModel();

        if (manager != null) {
            manager.close();
            manager = null;
        }
        network = null;
        bcFrames = null;

        MinimalAI.LOGGER.info("=== RL Training Stopped ===");
        MinimalAI.LOGGER.info("Episodes: {}, BC Cycles: {}, Final Avg Reward: {}",
            rlEpisodeCount, bcCycleCount, String.format("%.3f", avgReward));
    }

    // === Getters ===

    public SimpleNetworkRL getNetwork() {
        return network;
    }

    public boolean isTraining() {
        return isTraining;
    }

    public int getRLEpisodeCount() {
        return rlEpisodeCount;
    }

    public int getBCCycleCount() {
        return bcCycleCount;
    }

    public float getLastEpisodeReward() {
        return lastEpisodeReward;
    }

    public float getAvgReward() {
        return avgReward;
    }

    public Path getModelsPath() {
        return modelsPath;
    }

    // === Setters ===

    public void setRLLearningRate(float lr) {
        this.rlLearningRate = lr;
    }

    public void setBCLearningRate(float lr) {
        this.bcLearningRate = lr;
    }

    public void setGamma(float gamma) {
        this.gamma = gamma;
    }

    public void setBCStepsPerCycle(int steps) {
        this.bcStepsPerCycle = steps;
    }
}
