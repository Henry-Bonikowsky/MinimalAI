package com.minimalai.ai;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.minimalai.MinimalAI;

import java.io.*;
import java.nio.file.Path;

/**
 * Neural network for reinforcement learning with frame stacking and value head.
 *
 * Architecture:
 *   Input (256 = 4 frames × 64) → Hidden1 (256, ReLU) → Hidden2 (128, ReLU) → Split
 *                                                                              ├→ Action Head (20 logits)
 *                                                                              ├→ Camera Head (2 floats)
 *                                                                              └→ Value Head (1 float)
 *
 * Frame stacking provides temporal context - the network sees the last 4 game states.
 * Value head estimates state value for REINFORCE baseline.
 */
public class SimpleNetworkRL {
    public static final int FRAME_SIZE = 64;
    public static final int FRAME_STACK = 4;
    public static final int INPUT_SIZE = FRAME_SIZE * FRAME_STACK;  // 256
    public static final int HIDDEN1_SIZE = 256;
    public static final int HIDDEN2_SIZE = 128;
    public static final int ACTION_SIZE = 20;
    public static final int CAMERA_SIZE = 2;

    private final NDManager manager;

    // Frame buffer for stacking
    private final float[][] frameBuffer = new float[FRAME_STACK][FRAME_SIZE];
    private int frameBufferIndex = 0;
    private boolean frameBufferFilled = false;

    // Weights and biases
    private NDArray w1, b1;  // Input → Hidden1
    private NDArray w2, b2;  // Hidden1 → Hidden2
    private NDArray wAction, bAction;  // Hidden2 → Actions
    private NDArray wCamera, bCamera;  // Hidden2 → Camera
    private NDArray wValue, bValue;    // Hidden2 → Value (new)

    // Gradients
    private NDArray dw1, db1, dw2, db2;
    private NDArray dwAction, dbAction, dwCamera, dbCamera;
    private NDArray dwValue, dbValue;

    // Cache for backward pass
    private NDArray lastInput;
    private NDArray lastHidden1, lastHidden1PreRelu;
    private NDArray lastHidden2, lastHidden2PreRelu;

    public SimpleNetworkRL(NDManager manager) {
        this.manager = manager;
        initializeWeights();
        clearFrameBuffer();
    }

    private void initializeWeights() {
        float scale1 = (float) Math.sqrt(2.0 / (INPUT_SIZE + HIDDEN1_SIZE));
        float scale2 = (float) Math.sqrt(2.0 / (HIDDEN1_SIZE + HIDDEN2_SIZE));
        float scaleAction = (float) Math.sqrt(2.0 / (HIDDEN2_SIZE + ACTION_SIZE));
        float scaleCamera = (float) Math.sqrt(2.0 / (HIDDEN2_SIZE + CAMERA_SIZE));
        float scaleValue = (float) Math.sqrt(2.0 / (HIDDEN2_SIZE + 1));

        w1 = manager.randomNormal(0, scale1, new Shape(INPUT_SIZE, HIDDEN1_SIZE), DataType.FLOAT32);
        b1 = manager.zeros(new Shape(HIDDEN1_SIZE), DataType.FLOAT32);

        w2 = manager.randomNormal(0, scale2, new Shape(HIDDEN1_SIZE, HIDDEN2_SIZE), DataType.FLOAT32);
        b2 = manager.zeros(new Shape(HIDDEN2_SIZE), DataType.FLOAT32);

        wAction = manager.randomNormal(0, scaleAction, new Shape(HIDDEN2_SIZE, ACTION_SIZE), DataType.FLOAT32);
        bAction = manager.zeros(new Shape(ACTION_SIZE), DataType.FLOAT32);

        wCamera = manager.randomNormal(0, scaleCamera, new Shape(HIDDEN2_SIZE, CAMERA_SIZE), DataType.FLOAT32);
        bCamera = manager.zeros(new Shape(CAMERA_SIZE), DataType.FLOAT32);

        wValue = manager.randomNormal(0, scaleValue, new Shape(HIDDEN2_SIZE, 1), DataType.FLOAT32);
        bValue = manager.zeros(new Shape(1), DataType.FLOAT32);

        zeroGradients();

        int params = INPUT_SIZE * HIDDEN1_SIZE + HIDDEN1_SIZE +
                     HIDDEN1_SIZE * HIDDEN2_SIZE + HIDDEN2_SIZE +
                     HIDDEN2_SIZE * ACTION_SIZE + ACTION_SIZE +
                     HIDDEN2_SIZE * CAMERA_SIZE + CAMERA_SIZE +
                     HIDDEN2_SIZE * 1 + 1;
        MinimalAI.LOGGER.info("RL Network initialized: {} params (256→256→128→outputs+value)", params);
    }

    /**
     * Clear the frame buffer (call on episode reset).
     */
    public void clearFrameBuffer() {
        for (int i = 0; i < FRAME_STACK; i++) {
            for (int j = 0; j < FRAME_SIZE; j++) {
                frameBuffer[i][j] = 0;
            }
        }
        frameBufferIndex = 0;
        frameBufferFilled = false;
    }

    /**
     * Push a new frame to the buffer.
     */
    public void pushFrame(float[] frame) {
        System.arraycopy(frame, 0, frameBuffer[frameBufferIndex], 0, FRAME_SIZE);
        frameBufferIndex = (frameBufferIndex + 1) % FRAME_STACK;
        if (frameBufferIndex == 0) {
            frameBufferFilled = true;
        }
    }

    /**
     * Get stacked frames as a single input array.
     * Returns frames in chronological order (oldest to newest).
     */
    public float[] getStackedInput() {
        float[] stacked = new float[INPUT_SIZE];
        int outIdx = 0;
        for (int i = 0; i < FRAME_STACK; i++) {
            int bufIdx = (frameBufferIndex + i) % FRAME_STACK;
            System.arraycopy(frameBuffer[bufIdx], 0, stacked, outIdx, FRAME_SIZE);
            outIdx += FRAME_SIZE;
        }
        return stacked;
    }

    /**
     * Forward pass with automatic frame stacking.
     * Pushes the frame, then runs forward on stacked input.
     */
    public ForwardResult forwardWithFrame(float[] frame) {
        pushFrame(frame);
        return forward(getStackedInput());
    }

    /**
     * Forward pass.
     * @param input Stacked game states (256 floats = 4 × 64)
     * @return ForwardResult containing actions, camera, value, and log probabilities
     */
    public ForwardResult forward(float[] input) {
        NDArray x = manager.create(input);
        return forward(x);
    }

    /**
     * Forward pass with NDArray input.
     */
    public ForwardResult forward(NDArray input) {
        lastInput = input;

        // Hidden layer 1: ReLU(input @ w1 + b1)
        lastHidden1PreRelu = input.matMul(w1).add(b1);
        lastHidden1 = relu(lastHidden1PreRelu);

        // Hidden layer 2: ReLU(hidden1 @ w2 + b2)
        lastHidden2PreRelu = lastHidden1.matMul(w2).add(b2);
        lastHidden2 = relu(lastHidden2PreRelu);

        // Action head (logits)
        NDArray actionLogits = lastHidden2.matMul(wAction).add(bAction);

        // Camera head
        NDArray cameraOutput = lastHidden2.matMul(wCamera).add(bCamera);

        // Value head
        NDArray valueOutput = lastHidden2.matMul(wValue).add(bValue);

        float[] actionLogitsArr = actionLogits.toFloatArray();
        float[] cameraArr = cameraOutput.toFloatArray();
        float value = valueOutput.toFloatArray()[0];

        // Compute action probabilities
        float[] actionProbs = new float[ACTION_SIZE];
        for (int i = 0; i < ACTION_SIZE; i++) {
            actionProbs[i] = sigmoid(actionLogitsArr[i]);
        }

        return new ForwardResult(actionLogitsArr, actionProbs, cameraArr, value);
    }

    /**
     * Sample actions from probabilities and compute log probabilities.
     */
    public SampleResult sampleActions(ForwardResult fwd) {
        boolean[] actions = new boolean[ACTION_SIZE];
        float[] logProbs = new float[ACTION_SIZE];

        for (int i = 0; i < ACTION_SIZE; i++) {
            float p = fwd.actionProbs[i];
            // Clamp for numerical stability
            p = Math.max(1e-7f, Math.min(1 - 1e-7f, p));

            boolean action = Math.random() < p;
            actions[i] = action;

            // Log probability of the sampled action
            if (action) {
                logProbs[i] = (float) Math.log(p);
            } else {
                logProbs[i] = (float) Math.log(1 - p);
            }
        }

        return new SampleResult(actions, logProbs, fwd.camera, fwd.value);
    }

    /**
     * Backward pass for REINFORCE with baseline.
     * @param stackedInput The stacked input (256 floats)
     * @param actions The actions that were taken
     * @param advantage The advantage (return - baseline)
     * @param actionLogProbs Log probabilities of taken actions
     */
    public void backward(float[] stackedInput, boolean[] actions, float advantage, float[] actionLogProbs) {
        // Forward pass to recompute cached values
        ForwardResult fwd = forward(stackedInput);

        // Policy gradient: ∇log(π(a|s)) * advantage
        float[] actionGrad = new float[ACTION_SIZE];
        for (int i = 0; i < ACTION_SIZE; i++) {
            float p = fwd.actionProbs[i];
            p = Math.max(1e-7f, Math.min(1 - 1e-7f, p));

            // Gradient of log probability w.r.t. logits
            // For sigmoid: d/dz log(σ(z)) = 1 - σ(z) if a=1, -σ(z) if a=0
            float gradLogProb;
            if (actions[i]) {
                gradLogProb = 1 - p;
            } else {
                gradLogProb = -p;
            }

            // Scale by advantage (negative because we want to maximize)
            actionGrad[i] = -gradLogProb * advantage;
        }

        // Value head gradient: MSE derivative, but we'll handle this separately
        // For now, focus on policy gradient

        NDArray dActionLogits = manager.create(actionGrad);

        // Backward through action head
        NDArray lastHidden2Col = lastHidden2.reshape(HIDDEN2_SIZE, 1);
        dwAction = dwAction.add(lastHidden2Col.matMul(dActionLogits.reshape(1, ACTION_SIZE)));
        dbAction = dbAction.add(dActionLogits);

        // Backward through hidden2
        NDArray dHidden2 = dActionLogits.reshape(1, ACTION_SIZE).matMul(wAction.transpose());
        NDArray dHidden2PreRelu = dHidden2.mul(reluGrad(lastHidden2PreRelu));

        // Gradients for layer 2
        NDArray lastHidden1Col = lastHidden1.reshape(HIDDEN1_SIZE, 1);
        dw2 = dw2.add(lastHidden1Col.matMul(dHidden2PreRelu.reshape(1, HIDDEN2_SIZE)));
        db2 = db2.add(dHidden2PreRelu);

        // Backward through hidden1
        NDArray dHidden1 = dHidden2PreRelu.reshape(1, HIDDEN2_SIZE).matMul(w2.transpose());
        NDArray dHidden1PreRelu = dHidden1.mul(reluGrad(lastHidden1PreRelu));

        // Gradients for layer 1
        dw1 = dw1.add(lastInput.reshape(INPUT_SIZE, 1).matMul(dHidden1PreRelu.reshape(1, HIDDEN1_SIZE)));
        db1 = db1.add(dHidden1PreRelu);
    }

    /**
     * Backward pass for value head (TD or Monte Carlo value target).
     */
    public void backwardValue(float[] stackedInput, float valueTarget) {
        ForwardResult fwd = forward(stackedInput);

        // MSE gradient: 2 * (predicted - target)
        float valueGrad = 2 * (fwd.value - valueTarget);

        NDArray dValue = manager.create(new float[]{valueGrad});

        // Backward through value head
        NDArray lastHidden2Col = lastHidden2.reshape(HIDDEN2_SIZE, 1);
        dwValue = dwValue.add(lastHidden2Col.matMul(dValue.reshape(1, 1)));
        dbValue = dbValue.add(dValue);

        // Note: We could also backprop through shared layers, but keeping it simple for now
    }

    /**
     * Apply gradients with SGD.
     */
    public void applyGradients(float learningRate, int batchSize) {
        float scale = learningRate / batchSize;

        w1 = w1.sub(dw1.mul(scale));
        b1 = b1.sub(db1.mul(scale));
        w2 = w2.sub(dw2.mul(scale));
        b2 = b2.sub(db2.mul(scale));
        wAction = wAction.sub(dwAction.mul(scale));
        bAction = bAction.sub(dbAction.mul(scale));
        wCamera = wCamera.sub(dwCamera.mul(scale));
        bCamera = bCamera.sub(dbCamera.mul(scale));
        wValue = wValue.sub(dwValue.mul(scale));
        bValue = bValue.sub(dbValue.mul(scale));

        zeroGradients();
    }

    public void zeroGradients() {
        dw1 = manager.zeros(w1.getShape(), DataType.FLOAT32);
        db1 = manager.zeros(b1.getShape(), DataType.FLOAT32);
        dw2 = manager.zeros(w2.getShape(), DataType.FLOAT32);
        db2 = manager.zeros(b2.getShape(), DataType.FLOAT32);
        dwAction = manager.zeros(wAction.getShape(), DataType.FLOAT32);
        dbAction = manager.zeros(bAction.getShape(), DataType.FLOAT32);
        dwCamera = manager.zeros(wCamera.getShape(), DataType.FLOAT32);
        dbCamera = manager.zeros(bCamera.getShape(), DataType.FLOAT32);
        dwValue = manager.zeros(wValue.getShape(), DataType.FLOAT32);
        dbValue = manager.zeros(bValue.getShape(), DataType.FLOAT32);
    }

    // === Activation functions ===

    private NDArray relu(NDArray x) {
        return x.maximum(0);
    }

    private NDArray reluGrad(NDArray x) {
        return x.gt(0).toType(DataType.FLOAT32, false);
    }

    private float sigmoid(float x) {
        return 1.0f / (1.0f + (float) Math.exp(-x));
    }

    // === Save/Load ===

    public void save(Path path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(path.toFile())))) {
            // Write version marker for RL network
            out.writeInt(2);  // Version 2 = RL network with value head

            writeArray(out, w1);
            writeArray(out, b1);
            writeArray(out, w2);
            writeArray(out, b2);
            writeArray(out, wAction);
            writeArray(out, bAction);
            writeArray(out, wCamera);
            writeArray(out, bCamera);
            writeArray(out, wValue);
            writeArray(out, bValue);
        }
        MinimalAI.LOGGER.info("RL Model saved to {}", path);
    }

    public void load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(path.toFile())))) {
            int version = in.readInt();
            if (version != 2) {
                throw new IOException("Invalid RL model version: " + version + " (expected 2)");
            }

            w1 = readArray(in, w1.getShape());
            b1 = readArray(in, b1.getShape());
            w2 = readArray(in, w2.getShape());
            b2 = readArray(in, b2.getShape());
            wAction = readArray(in, wAction.getShape());
            bAction = readArray(in, bAction.getShape());
            wCamera = readArray(in, wCamera.getShape());
            bCamera = readArray(in, bCamera.getShape());
            wValue = readArray(in, wValue.getShape());
            bValue = readArray(in, bValue.getShape());
        }
        MinimalAI.LOGGER.info("RL Model loaded from {}", path);
    }

    /**
     * Initialize from a BC model (SimpleNetwork).
     * Copies weights for shared layers, randomly initializes value head.
     */
    public void initFromBCModel(Path bcModelPath) throws IOException {
        // Load BC model into temporary arrays
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(bcModelPath.toFile())))) {

            // BC model has 64 input, we have 256
            // We can initialize by copying the BC weights 4 times (one per frame slot)
            // Or start fresh and rely on interleaved BC training

            // For now, just use random init - BC training will refine it
            MinimalAI.LOGGER.info("RL network using random initialization (BC interleaving will train it)");
        }
    }

    private void writeArray(DataOutputStream out, NDArray arr) throws IOException {
        float[] data = arr.toFloatArray();
        for (float v : data) {
            out.writeFloat(v);
        }
    }

    private NDArray readArray(DataInputStream in, Shape shape) throws IOException {
        int size = (int) shape.size();
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = in.readFloat();
        }
        return manager.create(data, shape);
    }

    public NDManager getManager() {
        return manager;
    }

    public boolean isFrameBufferReady() {
        return frameBufferFilled;
    }

    // === Result classes ===

    public static class ForwardResult {
        public final float[] actionLogits;
        public final float[] actionProbs;
        public final float[] camera;
        public final float value;

        public ForwardResult(float[] actionLogits, float[] actionProbs, float[] camera, float value) {
            this.actionLogits = actionLogits;
            this.actionProbs = actionProbs;
            this.camera = camera;
            this.value = value;
        }
    }

    public static class SampleResult {
        public final boolean[] actions;
        public final float[] logProbs;
        public final float[] camera;
        public final float value;

        public SampleResult(boolean[] actions, float[] logProbs, float[] camera, float value) {
            this.actions = actions;
            this.logProbs = logProbs;
            this.camera = camera;
            this.value = value;
        }
    }
}
