package com.minimalai.ai;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.minimalai.MinimalAI;

import java.io.*;
import java.nio.file.Path;

/**
 * Feedforward neural network for behavior cloning.
 *
 * Architecture:
 *   Input (64) → Hidden1 (256, ReLU) → Hidden2 (128, ReLU) → Split
 *                                                            ├→ Action Head (20 logits)
 *                                                            └→ Camera Head (2 floats)
 *
 * ~50K parameters total.
 */
public class SimpleNetwork {
    public static final int INPUT_SIZE = 64;
    public static final int HIDDEN1_SIZE = 256;
    public static final int HIDDEN2_SIZE = 128;
    public static final int ACTION_SIZE = 20;
    public static final int CAMERA_SIZE = 2;

    private final NDManager manager;

    // Weights and biases
    private NDArray w1;  // Input → Hidden1 (64 x 256)
    private NDArray b1;  // Hidden1 bias (256)
    private NDArray w2;  // Hidden1 → Hidden2 (256 x 128)
    private NDArray b2;  // Hidden2 bias (128)
    private NDArray wAction;  // Hidden2 → Actions (128 x 20)
    private NDArray bAction;  // Action bias (20)
    private NDArray wCamera;  // Hidden2 → Camera (128 x 2)
    private NDArray bCamera;  // Camera bias (2)

    // Gradients (for training)
    private NDArray dw1, db1, dw2, db2, dwAction, dbAction, dwCamera, dbCamera;

    // Cache for backward pass
    private NDArray lastInput;
    private NDArray lastHidden1, lastHidden1PreRelu;
    private NDArray lastHidden2, lastHidden2PreRelu;

    public SimpleNetwork(NDManager manager) {
        this.manager = manager;
        initializeWeights();
    }

    private void initializeWeights() {
        // Xavier initialization
        float scale1 = (float) Math.sqrt(2.0 / (INPUT_SIZE + HIDDEN1_SIZE));
        float scale2 = (float) Math.sqrt(2.0 / (HIDDEN1_SIZE + HIDDEN2_SIZE));
        float scaleAction = (float) Math.sqrt(2.0 / (HIDDEN2_SIZE + ACTION_SIZE));
        float scaleCamera = (float) Math.sqrt(2.0 / (HIDDEN2_SIZE + CAMERA_SIZE));

        w1 = manager.randomNormal(0, scale1, new Shape(INPUT_SIZE, HIDDEN1_SIZE), DataType.FLOAT32);
        b1 = manager.zeros(new Shape(HIDDEN1_SIZE), DataType.FLOAT32);

        w2 = manager.randomNormal(0, scale2, new Shape(HIDDEN1_SIZE, HIDDEN2_SIZE), DataType.FLOAT32);
        b2 = manager.zeros(new Shape(HIDDEN2_SIZE), DataType.FLOAT32);

        wAction = manager.randomNormal(0, scaleAction, new Shape(HIDDEN2_SIZE, ACTION_SIZE), DataType.FLOAT32);
        bAction = manager.zeros(new Shape(ACTION_SIZE), DataType.FLOAT32);

        wCamera = manager.randomNormal(0, scaleCamera, new Shape(HIDDEN2_SIZE, CAMERA_SIZE), DataType.FLOAT32);
        bCamera = manager.zeros(new Shape(CAMERA_SIZE), DataType.FLOAT32);

        // Initialize gradients
        zeroGradients();

        int params = INPUT_SIZE * HIDDEN1_SIZE + HIDDEN1_SIZE +
                     HIDDEN1_SIZE * HIDDEN2_SIZE + HIDDEN2_SIZE +
                     HIDDEN2_SIZE * ACTION_SIZE + ACTION_SIZE +
                     HIDDEN2_SIZE * CAMERA_SIZE + CAMERA_SIZE;
        MinimalAI.LOGGER.info("Network initialized: {} params (64→256→128→outputs)", params);
    }

    /**
     * Forward pass.
     * @param input Game state (64 floats)
     * @return [actionLogits (20), cameraOutput (2)]
     */
    public float[][] forward(float[] input) {
        NDArray x = manager.create(input);
        return forward(x);
    }

    /**
     * Forward pass with NDArray input.
     * Caches values for backward pass.
     */
    public float[][] forward(NDArray input) {
        lastInput = input;

        // Hidden layer 1: ReLU(input @ w1 + b1)
        lastHidden1PreRelu = input.matMul(w1).add(b1);
        lastHidden1 = relu(lastHidden1PreRelu);

        // Hidden layer 2: ReLU(hidden1 @ w2 + b2)
        lastHidden2PreRelu = lastHidden1.matMul(w2).add(b2);
        lastHidden2 = relu(lastHidden2PreRelu);

        // Action head
        NDArray actionLogits = lastHidden2.matMul(wAction).add(bAction);

        // Camera head
        NDArray cameraOutput = lastHidden2.matMul(wCamera).add(bCamera);

        float[] actions = actionLogits.toFloatArray();
        float[] camera = cameraOutput.toFloatArray();

        return new float[][] { actions, camera };
    }

    /**
     * Backward pass - computes gradients.
     * @param input Input game state (64 floats)
     * @param actionTarget Target action probabilities (20)
     * @param cameraTarget Target camera deltas (2)
     * @param cameraWeight Weight for camera loss
     * @return [actionLoss, cameraLoss]
     */
    public float[] backward(float[] input, float[] actionTarget, float[] cameraTarget, float cameraWeight) {
        // Forward pass to get predictions and cache values for backprop
        float[][] outputs = forward(input);
        float[] actionLogits = outputs[0];
        float[] cameraPred = outputs[1];

        // === Action Loss (Cross-entropy with sigmoid) ===
        float actionLoss = 0;
        float[] actionGrad = new float[ACTION_SIZE];

        for (int i = 0; i < ACTION_SIZE; i++) {
            float p = sigmoid(actionLogits[i]);
            float y = actionTarget[i];

            // Numerical stability
            p = Math.max(1e-7f, Math.min(1 - 1e-7f, p));

            actionLoss -= y * Math.log(p) + (1 - y) * Math.log(1 - p);

            // Gradient of BCE w.r.t. logits = p - y (for sigmoid output)
            actionGrad[i] = p - y;
        }
        actionLoss /= ACTION_SIZE;

        // === Camera Loss (MSE) ===
        float cameraLoss = 0;
        float[] cameraGrad = new float[CAMERA_SIZE];

        for (int i = 0; i < CAMERA_SIZE; i++) {
            float diff = cameraPred[i] - cameraTarget[i];
            cameraLoss += diff * diff;
            cameraGrad[i] = 2 * diff * cameraWeight / CAMERA_SIZE;
        }
        cameraLoss /= CAMERA_SIZE;

        // === Backward through heads ===
        NDArray dActionLogits = manager.create(actionGrad);
        NDArray dCameraOutput = manager.create(cameraGrad);

        // Reshape hidden2 for outer product
        NDArray lastHidden2Col = lastHidden2.reshape(HIDDEN2_SIZE, 1);

        // Gradients for action head
        dwAction = dwAction.add(lastHidden2Col.matMul(dActionLogits.reshape(1, ACTION_SIZE)));
        dbAction = dbAction.add(dActionLogits);

        // Gradients for camera head
        dwCamera = dwCamera.add(lastHidden2Col.matMul(dCameraOutput.reshape(1, CAMERA_SIZE)));
        dbCamera = dbCamera.add(dCameraOutput);

        // Backward through hidden2
        NDArray dHidden2 = dActionLogits.reshape(1, ACTION_SIZE).matMul(wAction.transpose())
                          .add(dCameraOutput.reshape(1, CAMERA_SIZE).matMul(wCamera.transpose()));

        // ReLU backward for hidden2
        NDArray dHidden2PreRelu = dHidden2.mul(reluGrad(lastHidden2PreRelu));

        // Gradients for layer 2
        NDArray lastHidden1Col = lastHidden1.reshape(HIDDEN1_SIZE, 1);
        dw2 = dw2.add(lastHidden1Col.matMul(dHidden2PreRelu.reshape(1, HIDDEN2_SIZE)));
        db2 = db2.add(dHidden2PreRelu);

        // Backward through hidden1
        NDArray dHidden1 = dHidden2PreRelu.reshape(1, HIDDEN2_SIZE).matMul(w2.transpose());

        // ReLU backward for hidden1
        NDArray dHidden1PreRelu = dHidden1.mul(reluGrad(lastHidden1PreRelu));

        // Gradients for layer 1
        dw1 = dw1.add(lastInput.reshape(INPUT_SIZE, 1).matMul(dHidden1PreRelu.reshape(1, HIDDEN1_SIZE)));
        db1 = db1.add(dHidden1PreRelu);

        return new float[] { actionLoss, cameraLoss };
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
    }

    /**
     * Get action probabilities (sigmoid of logits).
     */
    public float[] getActionProbabilities(float[] logits) {
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = sigmoid(logits[i]);
        }
        return probs;
    }

    /**
     * Sample discrete actions from probabilities.
     */
    public boolean[] sampleActions(float[] probs) {
        boolean[] actions = new boolean[probs.length];
        for (int i = 0; i < probs.length; i++) {
            actions[i] = Math.random() < probs[i];
        }
        return actions;
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
            writeArray(out, w1);
            writeArray(out, b1);
            writeArray(out, w2);
            writeArray(out, b2);
            writeArray(out, wAction);
            writeArray(out, bAction);
            writeArray(out, wCamera);
            writeArray(out, bCamera);
        }
        MinimalAI.LOGGER.info("Model saved to {}", path);
    }

    public void load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(path.toFile())))) {
            w1 = readArray(in, w1.getShape());
            b1 = readArray(in, b1.getShape());
            w2 = readArray(in, w2.getShape());
            b2 = readArray(in, b2.getShape());
            wAction = readArray(in, wAction.getShape());
            bAction = readArray(in, bAction.getShape());
            wCamera = readArray(in, wCamera.getShape());
            bCamera = readArray(in, bCamera.getShape());
        }
        MinimalAI.LOGGER.info("Model loaded from {}", path);
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
}
