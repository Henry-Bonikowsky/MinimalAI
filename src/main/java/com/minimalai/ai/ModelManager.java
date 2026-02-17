package com.minimalai.ai;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;

import com.minimalai.util.TensorUtil;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Loads TorchScript models exported by export.py, runs inference through
 * the InferenceWrapper (7 tensor inputs -> action_probs, value, new_hidden),
 * and supports hot-reload for swapping models at runtime.
 *
 * <p>Each model is keyed by name (e.g. "master", "novice"). Predictors are
 * NOT thread-safe, so callers should create one per bot via
 * {@link #createPredictor(String)}.
 */
public class ModelManager implements AutoCloseable {

    private static final int NUM_ACTIONS = 35;
    private static final int GRU_HIDDEN_DIM = 128;

    private final Path modelsDir;
    private final ConcurrentHashMap<String, ZooModel<NDList, NDList>> loadedModels = new ConcurrentHashMap<>();
    private final Logger logger;

    /**
     * @param modelsDir directory containing .pt TorchScript files (e.g. plugins/MinimalAI/models/)
     * @param logger    plugin logger
     */
    public ModelManager(Path modelsDir, Logger logger) {
        this.modelsDir = modelsDir;
        this.logger = logger;
    }

    /**
     * Load a TorchScript model by name. The file must exist at
     * {@code modelsDir/name.pt}.
     */
    public void loadModel(String name) throws ModelNotFoundException, MalformedModelException, IOException {
        Path modelPath = modelsDir.resolve(name + ".pt");
        if (!Files.exists(modelPath)) {
            throw new IOException("Model file not found: " + modelPath);
        }

        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(modelPath)
                .optEngine("PyTorch")
                .optTranslator(new NoopTranslator())
                .build();

        ZooModel<NDList, NDList> model = criteria.loadModel();

        ZooModel<NDList, NDList> old = loadedModels.put(name, model);
        if (old != null) {
            old.close();
        }

        logger.info("Loaded model: " + name + " from " + modelPath);
    }

    /**
     * Create a new predictor for the named model. One predictor per bot --
     * Predictor is NOT thread-safe.
     */
    public Predictor<NDList, NDList> createPredictor(String modelName) {
        ZooModel<NDList, NDList> model = loadedModels.get(modelName);
        if (model == null) {
            throw new IllegalStateException("Model not loaded: " + modelName);
        }
        return model.newPredictor();
    }

    /**
     * Run a single inference step matching the InferenceWrapper signature:
     *
     * <pre>
     *   forward(self_state, entity_features, entity_mask,
     *           combat_ctx, sigil_state, env_state, hidden)
     * </pre>
     *
     * @param predictor      a predictor obtained from {@link #createPredictor}
     * @param manager        NDManager to allocate tensors (caller owns lifecycle)
     * @param selfState      (38,)  player state
     * @param entityFeatures (8*24) flat -- reshaped to (1, 8, 24)
     * @param entityMask     (8,)   1.0 for valid entities, 0.0 for padding
     * @param combatCtx      (26,)  combat context features
     * @param sigilState     (48,)  12 sigil slots * 4 dims
     * @param envState       (8,)   environment features
     * @param hidden         (128,) GRU hidden state (flat)
     * @return InferenceResult with actionProbs[35], value, newHidden[128]
     */
    public InferenceResult infer(Predictor<NDList, NDList> predictor,
                                 NDManager manager,
                                 float[] selfState,
                                 float[] entityFeatures,
                                 float[] entityMask,
                                 float[] combatCtx,
                                 float[] sigilState,
                                 float[] envState,
                                 float[] hidden) throws Exception {

        NDList input = new NDList(
                TensorUtil.toNDArray(manager, selfState, 1, 38),
                TensorUtil.toNDArray(manager, entityFeatures, 1, 8, 24),
                TensorUtil.toNDArray(manager, entityMask, 1, 8),
                TensorUtil.toNDArray(manager, combatCtx, 1, 26),
                TensorUtil.toNDArray(manager, sigilState, 1, 48),
                TensorUtil.toNDArray(manager, envState, 1, 8),
                TensorUtil.toNDArray(manager, hidden, 1, 1, GRU_HIDDEN_DIM)
        );

        NDList output = predictor.predict(input);

        // InferenceWrapper returns: action_probs (1,35), value (1,1), new_hidden (1,1,128)
        float[] actionProbs = TensorUtil.toFloatArray(output.get(0));
        float value = output.get(1).toFloatArray()[0];
        float[] newHidden = TensorUtil.toFloatArray(output.get(2));

        return new InferenceResult(actionProbs, value, newHidden);
    }

    /**
     * Hot-reload a model: close the old version and load the new one.
     */
    public void hotReload(String name) {
        try {
            ZooModel<NDList, NDList> old = loadedModels.remove(name);
            if (old != null) {
                old.close();
            }
            loadModel(name);
            logger.info("Hot-reloaded model: " + name);
        } catch (Exception e) {
            logger.severe("Failed to hot-reload model '" + name + "': " + e.getMessage());
        }
    }

    /**
     * List the names (without .pt extension) of all TorchScript models
     * in the models directory.
     */
    public List<String> listModels() {
        if (!Files.isDirectory(modelsDir)) {
            return Collections.emptyList();
        }
        try (Stream<Path> paths = Files.list(modelsDir)) {
            return paths
                    .filter(p -> p.toString().endsWith(".pt"))
                    .map(p -> {
                        String fn = p.getFileName().toString();
                        return fn.substring(0, fn.length() - 3);
                    })
                    .sorted()
                    .collect(Collectors.toList());
        } catch (IOException e) {
            logger.warning("Failed to list models in " + modelsDir + ": " + e.getMessage());
            return Collections.emptyList();
        }
    }

    @Override
    public void close() {
        loadedModels.values().forEach(ZooModel::close);
        loadedModels.clear();
        logger.info("ModelManager closed, all models released");
    }

    /**
     * Result of a single inference step.
     */
    public static class InferenceResult {
        public final float[] actionProbs;  // length 35
        public final float value;
        public final float[] newHidden;    // length 128

        public InferenceResult(float[] actionProbs, float value, float[] newHidden) {
            this.actionProbs = actionProbs;
            this.value = value;
            this.newHidden = newHidden;
        }
    }
}
