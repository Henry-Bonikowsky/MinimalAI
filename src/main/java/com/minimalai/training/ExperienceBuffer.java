package com.minimalai.training;

import com.minimalai.ai.ActionSpace;
import com.minimalai.ai.ObservationSpace;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * Ring buffer storing PPO experience tuples for TCP transmission to the Python trainer.
 *
 * <p>Each experience contains the full structured observation, multi-binary action vector,
 * per-action log-probabilities, value estimate, reward, done flag, and LSTM hidden state.
 *
 * <p>Dimension constants are sourced from {@link ObservationSpace} and {@link ActionSpace}.
 */
public class ExperienceBuffer {

    private static final Logger LOG = Logger.getLogger("MinimalAI");

    // Observation dimensions (from ObservationSpace)
    private static final int SELF_STATE_DIM     = ObservationSpace.SELF_STATE_DIM;      // 38
    private static final int ENTITY_FEATURES    = ObservationSpace.MAX_ENTITIES
                                                  * ObservationSpace.ENTITY_FEATURE_DIM; // 8*24 = 192
    private static final int ENTITY_MASK_DIM    = ObservationSpace.MAX_ENTITIES;         // 8
    private static final int COMBAT_CTX_DIM     = ObservationSpace.COMBAT_CTX_DIM;       // 26
    private static final int SIGIL_STATE_DIM    = ObservationSpace.SIGIL_STATE_DIM;      // 48
    private static final int ENV_STATE_DIM      = ObservationSpace.ENV_STATE_DIM;        // 8

    // Action / hidden dimensions (from ActionSpace / ObservationSpace)
    private static final int NUM_ACTIONS        = ActionSpace.NUM_ACTIONS;               // 35
    private static final int HIDDEN_DIM         = ObservationSpace.HIDDEN_DIM;           // 128

    /**
     * A single experience tuple mirroring one env step.
     */
    public record Experience(
        float[] selfState,       // [38]
        float[] entityFeatures,  // [192]  (8*24 flattened)
        float[] entityMask,      // [8]
        float[] combatCtx,       // [26]
        float[] sigilState,      // [48]
        float[] envState,        // [8]
        int[]   actions,         // [35]
        float[] actionProbs,     // [35]   per-action log-probs
        float   value,
        float   reward,
        boolean done,
        float[] hidden           // [128]
    ) {}

    private final Experience[] ring;
    private final int capacity;
    private int size;
    private int writeIndex;

    /**
     * Create a buffer with the given capacity (default 2048).
     */
    public ExperienceBuffer(int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("capacity must be > 0");
        this.capacity = capacity;
        this.ring = new Experience[capacity];
        this.size = 0;
        this.writeIndex = 0;
    }

    public ExperienceBuffer() {
        this(2048);
    }

    // ----------------------------------------------------------------
    //  Core API
    // ----------------------------------------------------------------

    /**
     * Add an experience to the ring buffer (overwrites oldest if full).
     */
    public void add(Experience exp) {
        ring[writeIndex] = exp;
        writeIndex = (writeIndex + 1) % capacity;
        if (size < capacity) size++;
    }

    /**
     * Whether the buffer has reached capacity at least once.
     */
    public boolean isFull() {
        return size >= capacity;
    }

    public int size() {
        return size;
    }

    public int capacity() {
        return capacity;
    }

    /**
     * Drain all stored experiences (oldest-first) and clear the buffer.
     */
    public List<Experience> drain() {
        List<Experience> out = new ArrayList<>(size);
        if (size < capacity) {
            // Haven't wrapped yet -- entries are [0 .. writeIndex)
            for (int i = 0; i < writeIndex; i++) {
                out.add(ring[i]);
                ring[i] = null;
            }
        } else {
            // Wrapped -- oldest is at writeIndex
            for (int i = 0; i < capacity; i++) {
                int idx = (writeIndex + i) % capacity;
                out.add(ring[idx]);
                ring[idx] = null;
            }
        }
        size = 0;
        writeIndex = 0;
        return out;
    }

    /**
     * Clear without returning data.
     */
    public void clear() {
        Arrays.fill(ring, null);
        size = 0;
        writeIndex = 0;
    }

    // ----------------------------------------------------------------
    //  Serialization for TCP transmission
    // ----------------------------------------------------------------

    /**
     * Serialize all buffered experiences to a compact binary format for TCP.
     *
     * <p>Wire format:
     * <pre>
     *   int32:  numExperiences
     *   -- per experience --
     *   float32[38]:  selfState
     *   float32[192]: entityFeatures
     *   float32[8]:   entityMask
     *   float32[26]:  combatCtx
     *   float32[48]:  sigilState
     *   float32[8]:   envState
     *   int32[35]:    actions
     *   float32[35]:  actionProbs
     *   float32:      value
     *   float32:      reward
     *   byte:         done (0 or 1)
     *   float32[128]: hidden
     * </pre>
     */
    public byte[] toSerializable() {
        List<Experience> exps = drain();
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream(exps.size() * 2200);
             DataOutputStream dos = new DataOutputStream(baos)) {

            dos.writeInt(exps.size());

            for (Experience exp : exps) {
                writeFloats(dos, exp.selfState,      SELF_STATE_DIM);
                writeFloats(dos, exp.entityFeatures,  ENTITY_FEATURES);
                writeFloats(dos, exp.entityMask,      ENTITY_MASK_DIM);
                writeFloats(dos, exp.combatCtx,       COMBAT_CTX_DIM);
                writeFloats(dos, exp.sigilState,      SIGIL_STATE_DIM);
                writeFloats(dos, exp.envState,        ENV_STATE_DIM);
                writeInts(dos,   exp.actions,         NUM_ACTIONS);
                writeFloats(dos, exp.actionProbs,     NUM_ACTIONS);
                dos.writeFloat(exp.value);
                dos.writeFloat(exp.reward);
                dos.writeByte(exp.done ? 1 : 0);
                writeFloats(dos, exp.hidden,          HIDDEN_DIM);
            }

            dos.flush();
            return baos.toByteArray();

        } catch (IOException e) {
            LOG.severe("[ExperienceBuffer] Serialization failed: " + e.getMessage());
            return new byte[0];
        }
    }

    // ----------------------------------------------------------------
    //  Helpers
    // ----------------------------------------------------------------

    private static void writeFloats(DataOutputStream dos, float[] arr, int expected) throws IOException {
        for (int i = 0; i < expected; i++) {
            dos.writeFloat(i < arr.length ? arr[i] : 0.0f);
        }
    }

    private static void writeInts(DataOutputStream dos, int[] arr, int expected) throws IOException {
        for (int i = 0; i < expected; i++) {
            dos.writeInt(i < arr.length ? arr[i] : 0);
        }
    }
}
