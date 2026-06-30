// AUTO-GENERATED from dimensions.json -- DO NOT EDIT DIRECTLY
package com.minimalai.ai;

/**
 * Constants defining observation tensor dimensions.
 * All values must match training/combat_sim/env.py exactly.
 */
public final class ObservationSpace {
    public static final int MAX_ENTITIES = 8;
    public static final int SELF_STATE_DIM = 38;
    public static final int ENTITY_FEATURE_DIM = 24;
    public static final int COMBAT_CTX_DIM = 26;
    public static final int NUM_SIGIL_SLOTS = 12;
    public static final int SIGIL_STATE_DIM = NUM_SIGIL_SLOTS * 4; // 48
    public static final int ENV_STATE_DIM = 8;
    public static final int HIDDEN_DIM = 128;

    private ObservationSpace() {}
}
