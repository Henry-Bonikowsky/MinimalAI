// AUTO-GENERATED from dimensions.json -- DO NOT EDIT DIRECTLY
package com.minimalai.ai;

/**
 * Constants matching the 35 multi-binary action space from training/combat_sim/env.py.
 *
 * Each action is an independent binary toggle (multi-binary, not one-hot).
 * The neural network outputs a probability for each; the executor reads
 * int[] actions where actions[i] is 0 or 1.
 */
public final class ActionSpace {

    // --- Movement (0-6) ---
    public static final int ACT_FORWARD        = 0;
    public static final int ACT_BACKWARD       = 1;
    public static final int ACT_STRAFE_LEFT    = 2;
    public static final int ACT_STRAFE_RIGHT   = 3;
    public static final int ACT_JUMP           = 4;
    public static final int ACT_SNEAK          = 5;
    public static final int ACT_SPRINT         = 6;

    // --- Combat (7-13) ---
    public static final int ACT_ATTACK         = 7;   // auto-derived from ENGAGE
    public static final int ACT_BLOCK          = 8;   // 1.8 sword block
    public static final int ACT_EAT_GAP        = 9;   // golden apple
    public static final int ACT_THROW_POT      = 10;   // splash health pot
    public static final int ACT_THROW_PEARL    = 11;   // ender pearl
    public static final int ACT_SPRINT_RESET   = 12;   // toggle sprint for KB reset
    public static final int ACT_SWAP_WEAPON    = 13;   // always masked

    // --- Sigil abilities (14-25, 12 slots) ---
    public static final int ACT_SIGIL_0        = 14;
    public static final int ACT_SIGIL_1        = 15;
    public static final int ACT_SIGIL_2        = 16;
    public static final int ACT_SIGIL_3        = 17;
    public static final int ACT_SIGIL_4        = 18;
    public static final int ACT_SIGIL_5        = 19;
    public static final int ACT_SIGIL_6        = 20;
    public static final int ACT_SIGIL_7        = 21;
    public static final int ACT_SIGIL_8        = 22;
    public static final int ACT_SIGIL_9        = 23;
    public static final int ACT_SIGIL_10       = 24;
    public static final int ACT_SIGIL_11       = 25;

    // --- Camera intent (26-29) ---
    // Priority: ENGAGE/FACE_TARGET > FACE_AWAY > LOOK_DOWN_SELF > FACE_MOVEMENT
    public static final int ACT_ENGAGE         = 26;  // face target + auto-attack when in reach
    public static final int ACT_FACE_AWAY      = 27;  // look opposite of target (kiting)
    public static final int ACT_LOOK_DOWN_SELF = 28;  // look at feet (self-pot)
    public static final int ACT_FACE_MOVEMENT  = 29;  // face movement direction
    public static final int ACT_FACE_TARGET    = 26;  // alias for ENGAGE (backward compat)

    // --- Target selection (30-34, up to 5 nearby entities) ---
    public static final int ACT_TARGET_0       = 30;
    public static final int ACT_TARGET_1       = 31;
    public static final int ACT_TARGET_2       = 32;
    public static final int ACT_TARGET_3       = 33;
    public static final int ACT_TARGET_4       = 34;

    public static final int NUM_ACTIONS = 35;

    /** Number of sigil ability slots. */
    public static final int NUM_SIGIL_SLOTS = 12;

    /** Number of target selection slots. */
    public static final int NUM_TARGET_SLOTS = 5;

    /** Smoothing factor for look transitions (1.0 = instant snap). */
    public static final float LOOK_SNAP_FACTOR = 1.0f;

    private ActionSpace() {}
}
