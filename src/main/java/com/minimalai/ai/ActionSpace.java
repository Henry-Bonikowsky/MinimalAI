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
    public static final int ACT_FORWARD      = 0;
    public static final int ACT_BACKWARD     = 1;
    public static final int ACT_STRAFE_LEFT  = 2;
    public static final int ACT_STRAFE_RIGHT = 3;
    public static final int ACT_JUMP         = 4;
    public static final int ACT_SNEAK        = 5;
    public static final int ACT_SPRINT       = 6;

    // --- Combat (7-13) ---
    public static final int ACT_ATTACK       = 7;
    public static final int ACT_BLOCK        = 8;   // 1.8 sword block
    public static final int ACT_EAT_GAP      = 9;   // golden apple
    public static final int ACT_THROW_POT    = 10;  // splash health pot
    public static final int ACT_THROW_PEARL  = 11;  // ender pearl
    public static final int ACT_SPRINT_RESET = 12;  // toggle sprint for KB reset
    public static final int ACT_SWAP_WEAPON  = 13;  // always masked

    // --- Sigil abilities (14-25, 12 slots) ---
    public static final int ACT_SIGIL_0  = 14;
    public static final int ACT_SIGIL_1  = 15;
    public static final int ACT_SIGIL_2  = 16;
    public static final int ACT_SIGIL_3  = 17;
    public static final int ACT_SIGIL_4  = 18;
    public static final int ACT_SIGIL_5  = 19;
    public static final int ACT_SIGIL_6  = 20;
    public static final int ACT_SIGIL_7  = 21;
    public static final int ACT_SIGIL_8  = 22;
    public static final int ACT_SIGIL_9  = 23;
    public static final int ACT_SIGIL_10 = 24;
    public static final int ACT_SIGIL_11 = 25;

    // --- Camera (26-29) ---
    public static final int ACT_LOOK_LEFT  = 26;
    public static final int ACT_LOOK_RIGHT = 27;
    public static final int ACT_LOOK_UP    = 28;
    public static final int ACT_LOOK_DOWN  = 29;

    // --- Target selection (30-34, up to 5 nearby entities) ---
    public static final int ACT_TARGET_0 = 30;
    public static final int ACT_TARGET_1 = 31;
    public static final int ACT_TARGET_2 = 32;
    public static final int ACT_TARGET_3 = 33;
    public static final int ACT_TARGET_4 = 34;

    public static final int NUM_ACTIONS = 35;

    /** Number of sigil ability slots. */
    public static final int NUM_SIGIL_SLOTS = 12;

    /** Number of target selection slots. */
    public static final int NUM_TARGET_SLOTS = 5;

    /** Camera turn step in degrees per tick when a look action fires. */
    public static final float CAMERA_STEP_DEGREES = 10f;

    private ActionSpace() {}
}
