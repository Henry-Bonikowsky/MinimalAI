package com.minimalai.recording;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.GameOptions;

/**
 * Captures player input actions each frame.
 *
 * Discrete actions (20 boolean values):
 * [0-3]   Movement: forward, back, left, right
 * [4-6]   Jump, sneak, sprint
 * [7-8]   Attack, use
 * [9-17]  Hotbar slots 1-9
 * [18-19] Drop, swap hands
 *
 * Continuous actions (2 float values):
 * [0] Camera yaw delta (degrees)
 * [1] Camera pitch delta (degrees)
 */
public class ActionRecorder {
    public static final int DISCRETE_ACTION_COUNT = 20;
    public static final int CONTINUOUS_ACTION_COUNT = 2;

    // Raw degrees - no normalization
    public static final float CAMERA_NORMALIZE_SCALE = 1.0f;

    private final MinecraftClient client;

    // Track previous rotation for delta calculation
    private float lastYaw = 0;
    private float lastPitch = 0;
    private boolean initialized = false;

    // Track hotbar selection changes
    private int lastSelectedSlot = -1;

    public ActionRecorder() {
        this.client = MinecraftClient.getInstance();
    }

    public void reset() {
        initialized = false;
        lastSelectedSlot = -1;
    }

    /**
     * Capture current discrete actions (keys pressed).
     * @return Array of 20 booleans representing key states
     */
    public boolean[] captureDiscreteActions() {
        boolean[] actions = new boolean[DISCRETE_ACTION_COUNT];

        ClientPlayerEntity player = client.player;
        GameOptions options = client.options;

        if (player == null || options == null) {
            return actions;
        }

        int idx = 0;

        // Movement (4)
        actions[idx++] = options.forwardKey.isPressed();
        actions[idx++] = options.backKey.isPressed();
        actions[idx++] = options.leftKey.isPressed();
        actions[idx++] = options.rightKey.isPressed();

        // Jump, sneak, sprint (3)
        actions[idx++] = options.jumpKey.isPressed();
        actions[idx++] = options.sneakKey.isPressed();
        actions[idx++] = options.sprintKey.isPressed();

        // Attack, use (2)
        actions[idx++] = options.attackKey.isPressed();
        actions[idx++] = options.useKey.isPressed();

        // Hotbar slots 1-9 (9) - detect slot changes
        int currentSlot = player.getInventory().getSelectedSlot();
        for (int i = 0; i < 9; i++) {
            // Mark as pressed if this slot was just selected
            actions[idx++] = (currentSlot == i && currentSlot != lastSelectedSlot && lastSelectedSlot >= 0);
        }
        lastSelectedSlot = currentSlot;

        // Drop, swap hands (2)
        actions[idx++] = options.dropKey.isPressed();
        actions[idx++] = options.swapHandsKey.isPressed();

        return actions;
    }

    /**
     * Capture continuous actions (camera movement).
     * Call this every frame to capture smooth mouse movement.
     * @return Array of 2 floats: [yaw_delta, pitch_delta]
     */
    public float[] captureContinuousActions() {
        float[] actions = new float[CONTINUOUS_ACTION_COUNT];

        ClientPlayerEntity player = client.player;
        if (player == null) {
            return actions;
        }

        float currentYaw = player.getYaw();
        float currentPitch = player.getPitch();

        if (!initialized) {
            lastYaw = currentYaw;
            lastPitch = currentPitch;
            initialized = true;
            return actions; // Return zeros for first frame
        }

        // Calculate delta (handle wraparound for yaw)
        float yawDelta = currentYaw - lastYaw;

        // Handle yaw wraparound (-180 to 180)
        if (yawDelta > 180) yawDelta -= 360;
        if (yawDelta < -180) yawDelta += 360;

        float pitchDelta = currentPitch - lastPitch;

        // Normalize to roughly [-1, 1] range for better learning
        actions[0] = yawDelta / CAMERA_NORMALIZE_SCALE;
        actions[1] = pitchDelta / CAMERA_NORMALIZE_SCALE;

        lastYaw = currentYaw;
        lastPitch = currentPitch;

        return actions;
    }

    /**
     * Get names of discrete actions for debugging/display.
     */
    public static String[] getDiscreteActionNames() {
        return new String[] {
            "forward", "back", "left", "right",
            "jump", "sneak", "sprint",
            "attack", "use",
            "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4", "hotbar_5",
            "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9",
            "drop", "swap_hands"
        };
    }

    /**
     * Get names of continuous actions for debugging/display.
     */
    public static String[] getContinuousActionNames() {
        return new String[] { "yaw_delta", "pitch_delta" };
    }
}
