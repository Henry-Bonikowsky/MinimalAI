package com.minimalai.playback;

import com.minimalai.recording.ActionRecorder;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.client.option.KeyBinding;

/**
 * Executes recorded actions on the player.
 * Handles both discrete actions (key presses) and continuous actions (camera).
 */
public class ActionExecutor {
    private final MinecraftClient client;

    public ActionExecutor() {
        this.client = MinecraftClient.getInstance();
    }

    /**
     * Apply discrete actions (key states).
     * Actions array layout:
     * [0-3]   Movement: forward, back, left, right
     * [4-6]   Jump, sneak, sprint
     * [7-8]   Attack, use
     * [9-17]  Hotbar slots 1-9
     * [18-19] Drop, swap hands
     */
    public void applyDiscreteActions(boolean[] actions) {
        if (client.player == null || client.options == null) return;

        // Movement
        setKeyState(client.options.forwardKey, actions[0]);
        setKeyState(client.options.backKey, actions[1]);
        setKeyState(client.options.leftKey, actions[2]);
        setKeyState(client.options.rightKey, actions[3]);

        // Jump, sneak, sprint
        setKeyState(client.options.jumpKey, actions[4]);
        setKeyState(client.options.sneakKey, actions[5]);
        setKeyState(client.options.sprintKey, actions[6]);

        // Attack, use
        setKeyState(client.options.attackKey, actions[7]);
        setKeyState(client.options.useKey, actions[8]);

        // Hotbar slots - select if pressed
        for (int i = 0; i < 9; i++) {
            if (actions[9 + i]) {
                client.player.getInventory().setSelectedSlot(i);
            }
        }

        // Drop, swap hands
        setKeyState(client.options.dropKey, actions[18]);
        setKeyState(client.options.swapHandsKey, actions[19]);
    }

    /**
     * Apply continuous actions (camera movement).
     * Values are normalized - will be denormalized back to degrees.
     * @param yawDelta Normalized yaw delta
     * @param pitchDelta Normalized pitch delta
     */
    public void applyCameraMovement(float yawDelta, float pitchDelta) {
        ClientPlayerEntity player = client.player;
        if (player == null) return;

        // Denormalize from [-1, 1] back to degrees
        float yawDeg = yawDelta * ActionRecorder.CAMERA_NORMALIZE_SCALE;
        float pitchDeg = pitchDelta * ActionRecorder.CAMERA_NORMALIZE_SCALE;

        float newYaw = player.getYaw() + yawDeg;
        float newPitch = player.getPitch() + pitchDeg;

        // Clamp pitch to valid range
        newPitch = Math.max(-90.0f, Math.min(90.0f, newPitch));

        player.setYaw(newYaw);
        player.setPitch(newPitch);
    }

    /**
     * Release all keys - call when stopping playback.
     */
    public void releaseAllKeys() {
        if (client.options == null) return;

        setKeyState(client.options.forwardKey, false);
        setKeyState(client.options.backKey, false);
        setKeyState(client.options.leftKey, false);
        setKeyState(client.options.rightKey, false);
        setKeyState(client.options.jumpKey, false);
        setKeyState(client.options.sneakKey, false);
        setKeyState(client.options.sprintKey, false);
        setKeyState(client.options.attackKey, false);
        setKeyState(client.options.useKey, false);
        setKeyState(client.options.dropKey, false);
        setKeyState(client.options.swapHandsKey, false);
    }

    private void setKeyState(KeyBinding key, boolean pressed) {
        key.setPressed(pressed);
    }
}
