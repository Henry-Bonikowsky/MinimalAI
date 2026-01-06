package com.minimalai;

import com.minimalai.ai.AIController;
import com.minimalai.ai.RLController;
import com.minimalai.ai.Trainer;
import com.minimalai.playback.PlaybackController;
import com.minimalai.recording.RecordingManager;
import com.minimalai.ui.DashboardScreen;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;

public class ModKeybinds {
    private static KeyBinding dashboardKey;
    private static KeyBinding recordKey;
    private static KeyBinding playbackKey;
    private static KeyBinding trainKey;
    private static KeyBinding aiKey;
    private static KeyBinding rlKey;
    private static KeyBinding stopKey;

    public static void register() {
        // Y - Open dashboard
        dashboardKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.minimalai.dashboard",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_Y,
            "key.category.minimalai"
        ));

        // R - Toggle recording
        recordKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.minimalai.record",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_R,
            "key.category.minimalai"
        ));

        // P - Play latest recording
        playbackKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.minimalai.playback",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_P,
            "key.category.minimalai"
        ));

        // T - Train on recordings
        trainKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.minimalai.train",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_T,
            "key.category.minimalai"
        ));

        // I - Run AI
        aiKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.minimalai.ai",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_I,
            "key.category.minimalai"
        ));

        // L - RL Training (Learn)
        rlKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.minimalai.rl",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_L,
            "key.category.minimalai"
        ));

        // End - Stop everything
        stopKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.minimalai.stop",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_END,
            "key.category.minimalai"
        ));

        ClientTickEvents.END_CLIENT_TICK.register(ModKeybinds::onTick);

        MinimalAI.LOGGER.info("Keybinds registered: Y=Dashboard, R=Record, P=Play, T=Train, I=AI, L=RL, End=Stop");
    }

    private static void onTick(MinecraftClient client) {
        if (client.player == null) return;

        while (dashboardKey.wasPressed()) {
            if (client.currentScreen == null) {
                client.setScreen(new DashboardScreen());
            } else if (client.currentScreen instanceof DashboardScreen) {
                client.setScreen(null);
            }
        }

        while (recordKey.wasPressed()) {
            MinimalAI.LOGGER.info("Record key pressed (R)");
            // Stop playback if playing
            if (PlaybackController.getInstance().isPlaying()) {
                PlaybackController.getInstance().stop();
            }
            RecordingManager.getInstance().toggleRecording();
        }

        while (playbackKey.wasPressed()) {
            MinimalAI.LOGGER.info("Playback key pressed (P)");
            // Stop recording if recording
            if (RecordingManager.getInstance().isRecording()) {
                RecordingManager.getInstance().stopRecording();
            }
            // Toggle playback
            if (PlaybackController.getInstance().isPlaying()) {
                PlaybackController.getInstance().togglePause();
            } else {
                PlaybackController.getInstance().playLatest();
            }
        }

        while (trainKey.wasPressed()) {
            MinimalAI.LOGGER.info("Train key pressed (T)");
            if (Trainer.getInstance().isTraining()) {
                Trainer.getInstance().stopTraining();
            } else {
                // Stop recording/playback/AI before training
                if (RecordingManager.getInstance().isRecording()) {
                    RecordingManager.getInstance().stopRecording();
                }
                if (PlaybackController.getInstance().isPlaying()) {
                    PlaybackController.getInstance().stop();
                }
                if (AIController.getInstance().isRunning()) {
                    AIController.getInstance().stop();
                }
                Trainer.getInstance().startTraining();
            }
        }

        while (aiKey.wasPressed()) {
            MinimalAI.LOGGER.info("AI key pressed (I)");
            if (AIController.getInstance().isRunning()) {
                AIController.getInstance().stop();
            } else {
                // Stop recording/playback/RL before AI
                if (RecordingManager.getInstance().isRecording()) {
                    RecordingManager.getInstance().stopRecording();
                }
                if (PlaybackController.getInstance().isPlaying()) {
                    PlaybackController.getInstance().stop();
                }
                if (RLController.getInstance().isRunning()) {
                    RLController.getInstance().stop();
                }
                AIController.getInstance().start();
            }
        }

        while (rlKey.wasPressed()) {
            MinimalAI.LOGGER.info("RL key pressed (L)");
            if (RLController.getInstance().isRunning()) {
                RLController.getInstance().stop();
            } else {
                // Stop recording/playback/AI before RL
                if (RecordingManager.getInstance().isRecording()) {
                    RecordingManager.getInstance().stopRecording();
                }
                if (PlaybackController.getInstance().isPlaying()) {
                    PlaybackController.getInstance().stop();
                }
                if (AIController.getInstance().isRunning()) {
                    AIController.getInstance().stop();
                }
                if (Trainer.getInstance().isTraining()) {
                    Trainer.getInstance().stopTraining();
                }
                RLController.getInstance().start();
            }
        }

        while (stopKey.wasPressed()) {
            MinimalAI.LOGGER.info("Stop key pressed (End)");
            if (RecordingManager.getInstance().isRecording()) {
                RecordingManager.getInstance().stopRecording();
            }
            if (PlaybackController.getInstance().isPlaying()) {
                PlaybackController.getInstance().stop();
            }
            if (Trainer.getInstance().isTraining()) {
                Trainer.getInstance().stopTraining();
            }
            if (AIController.getInstance().isRunning()) {
                AIController.getInstance().stop();
            }
            if (RLController.getInstance().isRunning()) {
                RLController.getInstance().stop();
            }
        }
    }
}
