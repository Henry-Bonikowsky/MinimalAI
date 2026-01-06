package com.minimalai.ui;

import com.minimalai.ai.AIController;
import com.minimalai.ai.Trainer;
import com.minimalai.playback.PlaybackController;
import com.minimalai.recording.RecordingManager;
import net.fabricmc.fabric.api.client.rendering.v1.HudRenderCallback;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.font.TextRenderer;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.render.RenderTickCounter;

/**
 * Renders status overlay in the corner of the screen.
 * Shows recording, playback, and training status.
 */
public class OverlayRenderer {
    private static OverlayRenderer instance;

    private static final int PADDING = 5;
    private static final int LINE_HEIGHT = 12;

    // Colors (ARGB)
    private static final int RED = 0xFFFF4444;
    private static final int GREEN = 0xFF44FF44;
    private static final int YELLOW = 0xFFFFFF44;
    private static final int CYAN = 0xFF44FFFF;
    private static final int WHITE = 0xFFFFFFFF;
    private static final int GRAY = 0xFFAAAAAA;
    private static final int BG_COLOR = 0x88000000;

    private boolean enabled = true;

    public static OverlayRenderer getInstance() {
        if (instance == null) {
            instance = new OverlayRenderer();
        }
        return instance;
    }

    private OverlayRenderer() {
        HudRenderCallback.EVENT.register(this::render);
    }

    public static void init() {
        getInstance();
    }

    private void render(DrawContext context, RenderTickCounter tickCounter) {
        if (!enabled) return;

        MinecraftClient client = MinecraftClient.getInstance();
        if (client.player == null || client.options.hudHidden) return;

        // Check what's active
        boolean isRecording = RecordingManager.getInstance().isRecording();
        boolean isPlaying = PlaybackController.getInstance().isPlaying();
        boolean isTraining = Trainer.getInstance().isTraining();
        boolean isAI = AIController.getInstance().isRunning();

        if (!isRecording && !isPlaying && !isTraining && !isAI) return;

        TextRenderer textRenderer = client.textRenderer;
        int x = PADDING;
        int y = PADDING;

        // Calculate box size
        int lines = 0;
        int maxWidth = 0;

        if (isRecording) {
            lines += 2;
            maxWidth = Math.max(maxWidth, textRenderer.getWidth("● REC  00:00"));
            maxWidth = Math.max(maxWidth, textRenderer.getWidth("Frames: 00000"));
        }
        if (isPlaying) {
            lines += 2;
            maxWidth = Math.max(maxWidth, textRenderer.getWidth("▶ PLAY  100%"));
            maxWidth = Math.max(maxWidth, textRenderer.getWidth("Frame: 0000/0000"));
        }
        if (isTraining) {
            lines += 3;
            maxWidth = Math.max(maxWidth, textRenderer.getWidth("⚡ TRAIN"));
            maxWidth = Math.max(maxWidth, textRenderer.getWidth("Epoch: 00/00"));
            maxWidth = Math.max(maxWidth, textRenderer.getWidth("Acc: 100.0%"));
        }
        if (isAI) {
            lines += 2;
            maxWidth = Math.max(maxWidth, textRenderer.getWidth("🤖 AI RUNNING"));
            maxWidth = Math.max(maxWidth, textRenderer.getWidth("Frames: 00000"));
        }

        // Draw background
        int boxWidth = maxWidth + PADDING * 2;
        int boxHeight = lines * LINE_HEIGHT + PADDING * 2;
        context.fill(x, y, x + boxWidth, y + boxHeight, BG_COLOR);

        // Draw content
        int textX = x + PADDING;
        int textY = y + PADDING;

        if (isRecording) {
            RecordingManager rm = RecordingManager.getInstance();
            var recording = rm.getCurrentRecording();

            // Blinking red dot
            long time = System.currentTimeMillis();
            String dot = (time % 1000 < 500) ? "●" : "○";

            String duration = "00:00";
            int frames = 0;
            if (recording != null) {
                frames = recording.getFrameCount();
                long ms = recording.getFrameCount() > 0 ?
                    recording.getFrames().get(recording.getFrameCount() - 1).timestamp : 0;
                duration = formatDuration(ms);
            }

            context.drawText(textRenderer, dot + " REC  " + duration, textX, textY, RED, true);
            textY += LINE_HEIGHT;
            context.drawText(textRenderer, "Frames: " + frames, textX, textY, GRAY, true);
            textY += LINE_HEIGHT;
        }

        if (isPlaying) {
            PlaybackController pc = PlaybackController.getInstance();
            int current = pc.getCurrentFrameIndex();
            int total = pc.getTotalFrames();
            int percent = total > 0 ? (current * 100 / total) : 0;

            String icon = pc.isPaused() ? "⏸" : "▶";
            String speed = String.format("%.1fx", pc.getPlaybackSpeed());

            context.drawText(textRenderer, icon + " PLAY  " + percent + "%  " + speed, textX, textY, GREEN, true);
            textY += LINE_HEIGHT;
            context.drawText(textRenderer, "Frame: " + current + "/" + total, textX, textY, GRAY, true);
            textY += LINE_HEIGHT;
        }

        if (isTraining) {
            Trainer trainer = Trainer.getInstance();
            int epoch = trainer.getCurrentEpoch() + 1;
            int total = trainer.getTotalEpochs();
            float acc = trainer.getLastActionAccuracy();

            context.drawText(textRenderer, "⚡ TRAIN", textX, textY, YELLOW, true);
            textY += LINE_HEIGHT;
            context.drawText(textRenderer, "Epoch: " + epoch + "/" + total, textX, textY, GRAY, true);
            textY += LINE_HEIGHT;
            context.drawText(textRenderer, "Acc: " + String.format("%.1f", acc) + "%", textX, textY, GRAY, true);
            textY += LINE_HEIGHT;
        }

        if (isAI) {
            AIController ai = AIController.getInstance();
            int frames = ai.getFrameCount();

            context.drawText(textRenderer, "🤖 AI RUNNING", textX, textY, CYAN, true);
            textY += LINE_HEIGHT;
            context.drawText(textRenderer, "Frames: " + frames, textX, textY, GRAY, true);
            textY += LINE_HEIGHT;
        }
    }

    private String formatDuration(long ms) {
        long seconds = ms / 1000;
        long minutes = seconds / 60;
        seconds = seconds % 60;
        return String.format("%02d:%02d", minutes, seconds);
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public boolean isEnabled() {
        return enabled;
    }
}
