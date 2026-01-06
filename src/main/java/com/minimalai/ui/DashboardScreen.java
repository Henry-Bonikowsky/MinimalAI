package com.minimalai.ui;

import com.minimalai.ai.AIController;
import com.minimalai.ai.Trainer;
import com.minimalai.playback.PlaybackController;
import com.minimalai.recording.RecordingManager;
import com.minimalai.recording.RecordingStorage;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.widget.ButtonWidget;
import net.minecraft.text.Text;

import java.nio.file.Path;
import java.util.List;

/**
 * Main dashboard screen for MinimalAI.
 * Shows recording/training/playback controls and lists.
 */
public class DashboardScreen extends Screen {
    private static final int BUTTON_WIDTH = 100;
    private static final int BUTTON_HEIGHT = 20;
    private static final int PADDING = 10;

    // Colors
    private static final int RED = 0xFFFF4444;
    private static final int GREEN = 0xFF44FF44;
    private static final int YELLOW = 0xFFFFFF44;
    private static final int WHITE = 0xFFFFFFFF;
    private static final int GRAY = 0xFFAAAAAA;
    private static final int DARK_GRAY = 0xFF333333;

    private List<RecordingStorage.RecordingInfo> recordings;
    private List<Path> models;
    private int selectedRecordingIndex = -1;
    private int selectedModelIndex = -1;

    private ButtonWidget recordButton;
    private ButtonWidget playButton;
    private ButtonWidget trainButton;
    private ButtonWidget aiButton;
    private ButtonWidget deleteButton;
    private ButtonWidget deleteModelsButton;

    public DashboardScreen() {
        super(Text.literal("MinimalAI Dashboard"));
    }

    @Override
    protected void init() {
        super.init();

        // Refresh recordings and models lists
        recordings = RecordingManager.getInstance().getStorage().listRecordings();
        models = AIController.getInstance().listModels();

        int centerX = width / 2;
        int buttonY = 40;

        // Main action buttons at top
        recordButton = ButtonWidget.builder(Text.literal("Record"), button -> onRecord())
            .dimensions(centerX - 210, buttonY, BUTTON_WIDTH, BUTTON_HEIGHT)
            .build();
        addDrawableChild(recordButton);

        playButton = ButtonWidget.builder(Text.literal("Play"), button -> onPlay())
            .dimensions(centerX - 105, buttonY, BUTTON_WIDTH, BUTTON_HEIGHT)
            .build();
        addDrawableChild(playButton);

        trainButton = ButtonWidget.builder(Text.literal("Train"), button -> onTrain())
            .dimensions(centerX, buttonY, BUTTON_WIDTH, BUTTON_HEIGHT)
            .build();
        addDrawableChild(trainButton);

        aiButton = ButtonWidget.builder(Text.literal("Run AI"), button -> onAI())
            .dimensions(centerX + 105, buttonY, BUTTON_WIDTH, BUTTON_HEIGHT)
            .build();
        addDrawableChild(aiButton);

        // Delete button for selected recording
        deleteButton = ButtonWidget.builder(Text.literal("Delete Rec"), button -> onDelete())
            .dimensions(PADDING, height - BUTTON_HEIGHT - PADDING, BUTTON_WIDTH, BUTTON_HEIGHT)
            .build();
        deleteButton.active = false;
        addDrawableChild(deleteButton);

        // Delete models button
        deleteModelsButton = ButtonWidget.builder(Text.literal("Delete Models"), button -> onDeleteModels())
            .dimensions(width - BUTTON_WIDTH - PADDING, height - BUTTON_HEIGHT - PADDING, BUTTON_WIDTH, BUTTON_HEIGHT)
            .build();
        addDrawableChild(deleteModelsButton);

        // Close button
        addDrawableChild(ButtonWidget.builder(Text.literal("Close"), button -> close())
            .dimensions(centerX - 50, height - BUTTON_HEIGHT - PADDING, BUTTON_WIDTH, BUTTON_HEIGHT)
            .build());

        updateButtonStates();
    }

    @Override
    public void render(DrawContext context, int mouseX, int mouseY, float delta) {
        // Let super handle background (don't call renderBackground ourselves - causes blur crash in 1.21+)
        super.render(context, mouseX, mouseY, delta);

        // Title
        context.drawCenteredTextWithShadow(textRenderer, title, width / 2, 15, WHITE);

        // Status line
        String status = getStatusText();
        int statusColor = getStatusColor();
        context.drawCenteredTextWithShadow(textRenderer, status, width / 2, 65, statusColor);

        // Recordings list
        int listX = PADDING;
        int listY = 85;
        int listWidth = width / 2 - PADDING * 2;
        int listHeight = height - listY - 50;

        // List header
        context.drawTextWithShadow(textRenderer, "Recordings (" + recordings.size() + ")", listX, listY, WHITE);
        listY += 15;

        // List background
        context.fill(listX, listY, listX + listWidth, listY + listHeight, 0x88000000);

        // List items
        int itemY = listY + 2;
        int itemHeight = 24;
        for (int i = 0; i < recordings.size() && itemY + itemHeight < listY + listHeight; i++) {
            RecordingStorage.RecordingInfo info = recordings.get(i);

            boolean isHovered = mouseX >= listX && mouseX < listX + listWidth &&
                               mouseY >= itemY && mouseY < itemY + itemHeight;
            boolean isSelected = i == selectedRecordingIndex;

            // Item background
            if (isSelected) {
                context.fill(listX + 1, itemY, listX + listWidth - 1, itemY + itemHeight, 0xFF446688);
            } else if (isHovered) {
                context.fill(listX + 1, itemY, listX + listWidth - 1, itemY + itemHeight, 0xFF333344);
            }

            // Recording name
            context.drawTextWithShadow(textRenderer, info.name, listX + 5, itemY + 2, WHITE);

            // Recording details
            String details = String.format("%s | %d frames | %.0f FPS",
                info.getFormattedDuration(), info.frameCount, info.avgFps);
            context.drawTextWithShadow(textRenderer, details, listX + 5, itemY + 12, GRAY);

            itemY += itemHeight;
        }

        // Training stats (right side)
        int statsX = width / 2 + PADDING;
        int statsY = 85;

        context.drawTextWithShadow(textRenderer, "Training Stats", statsX, statsY, WHITE);
        statsY += 15;

        Trainer trainer = Trainer.getInstance();
        if (trainer.isTraining()) {
            context.drawTextWithShadow(textRenderer,
                String.format("Epoch: %d/%d", trainer.getCurrentEpoch() + 1, trainer.getTotalEpochs()),
                statsX, statsY, YELLOW);
            statsY += 12;
        }

        context.drawTextWithShadow(textRenderer,
            String.format("Action Loss: %.4f", trainer.getLastActionLoss()),
            statsX, statsY, GRAY);
        statsY += 12;

        context.drawTextWithShadow(textRenderer,
            String.format("Camera Loss: %.4f", trainer.getLastCameraLoss()),
            statsX, statsY, GRAY);
        statsY += 12;

        context.drawTextWithShadow(textRenderer,
            String.format("Accuracy: %.1f%%", trainer.getLastActionAccuracy()),
            statsX, statsY, GRAY);
        statsY += 20;

        // Models list
        int modelListY = statsY;
        int modelListWidth = width / 2 - PADDING * 2;
        int modelListHeight = height - modelListY - 50;
        int modelItemHeight = 16;

        context.drawTextWithShadow(textRenderer, "Models (" + models.size() + ")", statsX, modelListY, WHITE);
        modelListY += 15;

        // List background
        context.fill(statsX, modelListY, statsX + modelListWidth, modelListY + modelListHeight, 0x88000000);

        // List items
        int modelItemY = modelListY + 2;
        for (int i = 0; i < models.size() && modelItemY + modelItemHeight < modelListY + modelListHeight; i++) {
            Path model = models.get(i);
            String modelName = model.getFileName().toString();

            boolean isHovered = mouseX >= statsX && mouseX < statsX + modelListWidth &&
                               mouseY >= modelItemY && mouseY < modelItemY + modelItemHeight;
            boolean isSelected = i == selectedModelIndex;

            // Item background
            if (isSelected) {
                context.fill(statsX + 1, modelItemY, statsX + modelListWidth - 1, modelItemY + modelItemHeight, 0xFF446688);
            } else if (isHovered) {
                context.fill(statsX + 1, modelItemY, statsX + modelListWidth - 1, modelItemY + modelItemHeight, 0xFF333344);
            }

            // Model name (truncate if needed)
            String displayName = modelName.length() > 25 ? modelName.substring(0, 22) + "..." : modelName;
            context.drawTextWithShadow(textRenderer, displayName, statsX + 5, modelItemY + 4, WHITE);

            modelItemY += modelItemHeight;
        }

        // Keybind hints
        int hintY = height - 35;
        context.drawCenteredTextWithShadow(textRenderer,
            "R=Record  P=Play  T=Train  I=AI  End=Stop", width / 2, hintY, GRAY);
    }

    @Override
    public boolean mouseClicked(double mouseX, double mouseY, int button) {
        // Check if clicked on recording list (left side)
        int listX = PADDING;
        int listY = 100;
        int listWidth = width / 2 - PADDING * 2;
        int itemHeight = 24;

        if (mouseX >= listX && mouseX < listX + listWidth && mouseY >= listY) {
            int clickedIndex = (int) ((mouseY - listY) / itemHeight);
            if (clickedIndex >= 0 && clickedIndex < recordings.size()) {
                selectedRecordingIndex = clickedIndex;
                updateButtonStates();
                return true;
            }
        }

        // Check if clicked on model list (right side)
        int statsX = width / 2 + PADDING;
        int modelListY = 85 + 15 + 12 + 12 + 12 + 20 + 15; // After training stats
        int modelListWidth = width / 2 - PADDING * 2;
        int modelItemHeight = 16;

        if (mouseX >= statsX && mouseX < statsX + modelListWidth && mouseY >= modelListY) {
            int clickedIndex = (int) ((mouseY - modelListY - 2) / modelItemHeight);
            if (clickedIndex >= 0 && clickedIndex < models.size()) {
                selectedModelIndex = clickedIndex;
                updateButtonStates();
                return true;
            }
        }

        return super.mouseClicked(mouseX, mouseY, button);
    }

    private void updateButtonStates() {
        boolean isRecording = RecordingManager.getInstance().isRecording();
        boolean isPlaying = PlaybackController.getInstance().isPlaying();
        boolean isTraining = Trainer.getInstance().isTraining();
        boolean isAI = AIController.getInstance().isRunning();

        recordButton.setMessage(Text.literal(isRecording ? "Stop Rec" : "Record"));
        playButton.setMessage(Text.literal(isPlaying ? "Stop Play" : "Play"));
        trainButton.setMessage(Text.literal(isTraining ? "Stop Train" : "Train"));
        aiButton.setMessage(Text.literal(isAI ? "Stop AI" : "Run AI"));

        playButton.active = !recordings.isEmpty() || isPlaying;
        trainButton.active = !recordings.isEmpty() || isTraining;
        aiButton.active = !models.isEmpty() || isAI;
        deleteButton.active = selectedRecordingIndex >= 0 && !isRecording && !isPlaying;
        deleteModelsButton.active = !models.isEmpty() && !isAI;
    }

    private String getStatusText() {
        if (RecordingManager.getInstance().isRecording()) {
            return "● Recording...";
        }
        if (PlaybackController.getInstance().isPlaying()) {
            return "▶ Playing...";
        }
        if (Trainer.getInstance().isTraining()) {
            return "⚡ Training...";
        }
        if (AIController.getInstance().isRunning()) {
            return "🤖 AI Running...";
        }
        return "Ready";
    }

    private int getStatusColor() {
        if (RecordingManager.getInstance().isRecording()) return RED;
        if (PlaybackController.getInstance().isPlaying()) return GREEN;
        if (Trainer.getInstance().isTraining()) return YELLOW;
        if (AIController.getInstance().isRunning()) return 0xFF44FFFF; // CYAN
        return WHITE;
    }

    private void onRecord() {
        RecordingManager.getInstance().toggleRecording();
        if (!RecordingManager.getInstance().isRecording()) {
            // Refresh list after stopping
            recordings = RecordingManager.getInstance().getStorage().listRecordings();
        }
        updateButtonStates();
    }

    private void onPlay() {
        if (PlaybackController.getInstance().isPlaying()) {
            PlaybackController.getInstance().stop();
        } else if (selectedRecordingIndex >= 0 && selectedRecordingIndex < recordings.size()) {
            close();
            PlaybackController.getInstance().play(recordings.get(selectedRecordingIndex).name);
        } else if (!recordings.isEmpty()) {
            close();
            PlaybackController.getInstance().playLatest();
        }
        updateButtonStates();
    }

    private void onTrain() {
        if (Trainer.getInstance().isTraining()) {
            Trainer.getInstance().stopTraining();
        } else {
            Trainer.getInstance().startTraining();
        }
        updateButtonStates();
    }

    private void onAI() {
        if (AIController.getInstance().isRunning()) {
            AIController.getInstance().stop();
        } else if (selectedModelIndex >= 0 && selectedModelIndex < models.size()) {
            // Use selected model
            close();
            AIController.getInstance().start(models.get(selectedModelIndex));
        } else if (!models.isEmpty()) {
            // Use latest model
            close();
            AIController.getInstance().start();
        }
        updateButtonStates();
    }

    private void onDelete() {
        if (selectedRecordingIndex >= 0 && selectedRecordingIndex < recordings.size()) {
            String name = recordings.get(selectedRecordingIndex).name;
            RecordingManager.getInstance().getStorage().delete(name);
            recordings = RecordingManager.getInstance().getStorage().listRecordings();
            selectedRecordingIndex = -1;
            updateButtonStates();
        }
    }

    private void onDeleteModels() {
        AIController.getInstance().deleteAllModels();
        models = AIController.getInstance().listModels();
        updateButtonStates();
    }

    @Override
    public void tick() {
        super.tick();
        models = AIController.getInstance().listModels();
        updateButtonStates();
    }

    @Override
    public boolean shouldPause() {
        return false; // Don't pause game while dashboard is open
    }
}
