package com.minimalai.recording;

import com.minimalai.MinimalAI;
import net.fabricmc.loader.api.FabricLoader;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Saves and loads recordings in binary format.
 *
 * File format (version 1):
 * - Magic bytes: "MAIR" (4 bytes)
 * - Version: int (4 bytes)
 * - Name length: int (4 bytes)
 * - Name: UTF bytes
 * - Created at: long (8 bytes)
 * - Duration ms: long (8 bytes)
 * - Avg FPS: float (4 bytes)
 * - Frame count: int (4 bytes)
 * - Frames: [timestamp(long), gameState(64 floats), discreteActions(20 bits packed), continuousActions(2 floats)]
 */
public class RecordingStorage {
    private static final byte[] MAGIC = {'M', 'A', 'I', 'R'};
    private static final String RECORDINGS_DIR = "recordings";
    private static final String EXTENSION = ".mair";

    private final Path recordingsPath;

    public RecordingStorage() {
        Path gameDir = FabricLoader.getInstance().getGameDir();
        this.recordingsPath = gameDir.resolve("minimalai").resolve(RECORDINGS_DIR);
        try {
            Files.createDirectories(recordingsPath);
        } catch (IOException e) {
            MinimalAI.LOGGER.error("Failed to create recordings directory", e);
        }
    }

    public void save(Recording recording) throws IOException {
        Path filePath = recordingsPath.resolve(recording.getName() + EXTENSION);

        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(filePath.toFile())))) {

            // Magic and version
            out.write(MAGIC);
            out.writeInt(recording.getVersion());

            // Metadata
            out.writeUTF(recording.getName());
            out.writeLong(recording.getCreatedAt());
            out.writeLong(recording.getDurationMs());
            out.writeFloat(recording.getAvgFps());

            // Frames
            List<Recording.Frame> frames = recording.getFrames();
            out.writeInt(frames.size());

            for (Recording.Frame frame : frames) {
                out.writeLong(frame.timestamp);

                // Game state (64 floats)
                for (float v : frame.gameState) {
                    out.writeFloat(v);
                }

                // Discrete actions (20 booleans packed into 3 bytes)
                int packed = 0;
                for (int i = 0; i < frame.discreteActions.length; i++) {
                    if (frame.discreteActions[i]) {
                        packed |= (1 << i);
                    }
                }
                out.writeInt(packed);

                // Continuous actions (2 floats)
                for (float v : frame.continuousActions) {
                    out.writeFloat(v);
                }
            }
        }

        MinimalAI.LOGGER.info("Saved recording '{}' with {} frames to {}",
            recording.getName(), recording.getFrameCount(), filePath);
    }

    public Recording load(String name) throws IOException {
        Path filePath = recordingsPath.resolve(name + EXTENSION);
        return loadFromPath(filePath);
    }

    public Recording loadFromPath(Path filePath) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filePath.toFile())))) {

            // Verify magic
            byte[] magic = new byte[4];
            in.readFully(magic);
            for (int i = 0; i < 4; i++) {
                if (magic[i] != MAGIC[i]) {
                    throw new IOException("Invalid recording file (bad magic)");
                }
            }

            // Version
            int version = in.readInt();
            if (version > Recording.VERSION) {
                throw new IOException("Recording version " + version + " is newer than supported " + Recording.VERSION);
            }

            // Metadata
            String name = in.readUTF();
            long createdAt = in.readLong();
            long durationMs = in.readLong();
            float avgFps = in.readFloat();

            Recording recording = new Recording(name);
            recording.setCreatedAt(createdAt);
            recording.setDurationMs(durationMs);
            recording.setAvgFps(avgFps);
            recording.setVersion(version);

            // Frames
            int frameCount = in.readInt();

            for (int f = 0; f < frameCount; f++) {
                long timestamp = in.readLong();

                // Game state (64 floats)
                float[] gameState = new float[GameStateCollector.STATE_SIZE];
                for (int i = 0; i < gameState.length; i++) {
                    gameState[i] = in.readFloat();
                }

                // Discrete actions (packed int)
                int packed = in.readInt();
                boolean[] discreteActions = new boolean[ActionRecorder.DISCRETE_ACTION_COUNT];
                for (int i = 0; i < discreteActions.length; i++) {
                    discreteActions[i] = (packed & (1 << i)) != 0;
                }

                // Continuous actions (2 floats)
                float[] continuousActions = new float[ActionRecorder.CONTINUOUS_ACTION_COUNT];
                for (int i = 0; i < continuousActions.length; i++) {
                    continuousActions[i] = in.readFloat();
                }

                recording.addFrame(new Recording.Frame(timestamp, gameState, discreteActions, continuousActions));
            }

            MinimalAI.LOGGER.info("Loaded recording '{}' with {} frames", name, frameCount);
            return recording;
        }
    }

    public List<RecordingInfo> listRecordings() {
        List<RecordingInfo> recordings = new ArrayList<>();

        File[] files = recordingsPath.toFile().listFiles((dir, name) -> name.endsWith(EXTENSION));
        if (files == null) return recordings;

        for (File file : files) {
            try {
                RecordingInfo info = readInfo(file.toPath());
                recordings.add(info);
            } catch (IOException e) {
                MinimalAI.LOGGER.warn("Failed to read recording info: {}", file.getName(), e);
            }
        }

        // Sort by creation time, newest first
        recordings.sort((a, b) -> Long.compare(b.createdAt, a.createdAt));
        return recordings;
    }

    private RecordingInfo readInfo(Path filePath) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(filePath.toFile())))) {

            // Skip magic
            in.skipBytes(4);

            int version = in.readInt();
            String name = in.readUTF();
            long createdAt = in.readLong();
            long durationMs = in.readLong();
            float avgFps = in.readFloat();
            int frameCount = in.readInt();

            return new RecordingInfo(name, createdAt, durationMs, avgFps, frameCount, filePath);
        }
    }

    public boolean delete(String name) {
        Path filePath = recordingsPath.resolve(name + EXTENSION);
        try {
            return Files.deleteIfExists(filePath);
        } catch (IOException e) {
            MinimalAI.LOGGER.error("Failed to delete recording: {}", name, e);
            return false;
        }
    }

    public Path getRecordingsPath() {
        return recordingsPath;
    }

    /**
     * Lightweight info about a recording without loading all frames.
     */
    public static class RecordingInfo {
        public final String name;
        public final long createdAt;
        public final long durationMs;
        public final float avgFps;
        public final int frameCount;
        public final Path filePath;

        public RecordingInfo(String name, long createdAt, long durationMs, float avgFps, int frameCount, Path filePath) {
            this.name = name;
            this.createdAt = createdAt;
            this.durationMs = durationMs;
            this.avgFps = avgFps;
            this.frameCount = frameCount;
            this.filePath = filePath;
        }

        public String getFormattedDuration() {
            long seconds = durationMs / 1000;
            long minutes = seconds / 60;
            seconds = seconds % 60;
            return String.format("%d:%02d", minutes, seconds);
        }
    }
}
