package com.minimalai.training;

import java.io.*;
import java.net.Socket;
import java.net.SocketException;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * TCP client that sends experience batches to the Python training server
 * and receives model update notifications.
 *
 * <p>Runs on a background thread to avoid blocking the server tick.
 * Automatically reconnects if the connection drops.
 *
 * <h3>Protocol (big-endian):</h3>
 * <pre>
 * Java -> Python:
 *   [byte: msgType=0x01] [int: dataLen] [byte[]: experienceData]
 *
 * Python -> Java:
 *   [byte: msgType=0x02] [int: pathLen] [byte[]: modelPath(UTF-8)]
 *   [byte: msgType=0x03] [int: dataLen] [byte[]: statsJson(UTF-8)]
 * </pre>
 */
public class TrainingClient {

    // Protocol message types
    private static final byte MSG_EXPERIENCE = 0x01;
    private static final byte MSG_MODEL_UPDATED = 0x02;
    private static final byte MSG_TRAINING_STATS = 0x03;

    private static final long RECONNECT_INTERVAL_MS = 10_000;

    private final String host;
    private final int port;
    private final Logger logger;
    private final Runnable onModelUpdated;

    private final ExecutorService executor;
    private ScheduledExecutorService reconnectScheduler;
    private Socket socket;
    private DataOutputStream out;
    private DataInputStream in;
    private volatile boolean connected;
    private volatile boolean shutdownRequested;

    // Latest training stats (updated by listener thread)
    private volatile String lastStatsJson;

    /**
     * @param host           Python training server hostname
     * @param port           Python training server port
     * @param logger         Plugin logger
     * @param onModelUpdated Called on the main thread when Python signals a model update
     */
    public TrainingClient(String host, int port, Logger logger, Runnable onModelUpdated) {
        this.host = host;
        this.port = port;
        this.logger = logger;
        this.onModelUpdated = onModelUpdated;
        this.executor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "MinimalAI-TrainingClient");
            t.setDaemon(true);
            return t;
        });
    }

    /**
     * Attempt to connect to the Python training server.
     * On success, starts a listener thread for incoming messages.
     */
    public void connect() {
        shutdownRequested = false;
        executor.submit(() -> {
            try {
                socket = new Socket(host, port);
                socket.setTcpNoDelay(true);
                socket.setKeepAlive(true);
                out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
                in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
                connected = true;
                logger.info("[TrainingClient] Connected to " + host + ":" + port);

                // Start listener on a separate daemon thread
                Thread listener = new Thread(this::listenerLoop, "MinimalAI-TrainingListener");
                listener.setDaemon(true);
                listener.start();

            } catch (IOException e) {
                connected = false;
                logger.warning("[TrainingClient] Failed to connect to " + host + ":" + port
                        + " - " + e.getMessage());
                scheduleReconnect();
            }
        });
    }

    /**
     * Disconnect from the server and shut down the executor.
     */
    public void disconnect() {
        shutdownRequested = true;
        connected = false;

        if (reconnectScheduler != null) {
            reconnectScheduler.shutdownNow();
            reconnectScheduler = null;
        }

        closeSocket();

        executor.shutdown();
        try {
            if (!executor.awaitTermination(3, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        logger.info("[TrainingClient] Disconnected");
    }

    public boolean isConnected() {
        return connected;
    }

    /**
     * Get the latest training stats JSON received from the server.
     * May be null if no stats have been received yet.
     */
    public String getLastStatsJson() {
        return lastStatsJson;
    }

    /**
     * Send serialized experience batch to the Python server on the background thread.
     *
     * @param serializedBatch binary experience data matching the expected protocol format
     */
    public void sendExperiences(byte[] serializedBatch) {
        if (!connected) {
            logger.fine("[TrainingClient] Not connected, dropping experience batch");
            return;
        }
        executor.submit(() -> {
            try {
                synchronized (out) {
                    out.writeByte(MSG_EXPERIENCE);
                    out.writeInt(serializedBatch.length);
                    out.write(serializedBatch);
                    out.flush();
                }
            } catch (IOException e) {
                logger.warning("[TrainingClient] Failed to send experiences: " + e.getMessage());
                handleDisconnect();
            }
        });
    }

    // ------------------------------------------------------------------
    //  Listener thread: reads messages from the Python server
    // ------------------------------------------------------------------

    private void listenerLoop() {
        try {
            while (connected && !shutdownRequested) {
                int msgType = in.readByte() & 0xFF;

                switch (msgType) {
                    case MSG_MODEL_UPDATED -> handleModelUpdated();
                    case MSG_TRAINING_STATS -> handleTrainingStats();
                    default -> {
                        logger.warning("[TrainingClient] Unknown message type: 0x"
                                + Integer.toHexString(msgType));
                        // Skip unknown message: read length + payload
                        int len = in.readInt();
                        in.skipBytes(len);
                    }
                }
            }
        } catch (EOFException e) {
            if (!shutdownRequested) {
                logger.info("[TrainingClient] Server closed the connection");
            }
        } catch (SocketException e) {
            if (!shutdownRequested) {
                logger.warning("[TrainingClient] Socket error: " + e.getMessage());
            }
        } catch (IOException e) {
            if (!shutdownRequested) {
                logger.log(Level.WARNING, "[TrainingClient] Listener error", e);
            }
        } finally {
            handleDisconnect();
        }
    }

    private void handleModelUpdated() throws IOException {
        int pathLen = in.readInt();
        byte[] pathBytes = new byte[pathLen];
        in.readFully(pathBytes);
        String modelPath = new String(pathBytes, StandardCharsets.UTF_8);

        logger.info("[TrainingClient] Model updated: " + modelPath);

        if (onModelUpdated != null) {
            // The callback is responsible for scheduling on the main thread if needed.
            // In Paper, the caller should wrap with Bukkit.getScheduler().runTask().
            onModelUpdated.run();
        }
    }

    private void handleTrainingStats() throws IOException {
        int dataLen = in.readInt();
        byte[] dataBytes = new byte[dataLen];
        in.readFully(dataBytes);
        String statsJson = new String(dataBytes, StandardCharsets.UTF_8);

        lastStatsJson = statsJson;
        logger.info("[TrainingClient] Training stats: " + statsJson);
    }

    // ------------------------------------------------------------------
    //  Reconnection logic
    // ------------------------------------------------------------------

    private void handleDisconnect() {
        if (!connected && !shutdownRequested) {
            return; // already handling
        }
        connected = false;
        closeSocket();
        if (!shutdownRequested) {
            scheduleReconnect();
        }
    }

    private void scheduleReconnect() {
        if (shutdownRequested) return;
        if (reconnectScheduler == null || reconnectScheduler.isShutdown()) {
            reconnectScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "MinimalAI-Reconnect");
                t.setDaemon(true);
                return t;
            });
        }
        logger.info("[TrainingClient] Will attempt reconnect in "
                + (RECONNECT_INTERVAL_MS / 1000) + "s");
        reconnectScheduler.schedule(this::reconnect, RECONNECT_INTERVAL_MS, TimeUnit.MILLISECONDS);
    }

    /**
     * Attempt to reconnect to the training server.
     * Called periodically if the connection drops.
     */
    private void reconnect() {
        if (shutdownRequested || connected) return;
        logger.info("[TrainingClient] Attempting reconnect to " + host + ":" + port + "...");
        try {
            socket = new Socket(host, port);
            socket.setTcpNoDelay(true);
            socket.setKeepAlive(true);
            out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));
            in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
            connected = true;
            logger.info("[TrainingClient] Reconnected to " + host + ":" + port);

            Thread listener = new Thread(this::listenerLoop, "MinimalAI-TrainingListener");
            listener.setDaemon(true);
            listener.start();

        } catch (IOException e) {
            connected = false;
            logger.fine("[TrainingClient] Reconnect failed: " + e.getMessage());
            scheduleReconnect();
        }
    }

    private void closeSocket() {
        try {
            if (socket != null && !socket.isClosed()) {
                socket.close();
            }
        } catch (IOException ignored) {
        }
        socket = null;
        out = null;
        in = null;
    }
}
