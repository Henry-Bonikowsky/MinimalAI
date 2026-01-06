package com.minimalai.ai;

import com.minimalai.MinimalAI;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.minecraft.block.BlockState;
import net.minecraft.block.entity.BlockEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Detects reward events during RL training.
 * Currently supports: block break detection.
 *
 * Fires callbacks when reward-triggering events occur.
 */
public class RewardDetector {
    private static RewardDetector instance;

    private boolean enabled = false;
    private final List<Consumer<RewardEvent>> listeners = new ArrayList<>();

    // Stats
    private int blocksMinedThisEpisode = 0;
    private int totalBlocksMined = 0;

    public static RewardDetector getInstance() {
        if (instance == null) {
            instance = new RewardDetector();
        }
        return instance;
    }

    private RewardDetector() {
        // Register for block break events
        PlayerBlockBreakEvents.AFTER.register(this::onBlockBreak);
        MinimalAI.LOGGER.info("RewardDetector initialized");
    }

    /**
     * Called when a block is broken.
     */
    private void onBlockBreak(World world, PlayerEntity player, BlockPos pos, BlockState state, BlockEntity blockEntity) {
        if (!enabled) return;

        // Only count if it's the local player (client-side check)
        if (world.isClient && player == net.minecraft.client.MinecraftClient.getInstance().player) {
            blocksMinedThisEpisode++;
            totalBlocksMined++;

            String blockName = state.getBlock().getTranslationKey();
            MinimalAI.LOGGER.info("[REWARD] Block broken: {} at {} (episode total: {})",
                blockName, pos, blocksMinedThisEpisode);

            // Fire reward event
            RewardEvent event = new RewardEvent(RewardType.BLOCK_BROKEN, 1.0f, blockName, pos);
            for (Consumer<RewardEvent> listener : listeners) {
                listener.accept(event);
            }
        }
    }

    /**
     * Enable reward detection.
     */
    public void enable() {
        enabled = true;
        blocksMinedThisEpisode = 0;
        MinimalAI.LOGGER.info("RewardDetector enabled");
    }

    /**
     * Disable reward detection.
     */
    public void disable() {
        enabled = false;
        MinimalAI.LOGGER.info("RewardDetector disabled (mined {} blocks this episode)", blocksMinedThisEpisode);
    }

    /**
     * Reset episode stats.
     */
    public void resetEpisode() {
        blocksMinedThisEpisode = 0;
    }

    /**
     * Add a listener for reward events.
     */
    public void addListener(Consumer<RewardEvent> listener) {
        listeners.add(listener);
    }

    /**
     * Remove a listener.
     */
    public void removeListener(Consumer<RewardEvent> listener) {
        listeners.remove(listener);
    }

    /**
     * Clear all listeners.
     */
    public void clearListeners() {
        listeners.clear();
    }

    public boolean isEnabled() {
        return enabled;
    }

    public int getBlocksMinedThisEpisode() {
        return blocksMinedThisEpisode;
    }

    public int getTotalBlocksMined() {
        return totalBlocksMined;
    }

    // === Event types ===

    public enum RewardType {
        BLOCK_BROKEN,
        ITEM_COLLECTED,
        DEATH,
        CUSTOM
    }

    public static class RewardEvent {
        public final RewardType type;
        public final float reward;
        public final String details;
        public final BlockPos position;

        public RewardEvent(RewardType type, float reward, String details, BlockPos position) {
            this.type = type;
            this.reward = reward;
            this.details = details;
            this.position = position;
        }
    }
}
