package com.minimalai.ai;

import java.util.ArrayList;
import java.util.List;

/**
 * Stores experiences from a single episode for REINFORCE training.
 * Each step stores: stacked state, actions taken, log probabilities, and value estimate.
 * Reward is assigned at episode end.
 */
public class ExperienceBuffer {

    private final List<Experience> experiences = new ArrayList<>();
    private boolean episodeComplete = false;
    private float episodeReward = 0;

    /**
     * Record a single step.
     */
    public void addStep(float[] stackedState, boolean[] actions, float[] logProbs, float[] camera, float value) {
        experiences.add(new Experience(
            stackedState.clone(),
            actions.clone(),
            logProbs.clone(),
            camera.clone(),
            value
        ));
    }

    /**
     * Mark episode as complete with the final reward.
     * In sparse reward setting, this is typically 1.0 for success, 0.0 for timeout/failure.
     */
    public void setEpisodeReward(float reward) {
        this.episodeReward = reward;
        this.episodeComplete = true;
    }

    /**
     * Get all experiences from this episode.
     */
    public List<Experience> getExperiences() {
        return experiences;
    }

    /**
     * Get the episode reward.
     */
    public float getEpisodeReward() {
        return episodeReward;
    }

    /**
     * Check if episode is complete.
     */
    public boolean isComplete() {
        return episodeComplete;
    }

    /**
     * Get number of steps in this episode.
     */
    public int size() {
        return experiences.size();
    }

    /**
     * Clear the buffer for a new episode.
     */
    public void clear() {
        experiences.clear();
        episodeComplete = false;
        episodeReward = 0;
    }

    /**
     * Compute returns (discounted cumulative rewards) for each step.
     * With sparse reward at end, return for step t = gamma^(T-t) * reward
     * where T is the final step.
     */
    public float[] computeReturns(float gamma) {
        int n = experiences.size();
        float[] returns = new float[n];

        // With sparse reward only at end, all steps get discounted final reward
        for (int t = 0; t < n; t++) {
            int stepsToEnd = n - 1 - t;
            returns[t] = (float) Math.pow(gamma, stepsToEnd) * episodeReward;
        }

        return returns;
    }

    /**
     * Compute advantages (return - baseline value).
     */
    public float[] computeAdvantages(float gamma) {
        float[] returns = computeReturns(gamma);
        float[] advantages = new float[experiences.size()];

        for (int i = 0; i < experiences.size(); i++) {
            advantages[i] = returns[i] - experiences.get(i).value;
        }

        return advantages;
    }

    /**
     * Normalize advantages to have mean 0 and std 1.
     * Helps with training stability.
     */
    public float[] normalizeAdvantages(float[] advantages) {
        if (advantages.length == 0) return advantages;

        // Compute mean
        float mean = 0;
        for (float a : advantages) mean += a;
        mean /= advantages.length;

        // Compute std
        float variance = 0;
        for (float a : advantages) {
            float diff = a - mean;
            variance += diff * diff;
        }
        variance /= advantages.length;
        float std = (float) Math.sqrt(variance + 1e-8);

        // Normalize
        float[] normalized = new float[advantages.length];
        for (int i = 0; i < advantages.length; i++) {
            normalized[i] = (advantages[i] - mean) / std;
        }

        return normalized;
    }

    /**
     * Single experience tuple.
     */
    public static class Experience {
        public final float[] stackedState;
        public final boolean[] actions;
        public final float[] logProbs;
        public final float[] camera;
        public final float value;

        public Experience(float[] stackedState, boolean[] actions, float[] logProbs, float[] camera, float value) {
            this.stackedState = stackedState;
            this.actions = actions;
            this.logProbs = logProbs;
            this.camera = camera;
            this.value = value;
        }
    }
}
