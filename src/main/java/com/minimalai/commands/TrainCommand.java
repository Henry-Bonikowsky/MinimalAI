package com.minimalai.commands;

import com.minimalai.training.EpisodeManager;
import com.minimalai.training.RewardComputer;

import org.bukkit.ChatColor;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabCompleter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Command executor for {@code /maitrain start|stop|status}.
 *
 * <p>Registered in plugin.yml under "maitrain" with permission minimalai.train.
 * Controls the training mode boolean that BotBrain checks each tick to decide
 * whether to collect experience.
 */
public class TrainCommand implements CommandExecutor, TabCompleter {

    private static final String PREFIX = ChatColor.GRAY + "[" + ChatColor.AQUA + "MinimalAI" + ChatColor.GRAY + "] " + ChatColor.RESET;
    private static final List<String> SUB_COMMANDS = Arrays.asList("start", "stop", "status");

    // External references
    private final RewardComputer rewardComputer;
    private final EpisodeManager episodeManager;
    private final com.minimalai.training.ExperienceBuffer experienceBuffer;

    // Training state
    private volatile boolean trainingEnabled = false;
    private int episodesCompleted = 0;
    private boolean connectedToPython = false;

    // Reward tracking for status display
    private final float[] recentRewards = new float[10];
    private int rewardWriteIndex = 0;
    private int rewardCount = 0;

    /**
     * @param rewardComputer   the reward computer (Bukkit event listener)
     * @param episodeManager   the per-bot episode manager
     * @param experienceBuffer the shared experience buffer for TCP transmission
     */
    public TrainCommand(RewardComputer rewardComputer,
                        EpisodeManager episodeManager,
                        com.minimalai.training.ExperienceBuffer experienceBuffer) {
        this.rewardComputer = rewardComputer;
        this.episodeManager = episodeManager;
        this.experienceBuffer = experienceBuffer;
    }

    // ----------------------------------------------------------------
    //  Command dispatch
    // ----------------------------------------------------------------

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 0) {
            sendUsage(sender);
            return true;
        }

        switch (args[0].toLowerCase()) {
            case "start"  -> handleStart(sender);
            case "stop"   -> handleStop(sender);
            case "status" -> handleStatus(sender);
            default       -> sendUsage(sender);
        }
        return true;
    }

    // ----------------------------------------------------------------
    //  /maitrain start
    // ----------------------------------------------------------------

    private void handleStart(CommandSender sender) {
        if (trainingEnabled) {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "Training is already running.");
            return;
        }

        trainingEnabled = true;
        sender.sendMessage(PREFIX + ChatColor.GREEN + "Training mode enabled. Bots will now collect experience.");
    }

    // ----------------------------------------------------------------
    //  /maitrain stop
    // ----------------------------------------------------------------

    private void handleStop(CommandSender sender) {
        if (!trainingEnabled) {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "Training is not running.");
            return;
        }

        trainingEnabled = false;
        sender.sendMessage(PREFIX + ChatColor.GREEN + "Training mode disabled.");
    }

    // ----------------------------------------------------------------
    //  /maitrain status
    // ----------------------------------------------------------------

    private void handleStatus(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.GREEN + "--- Training Status ---");

        String stateColor = trainingEnabled ? ChatColor.GREEN.toString() : ChatColor.RED.toString();
        String stateLabel = trainingEnabled ? "ENABLED" : "DISABLED";
        sender.sendMessage(ChatColor.GRAY + "  Training: " + stateColor + stateLabel);

        sender.sendMessage(ChatColor.GRAY + "  Buffer: " + ChatColor.WHITE + experienceBuffer.size()
            + "/" + experienceBuffer.capacity());

        sender.sendMessage(ChatColor.GRAY + "  Episodes completed: " + ChatColor.WHITE + episodesCompleted);

        String pyColor = connectedToPython ? ChatColor.GREEN.toString() : ChatColor.RED.toString();
        String pyLabel = connectedToPython ? "YES" : "NO";
        sender.sendMessage(ChatColor.GRAY + "  Python server: " + pyColor + pyLabel);

        sender.sendMessage(ChatColor.GRAY + "  Avg reward (last 10): " + ChatColor.WHITE
            + String.format("%.2f", getAverageReward()));
    }

    // ----------------------------------------------------------------
    //  Usage
    // ----------------------------------------------------------------

    private void sendUsage(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.YELLOW + "Usage:");
        sender.sendMessage(ChatColor.GRAY + "  /maitrain start" + ChatColor.WHITE + " - Enable training mode");
        sender.sendMessage(ChatColor.GRAY + "  /maitrain stop" + ChatColor.WHITE + " - Disable training mode");
        sender.sendMessage(ChatColor.GRAY + "  /maitrain status" + ChatColor.WHITE + " - Show training status");
    }

    // ----------------------------------------------------------------
    //  Tab completion
    // ----------------------------------------------------------------

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        if (args.length == 1) {
            String lower = args[0].toLowerCase();
            return SUB_COMMANDS.stream()
                .filter(s -> s.startsWith(lower))
                .collect(Collectors.toCollection(ArrayList::new));
        }
        return List.of();
    }

    // ----------------------------------------------------------------
    //  State accessors (for BotBrain / training loop)
    // ----------------------------------------------------------------

    /**
     * Whether training mode is active. Checked each tick by BotBrain.
     */
    public boolean isTrainingEnabled() {
        return trainingEnabled;
    }

    /**
     * Called by the training loop when an episode finishes.
     */
    public void recordEpisodeEnd(float totalReward) {
        episodesCompleted++;
        recentRewards[rewardWriteIndex] = totalReward;
        rewardWriteIndex = (rewardWriteIndex + 1) % recentRewards.length;
        if (rewardCount < recentRewards.length) rewardCount++;
    }

    /**
     * Set the Python connection status (updated by TCP bridge).
     */
    public void setConnectedToPython(boolean connected) {
        this.connectedToPython = connected;
    }

    // ----------------------------------------------------------------
    //  Internal helpers
    // ----------------------------------------------------------------

    private float getAverageReward() {
        if (rewardCount == 0) return 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < rewardCount; i++) {
            sum += recentRewards[i];
        }
        return sum / rewardCount;
    }
}
