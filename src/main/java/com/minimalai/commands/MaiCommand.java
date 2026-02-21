package com.minimalai.commands;

import com.minimalai.ai.ModelManager;
import com.minimalai.bot.BotBrain;
import com.minimalai.bot.FakePlayerManager;
import com.minimalai.bot.FakePlayerManager.BotContext;
import com.minimalai.bot.KitManager;
import com.minimalai.training.EpisodeManager;
import com.minimalai.training.ExperienceBuffer;
import com.minimalai.training.PhysicsRecorder;
import com.minimalai.training.RewardComputer;
import com.minimalai.training.TrainingClient;

import net.minecraft.server.level.ServerPlayer;
import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabCompleter;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.jetbrains.annotations.Nullable;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Unified command handler for {@code /mai} (aliases: /minimalai, /minai).
 * Handles all subcommands: model, train, bot, kit, status, stop, help.
 */
public class MaiCommand implements CommandExecutor, TabCompleter, Listener {

    private static final Logger LOG = Logger.getLogger("MinimalAI");
    private static final String PREFIX = ChatColor.GRAY + "[" + ChatColor.AQUA + "MinimalAI" + ChatColor.GRAY + "] " + ChatColor.RESET;
    private static final int SELFPLAY_EPISODE_TICKS = 6000; // 5 minutes
    private static final List<String> TOP_COMMANDS = Arrays.asList("model", "train", "bot", "kit", "record", "status", "stop", "help");
    private static final List<String> MODEL_SUBS = Arrays.asList("create", "delete", "list", "load", "duel", "default");
    private static final List<String> BOT_SUBS = Arrays.asList("spawn", "despawn", "list");
    private static final List<String> KIT_SUBS = Arrays.asList("save", "delete", "list", "default");
    private static final List<String> TEAM_SIZES = Arrays.asList("2v2", "3v3", "5v5");

    // Service references
    private final BotCommand botCmd;
    private final ModelManager modelManager;
    private final KitManager kitManager;
    private final RewardComputer rewardComputer;
    private final EpisodeManager episodeManager;
    private final ExperienceBuffer experienceBuffer;
    private final TrainingClient trainingClient;

    // Training state
    private volatile boolean trainingEnabled = false;
    private int episodesCompleted = 0;
    private final float[] recentRewards = new float[10];
    private int rewardWriteIndex = 0;
    private int rewardCount = 0;

    // Defaults
    private @Nullable String defaultModel = null;
    private @Nullable String defaultKit = null;

    // Active sessions (at most one at a time)
    private @Nullable SelfPlaySession selfPlay;
    private @Nullable DuelSession duel;

    // ================================================================
    //  Inner classes
    // ================================================================

    static class SelfPlaySession {
        final List<String> teamA = new ArrayList<>();
        final List<String> teamB = new ArrayList<>();
        final Set<String> deadBots = new HashSet<>();
        final Location spawnCenter;
        final String modelName;
        final @Nullable String kitName;
        final int maxRounds; // 0 = infinite
        int roundNumber = 0;

        SelfPlaySession(Location center, String model, @Nullable String kit, int maxRounds) {
            this.spawnCenter = center;
            this.modelName = model;
            this.kitName = kit;
            this.maxRounds = maxRounds;
        }

        boolean isBot(String name) { return teamA.contains(name) || teamB.contains(name); }

        @Nullable String getTeam(String name) {
            if (teamA.contains(name)) return "A";
            if (teamB.contains(name)) return "B";
            return null;
        }

        boolean isTeamWiped(String team) {
            List<String> members = "A".equals(team) ? teamA : teamB;
            return deadBots.containsAll(members);
        }

        List<String> allBots() {
            List<String> all = new ArrayList<>(teamA.size() + teamB.size());
            all.addAll(teamA);
            all.addAll(teamB);
            return all;
        }
    }

    static class DuelSession {
        final String botName;
        final String modelName;
        final @Nullable String kitName;
        final Location spawnLoc;
        final UUID playerUUID;
        int deaths = 0;

        DuelSession(String botName, String model, @Nullable String kit, Location spawn, UUID player) {
            this.botName = botName;
            this.modelName = model;
            this.kitName = kit;
            this.spawnLoc = spawn;
            this.playerUUID = player;
        }
    }

    // ================================================================
    //  Constructor
    // ================================================================

    public MaiCommand(BotCommand botCmd,
                      ModelManager modelManager,
                      KitManager kitManager,
                      RewardComputer rewardComputer,
                      EpisodeManager episodeManager,
                      ExperienceBuffer experienceBuffer,
                      TrainingClient trainingClient) {
        this.botCmd = botCmd;
        this.modelManager = modelManager;
        this.kitManager = kitManager;
        this.rewardComputer = rewardComputer;
        this.episodeManager = episodeManager;
        this.experienceBuffer = experienceBuffer;
        this.trainingClient = trainingClient;
    }

    // ================================================================
    //  Command dispatch
    // ================================================================

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 0) { sendHelp(sender); return true; }

        switch (args[0].toLowerCase()) {
            case "model"  -> handleModel(sender, drop(args));
            case "train"  -> handleTrain(sender, drop(args));
            case "bot"    -> handleBot(sender, drop(args));
            case "kit"    -> handleKit(sender, drop(args));
            case "record" -> handleRecord(sender, drop(args));
            case "status" -> handleStatus(sender);
            case "stop"   -> handleStopAll(sender);
            case "help"   -> sendHelp(sender);
            default       -> sendHelp(sender);
        }
        return true;
    }

    // ================================================================
    //  /mai model
    // ================================================================

    private void handleModel(CommandSender sender, String[] args) {
        if (args.length == 0) { sendModelHelp(sender); return; }
        switch (args[0].toLowerCase()) {
            case "create"  -> handleModelCreate(sender, args);
            case "delete"  -> handleModelDelete(sender, args);
            case "list"    -> handleModelList(sender);
            case "load"    -> handleModelLoad(sender, args);
            case "duel"    -> handleModelDuel(sender, args);
            case "default" -> handleModelDefault(sender, args);
            default        -> sendModelHelp(sender);
        }
    }

    private void handleModelCreate(CommandSender sender, String[] args) {
        if (args.length < 2) { sender.sendMessage(PREFIX + ChatColor.RED + "Usage: /mai model create <name>"); return; }
        try {
            modelManager.createFreshModel(args[1]);
            modelManager.loadModel(args[1]);
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Created model '" + args[1] + "' (random weights).");
        } catch (Exception e) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Failed: " + e.getMessage());
        }
    }

    private void handleModelDelete(CommandSender sender, String[] args) {
        if (args.length < 2) { sender.sendMessage(PREFIX + ChatColor.RED + "Usage: /mai model delete <name>"); return; }
        if (modelManager.removeModel(args[1])) {
            if (args[1].equals(defaultModel)) defaultModel = null;
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Deleted model '" + args[1] + "'.");
        } else {
            sender.sendMessage(PREFIX + ChatColor.RED + "Model '" + args[1] + "' not found.");
        }
    }

    private void handleModelList(CommandSender sender) {
        List<String> models = modelManager.listModels();
        if (models.isEmpty()) { sender.sendMessage(PREFIX + ChatColor.YELLOW + "No models found."); return; }
        sender.sendMessage(PREFIX + ChatColor.GREEN + "Models (" + models.size() + "):");
        for (String m : models) {
            String def = m.equals(defaultModel) ? ChatColor.AQUA + " (default)" : "";
            sender.sendMessage(ChatColor.GRAY + "  - " + ChatColor.WHITE + m + def);
        }
    }

    private void handleModelLoad(CommandSender sender, String[] args) {
        if (args.length < 2) { sender.sendMessage(PREFIX + ChatColor.RED + "Usage: /mai model load <name>"); return; }
        try {
            modelManager.loadModel(args[1]);
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Loaded model '" + args[1] + "'.");
        } catch (Exception e) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Failed: " + e.getMessage());
        }
    }

    private void handleModelDuel(CommandSender sender, String[] args) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Only players can duel.");
            return;
        }
        if (selfPlay != null || duel != null) {
            sender.sendMessage(PREFIX + ChatColor.RED + "A session is already active. /mai stop first.");
            return;
        }
        if (trainingEnabled) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Training is active. /mai stop first.");
            return;
        }

        // Parse: duel [model] [kit]
        String modelName = null;
        String kitName = null;
        for (int i = 1; i < args.length; i++) {
            if (modelName == null && modelManager.listModels().contains(args[i])) {
                modelName = args[i];
            } else if (kitName == null) {
                kitName = args[i];
            }
        }
        if (modelName == null) modelName = defaultModel;
        if (modelName == null && !modelManager.listModels().isEmpty()) modelName = modelManager.listModels().get(0);
        if (modelName == null) {
            sender.sendMessage(PREFIX + ChatColor.RED + "No model available. /mai model create <name>");
            return;
        }
        if (kitName != null && !kitManager.listKits().contains(kitName)) {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "Kit '" + kitName + "' not found, using default.");
            kitName = defaultKit;
        } else if (kitName == null) {
            kitName = defaultKit;
        }

        // Spawn bot 5 blocks in front of the player, facing them
        Location playerLoc = player.getLocation();
        double yaw = Math.toRadians(playerLoc.getYaw());
        Location botLoc = playerLoc.clone().add(-Math.sin(yaw) * 5, 0, Math.cos(yaw) * 5);
        botLoc.setYaw(playerLoc.getYaw() + 180);
        botLoc.setPitch(0);

        String botName = "Duel_Bot";
        BotContext ctx = botCmd.spawnBot(playerLoc.getWorld(), botLoc, botName, modelName, kitName);
        if (ctx == null) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Failed to spawn duel bot.");
            return;
        }

        duel = new DuelSession(botName, modelName, kitName, botLoc, player.getUniqueId());

        // Enable training
        trainingEnabled = true;
        if (!trainingClient.isConnected()) trainingClient.connect();
        BotBrain brain = botCmd.getBrains().get(botName);
        if (brain != null) {
            brain.setTrainingEnabled(true);
            episodeManager.startEpisode(botName, SELFPLAY_EPISODE_TICKS);
        }

        sender.sendMessage(PREFIX + ChatColor.GREEN + "Duel started! Model '" + modelName + "'"
                + (kitName != null ? " kit '" + kitName + "'" : ""));
        sender.sendMessage(PREFIX + ChatColor.GRAY + "The bot learns from your play. Stop with /mai stop");
    }

    private void handleModelDefault(CommandSender sender, String[] args) {
        if (args.length < 2) {
            sender.sendMessage(PREFIX + ChatColor.GRAY + "Default model: "
                    + (defaultModel != null ? ChatColor.WHITE + defaultModel : ChatColor.YELLOW + "none"));
            return;
        }
        if ("none".equalsIgnoreCase(args[1]) || "clear".equalsIgnoreCase(args[1])) {
            defaultModel = null;
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Default model cleared.");
            return;
        }
        if (!modelManager.listModels().contains(args[1])) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Model '" + args[1] + "' not found.");
            return;
        }
        defaultModel = args[1];
        sender.sendMessage(PREFIX + ChatColor.GREEN + "Default model set to '" + args[1] + "'.");
    }

    // ================================================================
    //  /mai train
    // ================================================================

    private void handleTrain(CommandSender sender, String[] args) {
        if (args.length == 0) { sendTrainHelp(sender); return; }
        if ("stop".equalsIgnoreCase(args[0])) { handleTrainStop(sender); return; }
        if ("group".equalsIgnoreCase(args[0])) { handleTrainGroup(sender, drop(args)); return; }
        handleTrainDirect(sender, args);
    }

    /** /mai train [model] [kit] [times] — 1v1 self-play */
    private void handleTrainDirect(CommandSender sender, String[] args) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Only players can start training.");
            return;
        }
        if (selfPlay != null || duel != null) {
            sender.sendMessage(PREFIX + ChatColor.RED + "A session is already active. /mai stop first.");
            return;
        }

        String modelName = null;
        String kitName = null;
        int times = 0;
        for (String arg : args) {
            if (isNumber(arg)) { times = Integer.parseInt(arg); }
            else if (modelName == null && modelManager.listModels().contains(arg)) { modelName = arg; }
            else if (kitName == null) { kitName = arg; }
        }

        modelName = resolveModel(modelName);
        kitName = resolveKit(kitName);
        if (modelName == null) { sender.sendMessage(PREFIX + ChatColor.RED + "No model available."); return; }

        startSelfPlay(sender, player.getLocation(), 1, modelName, kitName, times);
    }

    /** /mai train group <size> [model] [kit] [times] */
    private void handleTrainGroup(CommandSender sender, String[] args) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Only players can start training.");
            return;
        }
        if (args.length == 0) { sendTrainHelp(sender); return; }
        if (selfPlay != null || duel != null) {
            sender.sendMessage(PREFIX + ChatColor.RED + "A session is already active. /mai stop first.");
            return;
        }

        String sizeStr = null;
        String modelName = null;
        String kitName = null;
        int times = 0;
        for (String arg : args) {
            if (TEAM_SIZES.contains(arg.toLowerCase())) { sizeStr = arg.toLowerCase(); }
            else if (isNumber(arg)) { times = Integer.parseInt(arg); }
            else if (modelName == null && modelManager.listModels().contains(arg)) { modelName = arg; }
            else if (kitName == null) { kitName = arg; }
        }

        if (sizeStr == null) sizeStr = "3v3";
        modelName = resolveModel(modelName);
        kitName = resolveKit(kitName);
        if (modelName == null) { sender.sendMessage(PREFIX + ChatColor.RED + "No model available."); return; }

        int teamSize = switch (sizeStr) {
            case "2v2" -> 2;
            case "5v5" -> 5;
            default -> 3;
        };
        startSelfPlay(sender, player.getLocation(), teamSize, modelName, kitName, times);
    }

    private void handleTrainStop(CommandSender sender) {
        if (duel != null) { stopDuel(sender); }
        else if (selfPlay != null) { stopSelfPlay(sender); }
        else if (trainingEnabled) {
            trainingEnabled = false;
            for (BotBrain brain : botCmd.getBrains().values()) {
                brain.setTrainingEnabled(false);
                if (episodeManager.isActive(brain.getName()))
                    episodeManager.endEpisode(brain.getName(), EpisodeManager.EpisodeEndReason.MANUAL);
            }
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Training stopped.");
        } else {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "Nothing is running.");
        }
    }

    // ================================================================
    //  /mai bot
    // ================================================================

    private void handleBot(CommandSender sender, String[] args) {
        if (args.length == 0) { sendBotHelp(sender); return; }
        switch (args[0].toLowerCase()) {
            case "spawn"   -> handleBotSpawn(sender, args);
            case "despawn" -> handleBotDespawn(sender, args);
            case "list"    -> handleBotList(sender);
            default        -> sendBotHelp(sender);
        }
    }

    private void handleBotSpawn(CommandSender sender, String[] args) {
        Location loc;
        if (sender instanceof Player player) {
            loc = player.getLocation();
        } else {
            // Console/RCON: spawn <name> <world> <x> <y> <z> [model] [kit]
            if (args.length < 6) {
                sender.sendMessage(PREFIX + ChatColor.RED + "Console usage: /mai bot spawn <name> <world> <x> <y> <z> [model] [kit]");
                return;
            }
            World w = Bukkit.getWorld(args[2]);
            if (w == null) { sender.sendMessage(PREFIX + ChatColor.RED + "World '" + args[2] + "' not found."); return; }
            try {
                loc = new Location(w, Double.parseDouble(args[3]), Double.parseDouble(args[4]), Double.parseDouble(args[5]));
            } catch (NumberFormatException e) {
                sender.sendMessage(PREFIX + ChatColor.RED + "Invalid coordinates."); return;
            }
            String cName = args[1];
            String cModel = args.length >= 7 ? args[6] : defaultModel;
            String cKit = args.length >= 8 ? args[7] : defaultKit;
            if (cKit != null && !kitManager.listKits().contains(cKit)) cKit = null;
            BotContext ctx = botCmd.spawnBot(w, loc, cName, cModel, cKit);
            if (ctx == null) { sender.sendMessage(PREFIX + ChatColor.RED + "Failed to spawn bot."); return; }
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Spawned '" + ctx.name() + "'.");
            return;
        }
        // Positional: spawn [name] [model] [kit]
        String name = args.length >= 2 ? args[1] : null;
        String modelName = args.length >= 3 ? args[2] : defaultModel;
        String kitName = args.length >= 4 ? args[3] : defaultKit;

        if (kitName != null && !kitManager.listKits().contains(kitName)) {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "Kit '" + kitName + "' not found.");
            kitName = null;
        }

        BotContext ctx = botCmd.spawnBot(loc.getWorld(), loc, name, modelName, kitName);
        if (ctx == null) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Failed to spawn bot. Load a model first.");
            return;
        }
        sender.sendMessage(PREFIX + ChatColor.GREEN + "Spawned '" + ctx.name() + "'.");
    }

    private void handleBotDespawn(CommandSender sender, String[] args) {
        if (args.length < 2) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Usage: /mai bot despawn <name|*>");
            return;
        }
        String target = args[1];
        if ("*".equals(target)) {
            int count = botCmd.getBotManager().botCount();
            if (count == 0) { sender.sendMessage(PREFIX + ChatColor.YELLOW + "No bots to remove."); return; }
            // Stop active sessions first
            if (selfPlay != null) stopSelfPlay(null);
            if (duel != null) stopDuel(null);
            botCmd.closeAll();
            botCmd.getBotManager().despawnAll();
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Removed all " + count + " bot(s).");
        } else {
            botCmd.removeBrain(target);
            if (botCmd.getBotManager().despawn(target)) {
                sender.sendMessage(PREFIX + ChatColor.GREEN + "Removed '" + target + "'.");
            } else {
                sender.sendMessage(PREFIX + ChatColor.RED + "No bot named '" + target + "'.");
            }
        }
    }

    private void handleBotList(CommandSender sender) {
        Collection<BotContext> bots = botCmd.getBotManager().getAllBots();
        if (bots.isEmpty()) { sender.sendMessage(PREFIX + ChatColor.YELLOW + "No active bots."); return; }
        sender.sendMessage(PREFIX + ChatColor.GREEN + "Active bots (" + bots.size() + "):");
        for (BotContext ctx : bots) {
            Location loc = ctx.bukkitLocation();
            BotBrain brain = botCmd.getBrains().get(ctx.name());
            String status = brain != null ? ChatColor.GREEN + " [AI]" : ChatColor.RED + " [idle]";
            String hp = String.format("%.1f", ctx.serverPlayer().getHealth());
            String deaths = brain != null ? String.valueOf(brain.getDeaths()) : "?";
            sender.sendMessage(ChatColor.GRAY + "  - " + ChatColor.WHITE + ctx.name()
                    + status + ChatColor.GRAY + " HP:" + ChatColor.YELLOW + hp
                    + ChatColor.GRAY + " Deaths:" + ChatColor.YELLOW + deaths
                    + ChatColor.GRAY + " at " + String.format("%.0f, %.0f, %.0f", loc.getX(), loc.getY(), loc.getZ()));
        }
    }

    // ================================================================
    //  /mai kit
    // ================================================================

    private void handleKit(CommandSender sender, String[] args) {
        if (args.length == 0) { sendKitHelp(sender); return; }
        switch (args[0].toLowerCase()) {
            case "save"    -> handleKitSave(sender, args);
            case "delete"  -> handleKitDelete(sender, args);
            case "list"    -> handleKitList(sender);
            case "default" -> handleKitDefault(sender, args);
            default        -> sendKitHelp(sender);
        }
    }

    private void handleKitSave(CommandSender sender, String[] args) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Only players can save kits.");
            return;
        }
        if (args.length < 2) { sender.sendMessage(PREFIX + ChatColor.RED + "Usage: /mai kit save <name>"); return; }
        try {
            kitManager.saveKit(player, args[1].toLowerCase());
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Saved kit '" + args[1].toLowerCase() + "'.");
        } catch (Exception e) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Failed: " + e.getMessage());
        }
    }

    private void handleKitDelete(CommandSender sender, String[] args) {
        if (args.length < 2) { sender.sendMessage(PREFIX + ChatColor.RED + "Usage: /mai kit delete <name>"); return; }
        if (kitManager.deleteKit(args[1])) {
            if (args[1].equals(defaultKit)) defaultKit = null;
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Deleted kit '" + args[1] + "'.");
        } else {
            sender.sendMessage(PREFIX + ChatColor.RED + "Kit '" + args[1] + "' not found.");
        }
    }

    private void handleKitList(CommandSender sender) {
        List<String> kits = kitManager.listKits();
        if (kits.isEmpty()) {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "No saved kits. Use /mai kit save <name>");
            return;
        }
        sender.sendMessage(PREFIX + ChatColor.GREEN + "Kits (" + kits.size() + "):");
        for (String k : kits) {
            String def = k.equals(defaultKit) ? ChatColor.AQUA + " (default)" : "";
            sender.sendMessage(ChatColor.GRAY + "  - " + ChatColor.WHITE + k + def);
        }
    }

    private void handleKitDefault(CommandSender sender, String[] args) {
        if (args.length < 2) {
            sender.sendMessage(PREFIX + ChatColor.GRAY + "Default kit: "
                    + (defaultKit != null ? ChatColor.WHITE + defaultKit : ChatColor.YELLOW + "none"));
            return;
        }
        if ("none".equalsIgnoreCase(args[1]) || "clear".equalsIgnoreCase(args[1])) {
            defaultKit = null;
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Default kit cleared.");
            return;
        }
        if (!kitManager.listKits().contains(args[1])) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Kit '" + args[1] + "' not found.");
            return;
        }
        defaultKit = args[1];
        sender.sendMessage(PREFIX + ChatColor.GREEN + "Default kit set to '" + args[1] + "'.");
    }

    // ================================================================
    //  /mai status
    // ================================================================

    private void handleStatus(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.GREEN + "--- MinimalAI Status ---");

        String stateColor = trainingEnabled ? ChatColor.GREEN.toString() : ChatColor.RED.toString();
        sender.sendMessage(ChatColor.GRAY + "  Training: " + stateColor + (trainingEnabled ? "ENABLED" : "DISABLED"));

        boolean pyConn = trainingClient.isConnected();
        sender.sendMessage(ChatColor.GRAY + "  Python: " + (pyConn ? ChatColor.GREEN + "CONNECTED" : ChatColor.RED + "DISCONNECTED"));

        sender.sendMessage(ChatColor.GRAY + "  Buffer: " + ChatColor.WHITE + experienceBuffer.size() + "/" + experienceBuffer.capacity());
        sender.sendMessage(ChatColor.GRAY + "  Bots: " + ChatColor.WHITE + botCmd.getBotManager().botCount());
        sender.sendMessage(ChatColor.GRAY + "  Episodes: " + ChatColor.WHITE + episodesCompleted);
        sender.sendMessage(ChatColor.GRAY + "  Avg reward: " + ChatColor.WHITE + String.format("%.2f", getAverageReward()));
        sender.sendMessage(ChatColor.GRAY + "  Default model: " + (defaultModel != null ? ChatColor.WHITE + defaultModel : ChatColor.YELLOW + "none"));
        sender.sendMessage(ChatColor.GRAY + "  Default kit: " + (defaultKit != null ? ChatColor.WHITE + defaultKit : ChatColor.YELLOW + "none"));

        if (selfPlay != null) {
            int ts = selfPlay.teamA.size();
            int deadA = (int) selfPlay.teamA.stream().filter(selfPlay.deadBots::contains).count();
            int deadB = (int) selfPlay.teamB.stream().filter(selfPlay.deadBots::contains).count();
            sender.sendMessage("");
            sender.sendMessage(ChatColor.GREEN + "  Self-Play " + ts + "v" + ts);
            sender.sendMessage(ChatColor.GRAY + "  Round: " + ChatColor.WHITE + selfPlay.roundNumber
                    + (selfPlay.maxRounds > 0 ? "/" + selfPlay.maxRounds : ""));
            sender.sendMessage(ChatColor.RED + "  Red: " + ChatColor.WHITE + (ts - deadA) + "/" + ts + " alive");
            sender.sendMessage(ChatColor.BLUE + "  Blue: " + ChatColor.WHITE + (ts - deadB) + "/" + ts + " alive");
        }

        if (duel != null) {
            sender.sendMessage("");
            sender.sendMessage(ChatColor.GREEN + "  Duel Active");
            sender.sendMessage(ChatColor.GRAY + "  Model: " + ChatColor.WHITE + duel.modelName);
            sender.sendMessage(ChatColor.GRAY + "  Bot deaths: " + ChatColor.WHITE + duel.deaths);
        }

        String lastStats = trainingClient.getLastStatsJson();
        if (lastStats != null) {
            sender.sendMessage(ChatColor.GRAY + "  Last stats: " + ChatColor.WHITE + lastStats);
        }
    }

    // ================================================================
    //  /mai stop
    // ================================================================

    // ================================================================
    //  /mai record — physics recording for sim calibration
    // ================================================================

    private @Nullable String activeRecordingBot = null;

    private void handleRecord(CommandSender sender, String[] args) {
        if (args.length == 0) {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "Usage: /mai record start [seconds] [kit] | /mai record stop");
            return;
        }
        switch (args[0].toLowerCase()) {
            case "start" -> handleRecordStart(sender, args);
            case "stop"  -> handleRecordStop(sender);
            default      -> sender.sendMessage(PREFIX + ChatColor.RED + "Unknown: /mai record " + args[0]);
        }
    }

    private void handleRecordStart(CommandSender sender, String[] args) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Only players can start recording.");
            return;
        }
        if (activeRecordingBot != null) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Recording already active for '" + activeRecordingBot + "'. Stop it first.");
            return;
        }
        if (modelManager.listModels().isEmpty()) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Load a model first (needed for action mask).");
            return;
        }

        int seconds = args.length >= 2 ? parseInt(args[1], 60) : 60;
        String kitName = args.length >= 3 ? args[2] : defaultKit;

        // Spawn the recording bot
        String model = defaultModel != null ? defaultModel : modelManager.listModels().get(0);
        BotContext ctx = botCmd.spawnBot(player.getWorld(), player.getLocation(), "Recorder", model, kitName);
        if (ctx == null) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Failed to spawn recording bot.");
            return;
        }

        // Set up physics recorder
        Path outputFile = Path.of("plugins", "MinimalAI", "recordings",
                "physics_" + System.currentTimeMillis() + ".csv");
        try {
            PhysicsRecorder recorder = new PhysicsRecorder(outputFile);
            BotBrain brain = botCmd.getBrains().get(ctx.name());
            if (brain == null) {
                sender.sendMessage(PREFIX + ChatColor.RED + "Brain not found for recording bot.");
                return;
            }
            brain.setRecordingMode(recorder);
            rewardComputer.registerBot(ctx.name());
            rewardComputer.registerRecorder(ctx.name(), recorder);
            activeRecordingBot = ctx.name();

            sender.sendMessage(PREFIX + ChatColor.GREEN + "Recording started for " + seconds + "s. Bot uses random actions.");
            sender.sendMessage(PREFIX + ChatColor.GRAY + "Output: " + outputFile);

            // Auto-stop after duration
            org.bukkit.Bukkit.getScheduler().runTaskLater(
                    org.bukkit.Bukkit.getPluginManager().getPlugin("MinimalAI"),
                    () -> {
                        if (activeRecordingBot != null && activeRecordingBot.equals(ctx.name())) {
                            handleRecordStop(sender);
                        }
                    },
                    seconds * 20L
            );

        } catch (IOException e) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Failed to create recording file: " + e.getMessage());
        }
    }

    private void handleRecordStop(CommandSender sender) {
        if (activeRecordingBot == null) {
            if (sender != null) sender.sendMessage(PREFIX + ChatColor.YELLOW + "No recording active.");
            return;
        }

        BotBrain brain = botCmd.getBrains().get(activeRecordingBot);
        String botName = activeRecordingBot;
        activeRecordingBot = null;

        if (brain != null) {
            PhysicsRecorder recorder = brain.getPhysicsRecorder();
            int ticks = recorder != null ? recorder.getTickCount() : 0;
            Path file = recorder != null ? recorder.getFilePath() : null;

            // Unregister recorder and close brain
            rewardComputer.unregisterRecorder(botName);
            botCmd.removeBrain(botName);
            botCmd.getBotManager().despawn(botName);

            if (sender != null) {
                sender.sendMessage(PREFIX + ChatColor.GREEN + "Recording stopped. " + ticks + " ticks recorded.");
                if (file != null) {
                    sender.sendMessage(PREFIX + ChatColor.GRAY + "File: " + file);
                    sender.sendMessage(PREFIX + ChatColor.GRAY + "Run: python training/calibrate_physics.py " + file);
                }
            }
        } else {
            if (sender != null) sender.sendMessage(PREFIX + ChatColor.YELLOW + "Recording bot already gone.");
        }
    }

    private static int parseInt(String s, int fallback) {
        try { return Integer.parseInt(s); } catch (NumberFormatException e) { return fallback; }
    }

    private void handleStopAll(CommandSender sender) {
        boolean acted = false;
        if (duel != null) { stopDuel(sender); acted = true; }
        if (selfPlay != null) { stopSelfPlay(sender); acted = true; }
        if (trainingEnabled) {
            trainingEnabled = false;
            for (BotBrain brain : botCmd.getBrains().values()) {
                brain.setTrainingEnabled(false);
                if (episodeManager.isActive(brain.getName()))
                    episodeManager.endEpisode(brain.getName(), EpisodeManager.EpisodeEndReason.MANUAL);
            }
            acted = true;
        }
        int bots = botCmd.getBotManager().botCount();
        if (bots > 0) {
            botCmd.closeAll();
            botCmd.getBotManager().despawnAll();
            acted = true;
        }
        sender.sendMessage(acted
                ? PREFIX + ChatColor.GREEN + "Everything stopped."
                : PREFIX + ChatColor.YELLOW + "Nothing was running.");
    }

    // ================================================================
    //  Self-play logic
    // ================================================================

    private void startSelfPlay(CommandSender sender, Location center, int teamSize,
                               String modelName, @Nullable String kitName, int maxRounds) {
        selfPlay = new SelfPlaySession(center, modelName, kitName, maxRounds);

        double[] zOff = new double[teamSize];
        for (int i = 0; i < teamSize; i++)
            zOff[i] = (i - (teamSize - 1) / 2.0) * 3;

        // Spawn Team A (Red) — west side, facing east
        for (int i = 0; i < teamSize; i++) {
            String name = "Red_" + (i + 1);
            Location loc = center.clone().add(-5, 0, zOff[i]);
            loc.setYaw(-90); loc.setPitch(0);
            BotContext ctx = botCmd.spawnBot(center.getWorld(), loc, name, modelName, kitName);
            if (ctx == null) {
                sender.sendMessage(PREFIX + ChatColor.RED + "Failed to spawn " + name);
                stopSelfPlay(sender); return;
            }
            selfPlay.teamA.add(name);
        }

        // Spawn Team B (Blue) — east side, facing west
        for (int i = 0; i < teamSize; i++) {
            String name = "Blue_" + (i + 1);
            Location loc = center.clone().add(5, 0, zOff[i]);
            loc.setYaw(90); loc.setPitch(0);
            BotContext ctx = botCmd.spawnBot(center.getWorld(), loc, name, modelName, kitName);
            if (ctx == null) {
                sender.sendMessage(PREFIX + ChatColor.RED + "Failed to spawn " + name);
                stopSelfPlay(sender); return;
            }
            selfPlay.teamB.add(name);
        }

        setupAllies();

        // Enable training + start episodes
        trainingEnabled = true;
        if (!trainingClient.isConnected()) trainingClient.connect();
        for (String name : selfPlay.allBots()) {
            BotBrain brain = botCmd.getBrains().get(name);
            if (brain != null) {
                brain.setTrainingEnabled(true);
                episodeManager.startEpisode(name, SELFPLAY_EPISODE_TICKS);
            }
        }

        selfPlay.roundNumber = 1;
        String timesStr = maxRounds > 0 ? " for " + maxRounds + " rounds" : "";
        sender.sendMessage(PREFIX + ChatColor.GREEN + teamSize + "v" + teamSize + " self-play started"
                + timesStr + " — model '" + modelName + "'"
                + (kitName != null ? " kit '" + kitName + "'" : ""));
    }

    private void setupAllies() {
        if (selfPlay == null) return;
        Set<UUID> aUUIDs = new HashSet<>(), bUUIDs = new HashSet<>();
        for (String n : selfPlay.teamA) { BotBrain b = botCmd.getBrains().get(n); if (b != null) aUUIDs.add(b.getServerPlayer().getUUID()); }
        for (String n : selfPlay.teamB) { BotBrain b = botCmd.getBrains().get(n); if (b != null) bUUIDs.add(b.getServerPlayer().getUUID()); }
        for (String n : selfPlay.teamA) { BotBrain b = botCmd.getBrains().get(n); if (b != null) b.setAllies(aUUIDs); }
        for (String n : selfPlay.teamB) { BotBrain b = botCmd.getBrains().get(n); if (b != null) b.setAllies(bUUIDs); }
    }

    private void resetRound() {
        if (selfPlay == null) return;

        // Auto-stop if max rounds reached
        if (selfPlay.maxRounds > 0 && selfPlay.roundNumber >= selfPlay.maxRounds) {
            LOG.info("[SelfPlay] Completed " + selfPlay.roundNumber + " rounds — auto-stopping");
            stopSelfPlay(null);
            return;
        }

        // End all episodes
        for (String name : selfPlay.allBots())
            if (episodeManager.isActive(name))
                episodeManager.endEpisode(name, EpisodeManager.EpisodeEndReason.MANUAL);

        selfPlay.deadBots.clear();
        Location center = selfPlay.spawnCenter;
        int ts = selfPlay.teamA.size();
        double[] zOff = new double[ts];
        for (int i = 0; i < ts; i++) zOff[i] = (i - (ts - 1) / 2.0) * 3;

        for (int i = 0; i < ts; i++)
            resetBot(selfPlay.teamA.get(i), center.getX() - 5, center.getY(), center.getZ() + zOff[i], -90);
        for (int i = 0; i < ts; i++)
            resetBot(selfPlay.teamB.get(i), center.getX() + 5, center.getY(), center.getZ() + zOff[i], 90);

        setupAllies();
        for (String name : selfPlay.allBots()) {
            BotBrain brain = botCmd.getBrains().get(name);
            if (brain != null) {
                brain.setTrainingEnabled(true);
                episodeManager.startEpisode(name, SELFPLAY_EPISODE_TICKS);
            }
        }
        selfPlay.roundNumber++;
        LOG.info("[SelfPlay] Round " + selfPlay.roundNumber
                + (selfPlay.maxRounds > 0 ? "/" + selfPlay.maxRounds : "") + " started");
    }

    private void resetBot(String name, double x, double y, double z, float yaw) {
        BotBrain brain = botCmd.getBrains().get(name);
        if (brain == null) return;
        ServerPlayer sp = brain.getServerPlayer();
        sp.setHealth(sp.getMaxHealth());
        sp.getBukkitEntity().getActivePotionEffects().forEach(
                eff -> sp.getBukkitEntity().removePotionEffect(eff.getType()));
        sp.setDeltaMovement(0, 0, 0);
        sp.snapTo(x, y, z, yaw, 0);

        if (selfPlay != null && selfPlay.kitName != null)
            kitManager.applyKit(sp, selfPlay.kitName);
        brain.setPaused(false);
        brain.resetEpisode();
    }

    private void stopSelfPlay(@Nullable CommandSender sender) {
        if (selfPlay == null) return;
        for (String name : selfPlay.allBots()) {
            if (episodeManager.isActive(name))
                episodeManager.endEpisode(name, EpisodeManager.EpisodeEndReason.MANUAL);
            botCmd.removeBrain(name);
            botCmd.getBotManager().despawn(name);
        }
        trainingEnabled = false;
        int rounds = selfPlay.roundNumber;
        selfPlay = null;
        if (sender != null)
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Self-play stopped. " + rounds + " round(s) completed.");
        else
            LOG.info("[SelfPlay] Auto-stopped after " + rounds + " rounds");
    }

    // ================================================================
    //  Duel logic
    // ================================================================

    private void stopDuel(@Nullable CommandSender sender) {
        if (duel == null) return;
        if (episodeManager.isActive(duel.botName))
            episodeManager.endEpisode(duel.botName, EpisodeManager.EpisodeEndReason.MANUAL);
        botCmd.removeBrain(duel.botName);
        botCmd.getBotManager().despawn(duel.botName);
        trainingEnabled = false;
        int deaths = duel.deaths;
        duel = null;
        if (sender != null)
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Duel ended. Bot died " + deaths + " time(s).");
    }

    private void resetDuelBot() {
        if (duel == null) return;
        BotBrain brain = botCmd.getBrains().get(duel.botName);
        if (brain == null) return;
        ServerPlayer sp = brain.getServerPlayer();

        if (episodeManager.isActive(duel.botName))
            episodeManager.endEpisode(duel.botName, EpisodeManager.EpisodeEndReason.DEATH);

        sp.setHealth(sp.getMaxHealth());
        sp.getBukkitEntity().getActivePotionEffects().forEach(
                eff -> sp.getBukkitEntity().removePotionEffect(eff.getType()));
        sp.setDeltaMovement(0, 0, 0);
        sp.snapTo(duel.spawnLoc.getX(), duel.spawnLoc.getY(), duel.spawnLoc.getZ(), duel.spawnLoc.getYaw(), 0);
        if (duel.kitName != null) kitManager.applyKit(sp, duel.kitName);

        brain.resetEpisode();
        brain.setTrainingEnabled(true);
        episodeManager.startEpisode(duel.botName, SELFPLAY_EPISODE_TICKS);
        duel.deaths++;
    }

    // ================================================================
    //  Event handlers
    // ================================================================

    @EventHandler(priority = EventPriority.HIGHEST)
    public void onDamage(EntityDamageByEntityEvent event) {
        if (!(event.getEntity() instanceof Player victim)) return;
        if (!(event.getDamager() instanceof Player attacker)) return;

        String vName = victim.getName();
        String aName = attacker.getName();

        // --- Self-play ---
        if (selfPlay != null && selfPlay.isBot(vName)) {
            String vTeam = selfPlay.getTeam(vName);
            String aTeam = selfPlay.getTeam(aName);

            // Friendly fire
            if (vTeam != null && vTeam.equals(aTeam)) { event.setCancelled(true); return; }

            // Already dead
            if (selfPlay.deadBots.contains(vName)) { event.setCancelled(true); return; }

            // Lethal check
            if (victim.getHealth() - event.getFinalDamage() <= 0) {
                event.setCancelled(true);
                if (selfPlay.isBot(aName)) {
                    rewardComputer.addReward(aName, 5.0f);
                    episodeManager.onKill(aName);
                }
                rewardComputer.addReward(vName, -3.0f);
                episodeManager.onDeath(vName);

                BotBrain vBrain = botCmd.getBrains().get(vName);
                if (vBrain != null) {
                    vBrain.setPaused(true);
                    ServerPlayer sp = vBrain.getServerPlayer();
                    sp.snapTo(sp.getX(), sp.getY() + 100, sp.getZ(), sp.getYRot(), sp.getXRot());
                }
                selfPlay.deadBots.add(vName);
                if (vTeam != null && selfPlay.isTeamWiped(vTeam)) resetRound();
            }
            return;
        }

        // --- Duel ---
        if (duel != null && vName.equals(duel.botName)) {
            if (victim.getHealth() - event.getFinalDamage() <= 0) {
                event.setCancelled(true);
                rewardComputer.addReward(vName, -3.0f);
                resetDuelBot();
            }
        }
    }

    @EventHandler
    public void onPlayerDeath(PlayerDeathEvent event) {
        if (duel != null && event.getEntity().getUniqueId().equals(duel.playerUUID)) {
            rewardComputer.addReward(duel.botName, 5.0f);
            episodeManager.onKill(duel.botName);
            resetDuelBot();
        }
    }

    // ================================================================
    //  Tab completion
    // ================================================================

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        if (args.length == 1) return filter(TOP_COMMANDS, args[0]);
        String[] rest = drop(args);
        return switch (args[0].toLowerCase()) {
            case "model" -> tabModel(rest);
            case "train" -> tabTrain(rest);
            case "bot"   -> tabBot(rest);
            case "kit"   -> tabKit(rest);
            default      -> List.of();
        };
    }

    private List<String> tabModel(String[] a) {
        if (a.length == 1) return filter(MODEL_SUBS, a[0]);
        String sub = a[0].toLowerCase();
        if (a.length == 2 && Set.of("delete", "load", "duel", "default").contains(sub))
            return filter(modelManager.listModels(), a[1]);
        if (a.length == 3 && "duel".equals(sub))
            return filter(kitManager.listKits(), a[2]);
        return List.of();
    }

    private List<String> tabTrain(String[] a) {
        if (a.length == 1) {
            List<String> s = new ArrayList<>(List.of("group", "stop"));
            s.addAll(modelManager.listModels());
            return filter(s, a[0]);
        }
        if ("group".equalsIgnoreCase(a[0])) {
            if (a.length == 2) return filter(TEAM_SIZES, a[1]);
            if (a.length == 3) return filter(modelManager.listModels(), a[2]);
            if (a.length == 4) return filter(kitManager.listKits(), a[3]);
        } else {
            if (a.length == 2) return filter(kitManager.listKits(), a[1]);
        }
        return List.of();
    }

    private List<String> tabBot(String[] a) {
        if (a.length == 1) return filter(BOT_SUBS, a[0]);
        String sub = a[0].toLowerCase();
        if ("despawn".equals(sub) && a.length == 2) {
            List<String> n = botCmd.getBotManager().getAllBots().stream()
                    .map(BotContext::name).collect(Collectors.toList());
            n.add("*");
            return filter(n, a[1]);
        }
        if ("spawn".equals(sub)) {
            if (a.length == 3) return filter(modelManager.listModels(), a[2]);
            if (a.length == 4) return filter(kitManager.listKits(), a[3]);
        }
        return List.of();
    }

    private List<String> tabKit(String[] a) {
        if (a.length == 1) return filter(KIT_SUBS, a[0]);
        String sub = a[0].toLowerCase();
        if (a.length == 2 && ("delete".equals(sub) || "default".equals(sub))) {
            List<String> opts = new ArrayList<>(kitManager.listKits());
            if ("default".equals(sub)) opts.add("none");
            return filter(opts, a[1]);
        }
        return List.of();
    }

    // ================================================================
    //  Help messages
    // ================================================================

    private void sendHelp(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.GREEN + "--- MinimalAI Commands ---");
        sender.sendMessage(ChatColor.AQUA + "  /mai model " + ChatColor.GRAY + "create | delete | list | load | duel | default");
        sender.sendMessage(ChatColor.AQUA + "  /mai train " + ChatColor.GRAY + "[model] [kit] [times] | group <size> | stop");
        sender.sendMessage(ChatColor.AQUA + "  /mai bot   " + ChatColor.GRAY + "spawn | despawn | list");
        sender.sendMessage(ChatColor.AQUA + "  /mai kit   " + ChatColor.GRAY + "save | delete | list | default");
        sender.sendMessage(ChatColor.AQUA + "  /mai status");
        sender.sendMessage(ChatColor.AQUA + "  /mai stop");
    }

    private void sendModelHelp(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.GREEN + "/mai model");
        sender.sendMessage(ChatColor.GRAY + "  create <name>" + ChatColor.WHITE + " — Create a fresh model");
        sender.sendMessage(ChatColor.GRAY + "  delete <name>" + ChatColor.WHITE + " — Remove a model");
        sender.sendMessage(ChatColor.GRAY + "  list" + ChatColor.WHITE + " — List models");
        sender.sendMessage(ChatColor.GRAY + "  load <name>" + ChatColor.WHITE + " — Reload model from disk");
        sender.sendMessage(ChatColor.GRAY + "  duel [model] [kit]" + ChatColor.WHITE + " — Fight a bot (it learns)");
        sender.sendMessage(ChatColor.GRAY + "  default [name]" + ChatColor.WHITE + " — Set/show default model");
    }

    private void sendTrainHelp(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.GREEN + "/mai train");
        sender.sendMessage(ChatColor.GRAY + "  [model] [kit] [times]" + ChatColor.WHITE + " — 1v1 self-play");
        sender.sendMessage(ChatColor.GRAY + "  group <size> [model] [kit] [times]" + ChatColor.WHITE + " — Team self-play");
        sender.sendMessage(ChatColor.GRAY + "  stop" + ChatColor.WHITE + " — Stop training");
        sender.sendMessage(ChatColor.GRAY + "  Sizes: 2v2, 3v3, 5v5. Omit times for infinite.");
    }

    private void sendBotHelp(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.GREEN + "/mai bot");
        sender.sendMessage(ChatColor.GRAY + "  spawn [name] [model] [kit]" + ChatColor.WHITE + " — Spawn a bot");
        sender.sendMessage(ChatColor.GRAY + "  despawn <name|*>" + ChatColor.WHITE + " — Remove bot(s)");
        sender.sendMessage(ChatColor.GRAY + "  list" + ChatColor.WHITE + " — List active bots");
    }

    private void sendKitHelp(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.GREEN + "/mai kit");
        sender.sendMessage(ChatColor.GRAY + "  save <name>" + ChatColor.WHITE + " — Save your inventory");
        sender.sendMessage(ChatColor.GRAY + "  delete <name>" + ChatColor.WHITE + " — Delete a kit");
        sender.sendMessage(ChatColor.GRAY + "  list" + ChatColor.WHITE + " — List kits");
        sender.sendMessage(ChatColor.GRAY + "  default [name]" + ChatColor.WHITE + " — Set/show default kit");
    }

    // ================================================================
    //  State accessors (called by MinimalAIPlugin)
    // ================================================================

    public boolean isTrainingEnabled() { return trainingEnabled; }
    public boolean isSelfPlayActive() { return selfPlay != null; }

    public void recordEpisodeEnd(float totalReward) {
        episodesCompleted++;
        recentRewards[rewardWriteIndex] = totalReward;
        rewardWriteIndex = (rewardWriteIndex + 1) % recentRewards.length;
        if (rewardCount < recentRewards.length) rewardCount++;
    }

    public void onSelfPlayEpisodeTimeout(String botName) {
        if (selfPlay != null && selfPlay.isBot(botName)) {
            LOG.info("[SelfPlay] Episode timeout — resetting round (draw)");
            resetRound();
        }
    }

    // ================================================================
    //  Internal helpers
    // ================================================================

    private float getAverageReward() {
        if (rewardCount == 0) return 0f;
        float sum = 0f;
        for (int i = 0; i < rewardCount; i++) sum += recentRewards[i];
        return sum / rewardCount;
    }

    private @Nullable String resolveModel(@Nullable String explicit) {
        if (explicit != null) return explicit;
        if (defaultModel != null) return defaultModel;
        List<String> m = modelManager.listModels();
        return m.isEmpty() ? null : m.get(0);
    }

    private @Nullable String resolveKit(@Nullable String explicit) {
        if (explicit != null && kitManager.listKits().contains(explicit)) return explicit;
        return defaultKit;
    }

    private static String[] drop(String[] a) {
        return a.length <= 1 ? new String[0] : Arrays.copyOfRange(a, 1, a.length);
    }

    private static boolean isNumber(String s) {
        try { Integer.parseInt(s); return true; }
        catch (NumberFormatException e) { return false; }
    }

    private static List<String> filter(List<String> options, String prefix) {
        String lower = prefix.toLowerCase();
        return options.stream()
                .filter(s -> s.toLowerCase().startsWith(lower))
                .collect(Collectors.toCollection(ArrayList::new));
    }
}
