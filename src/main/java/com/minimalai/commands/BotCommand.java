package com.minimalai.commands;

import com.minimalai.ai.*;
import com.minimalai.bot.BotBrain;
import com.minimalai.bot.FakePlayerManager;
import com.minimalai.bot.FakePlayerManager.BotContext;
import com.minimalai.integration.ArcaneSigilsAPI;
import com.minimalai.training.ExperienceBuffer;
import com.minimalai.training.RewardComputer;
import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabCompleter;
import org.bukkit.entity.Player;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Command executor for {@code /bot spawn|despawn|list}.
 * Manages BotBrain lifecycle alongside FakePlayerManager.
 */
public class BotCommand implements CommandExecutor, TabCompleter {

    private static final String PREFIX = ChatColor.GRAY + "[" + ChatColor.AQUA + "MinimalAI" + ChatColor.GRAY + "] " + ChatColor.RESET;
    private static final List<String> SUB_COMMANDS = Arrays.asList("spawn", "despawn", "list");

    private final FakePlayerManager botManager;
    private final ModelManager modelManager;
    private final ObservationBuilder obsBuilder;
    private final ActionExecutor actionExecutor;
    private final @Nullable ArcaneSigilsAPI sigilsApi;
    private final @Nullable ExperienceBuffer experienceBuffer;
    private final @Nullable RewardComputer rewardComputer;

    // BotBrain instances managed here, ticked by MinimalAIPlugin
    private final Map<String, BotBrain> brains = new ConcurrentHashMap<>();

    public BotCommand(FakePlayerManager botManager,
                      ModelManager modelManager,
                      ObservationBuilder obsBuilder,
                      ActionExecutor actionExecutor,
                      @Nullable ArcaneSigilsAPI sigilsApi,
                      @Nullable ExperienceBuffer experienceBuffer,
                      @Nullable RewardComputer rewardComputer) {
        this.botManager = botManager;
        this.modelManager = modelManager;
        this.obsBuilder = obsBuilder;
        this.actionExecutor = actionExecutor;
        this.sigilsApi = sigilsApi;
        this.experienceBuffer = experienceBuffer;
        this.rewardComputer = rewardComputer;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 0) {
            sendUsage(sender);
            return true;
        }

        switch (args[0].toLowerCase()) {
            case "spawn" -> handleSpawn(sender, args);
            case "despawn" -> handleDespawn(sender, args);
            case "list" -> handleList(sender);
            default -> sendUsage(sender);
        }
        return true;
    }

    private void handleSpawn(CommandSender sender, String[] args) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Only players can spawn bots.");
            return;
        }

        // Check if any model is loaded
        if (modelManager.listModels().isEmpty()) {
            sender.sendMessage(PREFIX + ChatColor.RED + "No models loaded. Place a .pt file in plugins/MinimalAI/models/ and run /maimodel load <name>");
            return;
        }

        String name = args.length >= 2 ? args[1] : null;
        Location loc = player.getLocation();

        BotContext ctx = botManager.spawn(player.getWorld(), loc, name);
        if (ctx == null) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Failed to spawn bot. Check console.");
            return;
        }

        // Create BotBrain
        try {
            String modelName = modelManager.listModels().get(0); // use first loaded model
            BotBrain brain = new BotBrain(ctx.name(), ctx.serverPlayer(), modelManager, modelName,
                    obsBuilder, actionExecutor, sigilsApi);

            if (experienceBuffer != null && rewardComputer != null) {
                brain.setTrainingComponents(experienceBuffer, rewardComputer);
                rewardComputer.registerBot(ctx.name());
            }

            brains.put(ctx.name(), brain);
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Spawned bot '" + ctx.name() + "' with model '" + modelName + "'.");
        } catch (Exception e) {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "Bot spawned but brain failed: " + e.getMessage());
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "Bot will be idle until a model is loaded.");
        }
    }

    private void handleDespawn(CommandSender sender, String[] args) {
        if (args.length < 2) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Usage: /bot despawn <name|*>");
            return;
        }

        String target = args[1];

        if ("*".equals(target)) {
            int count = botManager.botCount();
            if (count == 0) {
                sender.sendMessage(PREFIX + ChatColor.YELLOW + "No bots to remove.");
                return;
            }
            for (BotBrain brain : brains.values()) {
                if (rewardComputer != null) rewardComputer.unregisterBot(brain.getName());
                brain.close();
            }
            brains.clear();
            botManager.despawnAll();
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Removed all " + count + " bot(s).");
        } else {
            BotBrain brain = brains.remove(target);
            if (brain != null) {
                if (rewardComputer != null) rewardComputer.unregisterBot(target);
                brain.close();
            }
            boolean removed = botManager.despawn(target);
            if (removed) {
                sender.sendMessage(PREFIX + ChatColor.GREEN + "Removed bot '" + target + "'.");
            } else {
                sender.sendMessage(PREFIX + ChatColor.RED + "No bot named '" + target + "' found.");
            }
        }
    }

    private void handleList(CommandSender sender) {
        Collection<BotContext> bots = botManager.getAllBots();
        if (bots.isEmpty()) {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "No active bots.");
            return;
        }

        sender.sendMessage(PREFIX + ChatColor.GREEN + "Active bots (" + bots.size() + "):");
        for (BotContext ctx : bots) {
            Location loc = ctx.bukkitLocation();
            boolean hasBrain = brains.containsKey(ctx.name());
            sender.sendMessage(ChatColor.GRAY + " - " + ChatColor.WHITE + ctx.name()
                    + (hasBrain ? ChatColor.GREEN + " [AI]" : ChatColor.RED + " [idle]")
                    + ChatColor.GRAY + " at "
                    + String.format("%.1f, %.1f, %.1f", loc.getX(), loc.getY(), loc.getZ())
                    + " in " + (loc.getWorld() != null ? loc.getWorld().getName() : "?"));
        }
    }

    private void sendUsage(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.YELLOW + "Usage:");
        sender.sendMessage(ChatColor.GRAY + "  /bot spawn [name]" + ChatColor.WHITE + " - Spawn AI bot");
        sender.sendMessage(ChatColor.GRAY + "  /bot despawn <name|*>" + ChatColor.WHITE + " - Remove bot(s)");
        sender.sendMessage(ChatColor.GRAY + "  /bot list" + ChatColor.WHITE + " - List active bots");
    }

    /** Called by MinimalAIPlugin tick loop. */
    public void tickAll() {
        for (BotBrain brain : brains.values()) {
            brain.tick();
        }
    }

    /** Called by MinimalAIPlugin on disable. */
    public void closeAll() {
        for (BotBrain brain : brains.values()) {
            if (rewardComputer != null) rewardComputer.unregisterBot(brain.getName());
            brain.close();
        }
        brains.clear();
    }

    public Map<String, BotBrain> getBrains() {
        return Collections.unmodifiableMap(brains);
    }

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        if (args.length == 1) {
            return filterStartsWith(SUB_COMMANDS, args[0]);
        }
        if (args.length == 2 && "despawn".equalsIgnoreCase(args[0])) {
            List<String> names = botManager.getAllBots().stream()
                    .map(BotContext::name)
                    .collect(Collectors.toList());
            names.add("*");
            return filterStartsWith(names, args[1]);
        }
        return List.of();
    }

    private static List<String> filterStartsWith(List<String> options, String prefix) {
        String lower = prefix.toLowerCase();
        return options.stream()
                .filter(s -> s.toLowerCase().startsWith(lower))
                .collect(Collectors.toCollection(ArrayList::new));
    }
}
