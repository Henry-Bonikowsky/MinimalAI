package com.minimalai.commands;

import com.minimalai.ai.ModelManager;
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
 * Command executor for {@code /maimodel load|list}.
 *
 * <p>Registered in plugin.yml under "maimodel" with permission minimalai.model.
 */
public class ModelCommand implements CommandExecutor, TabCompleter {

    private static final String PREFIX = ChatColor.GRAY + "["
            + ChatColor.AQUA + "MinimalAI"
            + ChatColor.GRAY + "] " + ChatColor.RESET;

    private static final List<String> SUB_COMMANDS = Arrays.asList("list", "load");

    private final ModelManager modelManager;

    public ModelCommand(ModelManager modelManager) {
        this.modelManager = modelManager;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 0) {
            sendUsage(sender);
            return true;
        }

        String sub = args[0].toLowerCase();

        switch (sub) {
            case "list" -> handleList(sender);
            case "load" -> handleLoad(sender, args);
            default -> sendUsage(sender);
        }
        return true;
    }

    // ---------------------------------------------------------------
    //  /maimodel list
    // ---------------------------------------------------------------
    private void handleList(CommandSender sender) {
        List<String> models = modelManager.listModels();

        if (models.isEmpty()) {
            sender.sendMessage(PREFIX + ChatColor.YELLOW + "No models found in models directory.");
            return;
        }

        sender.sendMessage(PREFIX + ChatColor.GREEN + "Available models (" + models.size() + "):");
        for (String name : models) {
            sender.sendMessage(ChatColor.GRAY + "  - " + ChatColor.WHITE + name);
        }
    }

    // ---------------------------------------------------------------
    //  /maimodel load <name>
    // ---------------------------------------------------------------
    private void handleLoad(CommandSender sender, String[] args) {
        if (args.length < 2) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Usage: /maimodel load <name>");
            return;
        }

        String name = args[1];

        try {
            modelManager.loadModel(name);
            sender.sendMessage(PREFIX + ChatColor.GREEN + "Model '" + name + "' loaded successfully.");
        } catch (Exception e) {
            sender.sendMessage(PREFIX + ChatColor.RED + "Failed to load model '" + name + "': " + e.getMessage());
        }
    }

    // ---------------------------------------------------------------
    //  Usage
    // ---------------------------------------------------------------
    private void sendUsage(CommandSender sender) {
        sender.sendMessage(PREFIX + ChatColor.YELLOW + "Usage:");
        sender.sendMessage(ChatColor.GRAY + "  /maimodel list"
                + ChatColor.WHITE + " - List available models");
        sender.sendMessage(ChatColor.GRAY + "  /maimodel load <name>"
                + ChatColor.WHITE + " - Load a model by name (without .pt)");
    }

    // ---------------------------------------------------------------
    //  Tab completion
    // ---------------------------------------------------------------
    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        if (args.length == 1) {
            return filterStartsWith(SUB_COMMANDS, args[0]);
        }

        if (args.length == 2 && "load".equalsIgnoreCase(args[0])) {
            return filterStartsWith(modelManager.listModels(), args[1]);
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
