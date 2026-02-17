package com.minimalai.bot;

import net.minecraft.server.level.ServerPlayer;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.PlayerInventory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Saves and loads player inventory kits as YAML files.
 * Kits are stored in plugins/MinimalAI/kits/ and can be applied to bots on spawn.
 */
public class KitManager {

    private final Path kitsDir;
    private final Logger logger;

    public KitManager(Path kitsDir, Logger logger) {
        this.kitsDir = kitsDir;
        this.logger = logger;
        kitsDir.toFile().mkdirs();
    }

    /**
     * Save a player's full inventory (hotbar, inventory, armor, offhand) to a YAML file.
     */
    public void saveKit(Player player, String name) throws IOException {
        YamlConfiguration config = new YamlConfiguration();
        PlayerInventory inv = player.getInventory();

        // Save all 36 main inventory slots (0-35: hotbar 0-8, then inventory 9-35)
        for (int i = 0; i < 36; i++) {
            ItemStack item = inv.getItem(i);
            if (item != null && !item.getType().isAir()) {
                config.set("inventory." + i, item);
            }
        }

        // Save armor (helmet, chestplate, leggings, boots)
        ItemStack[] armor = inv.getArmorContents();
        String[] armorSlots = {"boots", "leggings", "chestplate", "helmet"};
        for (int i = 0; i < armor.length; i++) {
            if (armor[i] != null && !armor[i].getType().isAir()) {
                config.set("armor." + armorSlots[i], armor[i]);
            }
        }

        // Save offhand
        ItemStack offhand = inv.getItemInOffHand();
        if (!offhand.getType().isAir()) {
            config.set("offhand", offhand);
        }

        // Save selected hotbar slot
        config.set("selected-slot", inv.getHeldItemSlot());

        File file = kitsDir.resolve(name + ".yml").toFile();
        config.save(file);
        logger.info("Saved kit '" + name + "' from player " + player.getName());
    }

    /**
     * Apply a saved kit to a bot's inventory.
     */
    public boolean applyKit(ServerPlayer bot, String name) {
        File file = kitsDir.resolve(name + ".yml").toFile();
        if (!file.exists()) {
            logger.warning("Kit not found: " + name);
            return false;
        }

        YamlConfiguration config = YamlConfiguration.loadConfiguration(file);
        Player bukkitPlayer = bot.getBukkitEntity();
        PlayerInventory inv = bukkitPlayer.getInventory();

        // Clear existing inventory
        inv.clear();

        // Load main inventory slots
        if (config.isConfigurationSection("inventory")) {
            for (String key : config.getConfigurationSection("inventory").getKeys(false)) {
                int slot = Integer.parseInt(key);
                ItemStack item = config.getItemStack("inventory." + key);
                if (item != null) {
                    inv.setItem(slot, item);
                }
            }
        }

        // Load armor
        if (config.isConfigurationSection("armor")) {
            ItemStack boots = config.getItemStack("armor.boots");
            ItemStack leggings = config.getItemStack("armor.leggings");
            ItemStack chestplate = config.getItemStack("armor.chestplate");
            ItemStack helmet = config.getItemStack("armor.helmet");
            if (boots != null) inv.setBoots(boots);
            if (leggings != null) inv.setLeggings(leggings);
            if (chestplate != null) inv.setChestplate(chestplate);
            if (helmet != null) inv.setHelmet(helmet);
        }

        // Load offhand
        ItemStack offhand = config.getItemStack("offhand");
        if (offhand != null) {
            inv.setItemInOffHand(offhand);
        }

        // Set selected slot
        int selectedSlot = config.getInt("selected-slot", 0);
        inv.setHeldItemSlot(selectedSlot);

        logger.info("Applied kit '" + name + "' to bot " + bot.getScoreboardName());
        return true;
    }

    /**
     * List all available kit names.
     */
    public List<String> listKits() {
        File dir = kitsDir.toFile();
        if (!dir.isDirectory()) return Collections.emptyList();

        String[] files = dir.list((d, n) -> n.endsWith(".yml"));
        if (files == null) return Collections.emptyList();

        return Stream.of(files)
                .map(f -> f.substring(0, f.length() - 4))
                .sorted()
                .collect(Collectors.toList());
    }

    /**
     * Delete a saved kit.
     */
    public boolean deleteKit(String name) {
        File file = kitsDir.resolve(name + ".yml").toFile();
        if (!file.exists()) return false;
        return file.delete();
    }
}
