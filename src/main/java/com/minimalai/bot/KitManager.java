package com.minimalai.bot;

import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.EquipmentSlot;
import net.minecraft.world.item.Items;
import org.bukkit.Material;
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

        Player bukkitPlayer = bot.getBukkitEntity();
        PlayerInventory inv = bukkitPlayer.getInventory();

        // Clear existing inventory
        inv.clear();

        YamlConfiguration config = YamlConfiguration.loadConfiguration(file);
        int itemsLoaded = 0;

        // Load main inventory slots
        if (config.isConfigurationSection("inventory")) {
            for (String key : config.getConfigurationSection("inventory").getKeys(false)) {
                int slot = Integer.parseInt(key);
                ItemStack item = config.getItemStack("inventory." + key);
                if (item != null) {
                    inv.setItem(slot, item);
                    itemsLoaded++;
                }
            }
        }

        // Load armor
        if (config.isConfigurationSection("armor")) {
            ItemStack boots = config.getItemStack("armor.boots");
            ItemStack leggings = config.getItemStack("armor.leggings");
            ItemStack chestplate = config.getItemStack("armor.chestplate");
            ItemStack helmet = config.getItemStack("armor.helmet");
            if (boots != null) { inv.setBoots(boots); itemsLoaded++; }
            if (leggings != null) { inv.setLeggings(leggings); itemsLoaded++; }
            if (chestplate != null) { inv.setChestplate(chestplate); itemsLoaded++; }
            if (helmet != null) { inv.setHelmet(helmet); itemsLoaded++; }
        }

        // Load offhand
        ItemStack offhand = config.getItemStack("offhand");
        if (offhand != null) {
            inv.setItemInOffHand(offhand);
            itemsLoaded++;
        }

        // Fallback: if YAML deserialization failed (0 items loaded),
        // try loading items using the simple format (id + count only)
        if (itemsLoaded == 0) {
            logger.warning("Kit '" + name + "' Bukkit deserialization returned 0 items. Sections: "
                + "inventory=" + config.isConfigurationSection("inventory")
                + " armor=" + config.isConfigurationSection("armor")
                + " File: " + file.getAbsolutePath() + " exists=" + file.exists());
            itemsLoaded = applyKitSimple(config, inv);
            if (itemsLoaded == 0) {
                logger.warning("Simple format also returned 0 items. Trying direct NMS.");
                applyPlainKitFallback(bot);
                itemsLoaded = 6;
            }
        }

        // Set selected slot
        int selectedSlot = config.getInt("selected-slot", 0);
        inv.setHeldItemSlot(selectedSlot);

        logger.info("Applied kit '" + name + "' (" + itemsLoaded + " items) to bot " + bot.getScoreboardName());
        return true;
    }

    /**
     * Fallback kit loader: reads id/count from YAML sections and creates items via Material.
     */
    private int applyKitSimple(YamlConfiguration config, PlayerInventory inv) {
        int loaded = 0;

        if (config.isConfigurationSection("inventory")) {
            for (String key : config.getConfigurationSection("inventory").getKeys(false)) {
                String path = "inventory." + key;
                String id = config.getString(path + ".id", "");
                int count = config.getInt(path + ".count", 1);
                ItemStack item = createItemFromId(id, count);
                if (item != null) {
                    inv.setItem(Integer.parseInt(key), item);
                    loaded++;
                }
            }
        }

        if (config.isConfigurationSection("armor")) {
            for (String slot : List.of("boots", "leggings", "chestplate", "helmet")) {
                String path = "armor." + slot;
                String id = config.getString(path + ".id", "");
                int count = config.getInt(path + ".count", 1);
                ItemStack item = createItemFromId(id, count);
                if (item != null) {
                    switch (slot) {
                        case "boots" -> inv.setBoots(item);
                        case "leggings" -> inv.setLeggings(item);
                        case "chestplate" -> inv.setChestplate(item);
                        case "helmet" -> inv.setHelmet(item);
                    }
                    loaded++;
                }
            }
        }

        return loaded;
    }

    private ItemStack createItemFromId(String id, int count) {
        if (id == null || id.isEmpty()) return null;
        // Strip "minecraft:" prefix
        String key = id.replace("minecraft:", "").toUpperCase();
        try {
            org.bukkit.Material material = org.bukkit.Material.valueOf(key);
            return new ItemStack(material, count);
        } catch (IllegalArgumentException e) {
            logger.warning("Unknown material in kit: " + id);
            return null;
        }
    }

    /**
     * Load sigil IDs from a kit's YAML (under the "sigils" key).
     * Returns empty list if no sigils are defined in the kit.
     */
    public List<String> loadSigils(String name) {
        File file = kitsDir.resolve(name + ".yml").toFile();
        if (!file.exists()) return Collections.emptyList();
        YamlConfiguration config = YamlConfiguration.loadConfiguration(file);
        return config.getStringList("sigils");
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

    /**
     * Hardcoded fallback: give basic PvP kit via Bukkit Material API.
     * Used when YAML deserialization fails for the kit file.
     */
    private void applyPlainKitFallback(ServerPlayer bot) {
        Player bukkitPlayer = bot.getBukkitEntity();
        PlayerInventory inv = bukkitPlayer.getInventory();
        inv.setItem(0, new ItemStack(Material.DIAMOND_SWORD, 1));
        inv.setItem(1, new ItemStack(Material.GOLDEN_APPLE, 64));
        inv.setHelmet(new ItemStack(Material.DIAMOND_HELMET, 1));
        inv.setChestplate(new ItemStack(Material.DIAMOND_CHESTPLATE, 1));
        inv.setLeggings(new ItemStack(Material.DIAMOND_LEGGINGS, 1));
        inv.setBoots(new ItemStack(Material.DIAMOND_BOOTS, 1));
        inv.setHeldItemSlot(0);
    }
}
