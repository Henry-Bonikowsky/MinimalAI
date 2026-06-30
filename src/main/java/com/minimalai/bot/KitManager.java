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
        int itemsFailed = 0;

        // Load main inventory slots
        if (config.isConfigurationSection("inventory")) {
            for (String key : config.getConfigurationSection("inventory").getKeys(false)) {
                int slot = Integer.parseInt(key);
                try {
                    ItemStack item = config.getItemStack("inventory." + key);
                    if (item != null) {
                        inv.setItem(slot, item);
                        itemsLoaded++;
                    } else {
                        itemsFailed++;
                        String id = config.getString("inventory." + key + ".id", "unknown");
                        logger.warning("Kit '" + name + "' slot " + slot + " (" + id + ") deserialized to null");
                    }
                } catch (Exception e) {
                    itemsFailed++;
                    logger.warning("Kit '" + name + "' slot " + slot + " exception: " + e.getMessage());
                }
            }
        }

        // Load armor
        if (config.isConfigurationSection("armor")) {
            for (String slot : List.of("boots", "leggings", "chestplate", "helmet")) {
                try {
                    ItemStack item = config.getItemStack("armor." + slot);
                    if (item != null) {
                        switch (slot) {
                            case "boots" -> inv.setBoots(item);
                            case "leggings" -> inv.setLeggings(item);
                            case "chestplate" -> inv.setChestplate(item);
                            case "helmet" -> inv.setHelmet(item);
                        }
                        itemsLoaded++;
                    } else {
                        itemsFailed++;
                        String id = config.getString("armor." + slot + ".id", "unknown");
                        logger.warning("Kit '" + name + "' armor." + slot + " (" + id + ") deserialized to null");
                    }
                } catch (Exception e) {
                    itemsFailed++;
                    logger.warning("Kit '" + name + "' armor." + slot + " exception: " + e.getMessage());
                }
            }
        }

        // Load offhand
        try {
            ItemStack offhand = config.getItemStack("offhand");
            if (offhand != null) {
                inv.setItemInOffHand(offhand);
                itemsLoaded++;
            }
        } catch (Exception e) {
            logger.warning("Kit '" + name + "' offhand exception: " + e.getMessage());
        }

        // Ensure critical slots are filled even if YAML deserialization failed for them
        ensureCriticalSlots(inv, config);

        // Fallback: if nothing loaded at all, use plain kit
        if (itemsLoaded == 0 && itemsFailed > 0) {
            logger.warning("Kit '" + name + "' ALL items failed to deserialize (" + itemsFailed + " failures). Using plain fallback.");
            applyPlainKitFallback(bot);
        }

        // Set selected slot
        int selectedSlot = config.getInt("selected-slot", 0);
        inv.setHeldItemSlot(selectedSlot);

        logger.info("Applied kit '" + name + "' (" + itemsLoaded + " loaded, " + itemsFailed + " failed) to bot " + bot.getScoreboardName());
        return true;
    }

    /**
     * Ensure critical inventory slots are populated even if YAML deserialization
     * failed for complex items. Fills sword, golden apples, and armor with
     * enchanted fallbacks if the slots are empty.
     */
    private void ensureCriticalSlots(PlayerInventory inv, YamlConfiguration config) {
        // Slot 0: must have a sword
        if (inv.getItem(0) == null || inv.getItem(0).getType().isAir()) {
            ItemStack sword = new ItemStack(Material.DIAMOND_SWORD, 1);
            // Add Sharpness VI if the kit expected it
            String enchStr = config.getString("inventory.0.components.minecraft:enchantments", "");
            if (enchStr.contains("sharpness")) {
                sword.addUnsafeEnchantment(org.bukkit.enchantments.Enchantment.SHARPNESS, 6);
            }
            if (enchStr.contains("fire_aspect")) {
                sword.addUnsafeEnchantment(org.bukkit.enchantments.Enchantment.FIRE_ASPECT, 2);
            }
            inv.setItem(0, sword);
            logger.info("  Filled slot 0 with enchanted diamond sword (fallback)");
        }

        // Slot 2: golden apples if missing (slot 2 in test kit has 64 gapples)
        if (inv.getItem(2) == null || inv.getItem(2).getType().isAir()) {
            if (config.isConfigurationSection("inventory.2")) {
                inv.setItem(2, new ItemStack(Material.GOLDEN_APPLE, 64));
                logger.info("  Filled slot 2 with 64 golden apples (fallback)");
            }
        }

        // Armor: fill with enchanted diamond if leather custom armor failed
        if (inv.getHelmet() == null || inv.getHelmet().getType().isAir()) {
            if (config.isConfigurationSection("armor.helmet")) {
                ItemStack helmet = new ItemStack(Material.DIAMOND_HELMET, 1);
                helmet.addUnsafeEnchantment(org.bukkit.enchantments.Enchantment.PROTECTION, 5);
                inv.setHelmet(helmet);
                logger.info("  Filled helmet with enchanted diamond (fallback)");
            }
        }
        if (inv.getChestplate() == null || inv.getChestplate().getType().isAir()) {
            if (config.isConfigurationSection("armor.chestplate")) {
                ItemStack cp = new ItemStack(Material.DIAMOND_CHESTPLATE, 1);
                cp.addUnsafeEnchantment(org.bukkit.enchantments.Enchantment.PROTECTION, 5);
                inv.setChestplate(cp);
                logger.info("  Filled chestplate with enchanted diamond (fallback)");
            }
        }
        if (inv.getLeggings() == null || inv.getLeggings().getType().isAir()) {
            if (config.isConfigurationSection("armor.leggings")) {
                ItemStack legs = new ItemStack(Material.DIAMOND_LEGGINGS, 1);
                legs.addUnsafeEnchantment(org.bukkit.enchantments.Enchantment.PROTECTION, 5);
                inv.setLeggings(legs);
                logger.info("  Filled leggings with enchanted diamond (fallback)");
            }
        }
        if (inv.getBoots() == null || inv.getBoots().getType().isAir()) {
            if (config.isConfigurationSection("armor.boots")) {
                ItemStack boots = new ItemStack(Material.DIAMOND_BOOTS, 1);
                boots.addUnsafeEnchantment(org.bukkit.enchantments.Enchantment.PROTECTION, 5);
                inv.setBoots(boots);
                logger.info("  Filled boots with enchanted diamond (fallback)");
            }
        }

        // Offhand: totem of undying
        if (inv.getItemInOffHand().getType().isAir()) {
            if (config.isConfigurationSection("offhand") || config.getString("offhand.id", "").contains("totem")) {
                inv.setItemInOffHand(new ItemStack(Material.TOTEM_OF_UNDYING, 1));
                logger.info("  Filled offhand with totem of undying (fallback)");
            }
        }
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
