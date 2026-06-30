package com.minimalai.bot;

import com.mojang.authlib.GameProfile;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.players.PlayerList;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.craftbukkit.CraftWorld;

import org.bukkit.metadata.FixedMetadataValue;
import org.bukkit.plugin.Plugin;

import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Manages the lifecycle of NMS fake players (bots).
 *
 * Uses the Imperium JAR's native fake player API (PlayerList.createFakePlayer /
 * removeFakePlayer) via reflection. These methods handle all networking, entity
 * tracking, and disconnect immunity server-side — no plugin-side hacks needed.
 */
public class FakePlayerManager {

    private final Logger logger;
    private final Plugin plugin;
    private final Map<String, BotContext> activeBots = new ConcurrentHashMap<>();
    private int nextBotId = 1;

    // Cached reflection handles for Imperium JAR's native API
    private Method createFakePlayerMethod;
    private Method removeFakePlayerMethod;
    private boolean nativeApiAvailable = false;

    public FakePlayerManager(Logger logger, Plugin plugin) {
        this.logger = logger;
        this.plugin = plugin;
        initNativeApi();
    }

    private void initNativeApi() {
        try {
            createFakePlayerMethod = PlayerList.class.getMethod(
                "createFakePlayer", ServerLevel.class, String.class,
                double.class, double.class, double.class
            );
            removeFakePlayerMethod = PlayerList.class.getMethod(
                "removeFakePlayer", ServerPlayer.class
            );
            nativeApiAvailable = true;
            logger.info("Imperium native fake player API detected — using server-side bot support");
        } catch (NoSuchMethodException e) {
            nativeApiAvailable = false;
            logger.severe("Imperium native fake player API NOT found — bots will not work!");
            logger.severe("Ensure you are running the patched Imperium JAR with createFakePlayer/removeFakePlayer.");
        }
    }

    /**
     * Spawn a fake player at the given location.
     */
    public BotContext spawn(World world, Location location, String name) {
        return spawn(world, location, name, null);
    }

    /**
     * Spawn a fake player with an optional pre-built GameProfile.
     */
    public BotContext spawn(World world, Location location, String name, GameProfile providedProfile) {
        if (!nativeApiAvailable) {
            logger.severe("Cannot spawn bot — Imperium native API not available");
            return null;
        }

        if (name == null || name.isBlank()) {
            name = "Bot_" + nextBotId++;
        }

        if (activeBots.containsKey(name)) {
            logger.warning("Bot with name '" + name + "' already exists");
            return null;
        }

        if (name.length() > 16) {
            name = name.substring(0, 16);
        }

        try {
            MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
            ServerLevel level = ((CraftWorld) world).getHandle();

            // Call Imperium's PlayerList.createFakePlayer(level, name, x, y, z)
            ServerPlayer bot = (ServerPlayer) createFakePlayerMethod.invoke(
                server.getPlayerList(),
                level, name, location.getX(), location.getY(), location.getZ()
            );

            // Set rotation (createFakePlayer sets 0,0)
            bot.snapTo(location.getX(), location.getY(), location.getZ(),
                       location.getYaw(), location.getPitch());

            // Mark as NPC so other plugins (TAB, Citizens-compat, etc.) skip this player
            bot.getBukkitEntity().setMetadata("NPC", new FixedMetadataValue(plugin, true));

            // Sync position tracking
            bot.connection.resetPosition();

            BotContext ctx = new BotContext(name, bot);
            activeBots.put(name, ctx);

            logger.info("Spawned bot '" + name + "' at " + formatLocation(location));
            return ctx;

        } catch (Exception e) {
            logger.severe("Failed to spawn bot '" + name + "': " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Remove a bot from the server.
     */
    public boolean despawn(String name) {
        BotContext ctx = activeBots.remove(name);
        if (ctx == null) return false;

        try {
            MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
            ServerPlayer bot = ctx.serverPlayer();

            // Call Imperium's PlayerList.removeFakePlayer(player)
            removeFakePlayerMethod.invoke(server.getPlayerList(), bot);

            logger.info("Despawned bot '" + name + "'");
        } catch (Exception e) {
            logger.warning("Error despawning bot '" + name + "': " + e.getMessage());
        }
        return true;
    }

    /**
     * Remove all active bots. Intended for plugin disable / cleanup.
     */
    public void despawnAll() {
        for (String name : new ArrayList<>(activeBots.keySet())) {
            despawn(name);
        }
    }

    public BotContext getBot(String name) {
        return activeBots.get(name);
    }

    public Collection<BotContext> getAllBots() {
        return Collections.unmodifiableCollection(activeBots.values());
    }

    public int botCount() {
        return activeBots.size();
    }

    public boolean isBot(String name) {
        return activeBots.containsKey(name);
    }

    private static String formatLocation(Location loc) {
        return String.format("(%.1f, %.1f, %.1f) in %s",
            loc.getX(), loc.getY(), loc.getZ(),
            loc.getWorld() != null ? loc.getWorld().getName() : "?");
    }

    /**
     * Lightweight wrapper around a bot's server-side state.
     */
    public record BotContext(String name, ServerPlayer serverPlayer) {

        public Location bukkitLocation() {
            return serverPlayer.getBukkitEntity().getLocation();
        }
    }
}
