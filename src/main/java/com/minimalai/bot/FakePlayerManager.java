package com.minimalai.bot;

import com.mojang.authlib.GameProfile;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ClientInformation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.craftbukkit.CraftServer;
import org.bukkit.craftbukkit.CraftWorld;

import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerKickEvent;
import org.bukkit.event.player.PlayerTeleportEvent;
import org.bukkit.metadata.FixedMetadataValue;
import org.bukkit.plugin.Plugin;

import net.minecraft.world.level.GameType;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Manages the lifecycle of NMS fake players (bots).
 *
 * Each bot is a real ServerPlayer injected into the server's player list
 * so that it is visible to all online players, appears in tab, and can
 * interact with the world exactly like a human player.
 *
 * Also listens for PlayerKickEvent to prevent plugins (PacketEvents, etc.)
 * from kicking bot players.
 */
public class FakePlayerManager implements Listener {

    private final Logger logger;
    private final Plugin plugin;
    private final Map<String, BotContext> activeBots = new ConcurrentHashMap<>();
    private int nextBotId = 1;

    public FakePlayerManager(Logger logger, Plugin plugin) {
        this.logger = logger;
        this.plugin = plugin;
    }

    /**
     * Spawn a fake player at the given location.
     *
     * @param world    Bukkit world to spawn in
     * @param location spawn location (position + look direction)
     * @param name     display name; if null a default "Bot_N" name is generated
     * @return the BotContext wrapping the ServerPlayer, or null on failure
     */
    public BotContext spawn(World world, Location location, String name) {
        if (name == null || name.isBlank()) {
            name = "Bot_" + nextBotId++;
        }

        // Prevent duplicate names
        if (activeBots.containsKey(name)) {
            logger.warning("Bot with name '" + name + "' already exists");
            return null;
        }

        // Cap name length to 16 (Minecraft username limit)
        if (name.length() > 16) {
            name = name.substring(0, 16);
        }

        try {
            MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
            ServerLevel level = ((CraftWorld) world).getHandle();

            // Build an offline-mode GameProfile with a random UUID
            GameProfile profile = new GameProfile(UUID.randomUUID(), name);

            // Create the ServerPlayer
            // TODO: ClientInformation.createDefault() is the typical factory in 1.21.x.
            //       If your Paper build names it differently, adjust here.
            ServerPlayer bot = new ServerPlayer(server, level, profile, ClientInformation.createDefault());

            // Position + look direction
            bot.setPos(location.getX(), location.getY(), location.getZ());
            bot.setYRot(location.getYaw());
            bot.setXRot(location.getPitch());
            bot.setYHeadRot(location.getYaw());

            // Build a CommonListenerCookie for the login handshake.
            // Paper 1.21.10 signature:
            // CommonListenerCookie(GameProfile, int, ClientInformation, boolean, String, Set<String>, KeepAlive)
            CommonListenerCookie cookie = new CommonListenerCookie(
                profile,
                0,                                    // latency (ms)
                ClientInformation.createDefault(),
                false,                                // transferred
                "",                                   // client brand
                java.util.Set.of(),                   // extra data
                null                                  // KeepAlive (not needed for fake players)
            );

            // Wire up fake networking (no-op packet handling)
            FakeConnection fakeConn = FakeConnection.create(server, bot, cookie);

            // Register the bot with the server's player list.
            // This fires PlayerJoinEvent, adds to tab list, etc.
            server.getPlayerList().placeNewPlayer(fakeConn.connection(), bot, cookie);

            // placeNewPlayer creates its own ServerGamePacketListenerImpl,
            // overwriting our NoOpPacketListener. Swap ours back in so
            // tick() is no-op'd (prevents keepalive timeout disconnect).
            fakeConn.reattach(bot);

            // Ensure bot is in survival mode and hittable
            bot.setGameMode(GameType.SURVIVAL);
            bot.setInvulnerable(false);

            // Mark as NPC so other plugins (TAB, Citizens-compat, etc.) skip this player
            bot.getBukkitEntity().setMetadata("NPC", new FixedMetadataValue(plugin, true));

            // Re-position after placeNewPlayer via NMS setPos (no Bukkit events).
            // placeNewPlayer sends the bot to world spawn; we force it back here.
            bot.setPos(location.getX(), location.getY(), location.getZ());
            bot.setYRot(location.getYaw());
            bot.setXRot(location.getPitch());
            bot.setYHeadRot(location.getYaw());

            BotContext ctx = new BotContext(name, bot, fakeConn);
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
     *
     * @param name the bot's display name
     * @return true if a bot with that name was found and removed
     */
    public boolean despawn(String name) {
        BotContext ctx = activeBots.remove(name);
        if (ctx == null) return false;

        try {
            MinecraftServer server = ((CraftServer) Bukkit.getServer()).getServer();
            ServerPlayer bot = ctx.serverPlayer();

            // Remove from the server's player list (fires PlayerQuitEvent, tab removal, etc.)
            server.getPlayerList().remove(bot);

            // Remove entity from the level
            bot.discard();

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
        // Copy keys to avoid ConcurrentModificationException
        for (String name : new ArrayList<>(activeBots.keySet())) {
            despawn(name);
        }
    }

    /**
     * Look up a bot by name.
     *
     * @return the BotContext, or null if not found
     */
    public BotContext getBot(String name) {
        return activeBots.get(name);
    }

    /**
     * @return an unmodifiable view of all active bots
     */
    public Collection<BotContext> getAllBots() {
        return Collections.unmodifiableCollection(activeBots.values());
    }

    /**
     * @return number of active bots
     */
    public int botCount() {
        return activeBots.size();
    }

    /**
     * Check if a player name belongs to one of our bots.
     */
    public boolean isBot(String name) {
        return activeBots.containsKey(name);
    }

    /**
     * Prevent plugins (PacketEvents, TAB, etc.) from kicking our bots.
     */
    @EventHandler(priority = EventPriority.LOWEST)
    public void onBotKick(PlayerKickEvent event) {
        if (isBot(event.getPlayer().getName())) {
            event.setCancelled(true);
        }
    }

    /**
     * Block ALL teleport events for bots. We reposition bots using NMS
     * setPos() which doesn't fire Bukkit events, so any teleport event
     * reaching here is from an external plugin (Essentials, WorldGuard, etc.)
     * and must be cancelled.
     */
    @EventHandler(priority = EventPriority.LOWEST)
    public void onBotTeleport(PlayerTeleportEvent event) {
        if (!isBot(event.getPlayer().getName())) return;
        event.setCancelled(true);
    }

    private static String formatLocation(Location loc) {
        return String.format("(%.1f, %.1f, %.1f) in %s",
            loc.getX(), loc.getY(), loc.getZ(),
            loc.getWorld() != null ? loc.getWorld().getName() : "?");
    }

    // ------------------------------------------------------------------
    //  Inner record that bundles a bot's NMS player + networking handles
    // ------------------------------------------------------------------

    /**
     * Lightweight wrapper around a bot's server-side state.
     */
    public record BotContext(String name, ServerPlayer serverPlayer, FakeConnection fakeConnection) {

        /**
         * @return the bot's current Bukkit Location
         */
        public Location bukkitLocation() {
            return serverPlayer.getBukkitEntity().getLocation();
        }
    }
}
