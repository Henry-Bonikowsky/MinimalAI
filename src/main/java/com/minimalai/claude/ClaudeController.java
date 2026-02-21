package com.minimalai.claude;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.minimalai.bot.FakePlayerManager;
import com.minimalai.bot.FakePlayerManager.BotContext;
import com.mojang.authlib.GameProfile;
import com.mojang.authlib.properties.Property;
import com.google.common.collect.LinkedHashMultimap;
import com.mojang.authlib.properties.PropertyMap;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.game.*;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.craftbukkit.CraftServer;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.logging.Logger;

/**
 * Manages the "Claude" fake player lifecycle and interactions.
 * Remote-controlled — no BotBrain or AI tick loop.
 */
public class ClaudeController {

    private static final String BOT_NAME = "Imperius";
    private static final int HUD_BUFFER_SIZE = 10;
    private static final String SKIN_TEXTURE_VALUE =
            "ewogICJ0aW1lc3RhbXAiIDogMTc0NjQ1NDAzNjk1OSwKICAicHJvZmlsZUlkIiA6ICI2OTU3YjQyOTZiOTI0NGJkOGQzNWJiZGU4M2JiZDI3OSIsCiAgInByb2ZpbGVOYW1lIiA6ICJjb2xpblBBUEEiLAogICJzaWduYXR1cmVSZXF1aXJlZCIgOiB0cnVlLAogICJ0ZXh0dXJlcyIgOiB7CiAgICAiU0tJTiIgOiB7CiAgICAgICJ1cmwiIDogImh0dHA6Ly90ZXh0dXJlcy5taW5lY3JhZnQubmV0L3RleHR1cmUvNGY1ZTdmN2NhYjE1OTZjY2E1OTg2MDQyYTBmZDM0YWI5MDZjNGI1ZjAxMzM2M2NkNGFlYmJlMTkzZDdiZWFjMiIKICAgIH0KICB9Cn0=";
    private static final String SKIN_TEXTURE_SIGNATURE =
            "d7B5aPCvOilUTIqGMoMEuGMChcwYuxuvxTAK0UB8Ddmkw5gDtt2IrQRIEUMOdv3ITptOZ1sxbAZKyfD/dNkuczcT8ekb+AJfmncLVcc6ptjMnRrqVGzSDpiwpGEUUn43s+bY0JPOEuG+sAiCjuQtI+TZkASzjTwWq/pxJT+X16rk3hMU0lPs5ShY//qql3jd92xu3Oe6BWdt281azSHDPllwz3rrHbJCsHyhKwMLHlXxnolZiiKD66t+1X9svSYlpcdqDlFZg2x2SpX6y2quNRGXuAY0m7izhIZpBhJeDJs7Ep/vDF6cD3XnB0ZhlpoDAxSwmkMLzZC24mc4FHFSCX2WKAPam7+/Uh057jZqd4f24rtsgjQNnvs2Zzs/EdJNe2BOiMpf9b0+gIh2Y1fNLAMltvY3W/XuHI7G8J9TM45nxoSuiGRE5r57BGn/erl/cdlA7DW9gK9E+l82fZBzGiH2hC+Syqy9UaDMz0L0Lzbev5WQyZzk/ViOeq6HulkfTwPOBgXQj2idtVfCvgr7PGCbnJqj4pM4VY57NA4aacAtqy21yccjVQwNfemjTH9g994lbvqm6tsrfculjoCxsDLc/ymKagQd2davdLJSUOyVk1gtRw46UgCmLR5JMb1A8o9YK/ZQRuW1sgDE19sYlnBMm7SJTbTVZewfp/HTJu0=";

    private final FakePlayerManager botManager;
    private final Logger logger;

    // HUD packet capture buffers
    private final ConcurrentLinkedDeque<String> titleBuffer = new ConcurrentLinkedDeque<>();
    private final ConcurrentLinkedDeque<String> actionBarBuffer = new ConcurrentLinkedDeque<>();
    private volatile String lastSubtitle = "";

    public ClaudeController(FakePlayerManager botManager, Logger logger) {
        this.botManager = botManager;
        this.logger = logger;
    }

    // ── Lifecycle ──

    public JsonObject spawn(String worldName, double x, double y, double z) {
        if (isSpawned()) return error(BOT_NAME + " is already spawned");

        World world = Bukkit.getWorld(worldName);
        if (world == null) return error("World '" + worldName + "' not found");

        // Build GameProfile with fixed UUID and skin BEFORE spawning
        UUID fixedUuid = UUID.nameUUIDFromBytes(("OfflinePlayer:" + BOT_NAME).getBytes(StandardCharsets.UTF_8));
        var multimap = LinkedHashMultimap.<String, Property>create();
        multimap.put("textures", new Property("textures", SKIN_TEXTURE_VALUE, SKIN_TEXTURE_SIGNATURE));
        PropertyMap props = new PropertyMap(multimap);
        GameProfile profile = new GameProfile(fixedUuid, BOT_NAME, props);

        Location loc = new Location(world, x, y, z);
        BotContext ctx = botManager.spawn(world, loc, BOT_NAME, profile);
        if (ctx == null) return error("Failed to spawn " + BOT_NAME);

        // Mark as remote-controlled and remove NPC tag (NPC tag makes plugins skip damage)
        var plugin = Bukkit.getPluginManager().getPlugin("MinimalAI");
        ctx.serverPlayer().getBukkitEntity().setMetadata("RemoteBot",
                new org.bukkit.metadata.FixedMetadataValue(plugin, true));
        ctx.serverPlayer().getBukkitEntity().removeMetadata("NPC", plugin);

        // Set LuckPerms owner group via API
        try {
            net.luckperms.api.LuckPerms lp = net.luckperms.api.LuckPermsProvider.get();
            lp.getUserManager().loadUser(fixedUuid, BOT_NAME).thenAcceptAsync(user -> {
                user.data().add(net.luckperms.api.node.types.InheritanceNode.builder("owner").build());
                lp.getUserManager().saveUser(user);
                logger.info("Set LuckPerms group 'owner' for " + BOT_NAME);
            });
        } catch (Exception e) {
            logger.warning("LuckPerms not available, skipping group assignment: " + e.getMessage());
        }

        // Set up packet capture for HUD interception
        ctx.fakeConnection().setPacketCapture(this::capturePacket);

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("name", BOT_NAME);
        result.add("pos", posJson(x, y, z));
        return result;
    }

    public JsonObject despawn() {
        if (!isSpawned()) return error(BOT_NAME + " is not spawned");
        botManager.despawn(BOT_NAME);
        titleBuffer.clear();
        actionBarBuffer.clear();

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("message", "Claude despawned");
        return result;
    }

    // ── Movement ──

    public JsonObject teleport(double x, double y, double z) {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");
        player.snapTo(x, y, z, player.getYRot(), player.getXRot());

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.add("pos", posJson(x, y, z));
        return result;
    }

    public JsonObject lookAt(double x, double y, double z) {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        double dx = x - player.getX();
        double dy = y - player.getEyeY();
        double dz = z - player.getZ();
        double dist = Math.sqrt(dx * dx + dz * dz);

        float yaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
        float pitch = (float) Math.toDegrees(-Math.atan2(dy, dist));

        player.setYRot(yaw);
        player.setXRot(pitch);
        player.setYHeadRot(yaw);

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("yaw", yaw);
        result.addProperty("pitch", pitch);
        return result;
    }

    public JsonObject walkTo(double x, double z) {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        double dx = x - player.getX();
        double dz = z - player.getZ();
        float yaw = (float) Math.toDegrees(Math.atan2(-dx, dz));

        player.setYRot(yaw);
        player.setYHeadRot(yaw);
        player.zza = 1.0f; // forward
        player.xxa = 0.0f; // no strafe

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("yaw", yaw);
        result.addProperty("walking", true);
        return result;
    }

    public JsonObject stopMoving() {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        player.zza = 0.0f;
        player.xxa = 0.0f;
        player.setJumping(false);

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("message", "Stopped");
        return result;
    }

    public JsonObject jump() {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        player.setJumping(true);

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        return result;
    }

    public JsonObject flyTo(double x, double y, double z) {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        // Enable creative flight
        player.getAbilities().mayfly = true;
        player.getAbilities().flying = true;
        player.onUpdateAbilities();

        player.snapTo(x, y, z, player.getYRot(), player.getXRot());
        // Clear any velocity
        player.setDeltaMovement(Vec3.ZERO);

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.add("pos", posJson(x, y, z));
        result.addProperty("flying", true);
        return result;
    }

    // ── Combat ──

    public JsonObject attack(String targetName) {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        // Swing arm
        player.swing(InteractionHand.MAIN_HAND);

        // Find target
        Entity target = null;
        if (targetName != null && !targetName.isEmpty()) {
            // Find by name
            AABB box = player.getBoundingBox().inflate(6);
            for (Entity e : player.level().getEntities(player, box)) {
                if (e instanceof ServerPlayer sp && sp.getGameProfile().name().equalsIgnoreCase(targetName)) {
                    target = e;
                    break;
                }
                if (e.getCustomName() != null && e.getCustomName().getString().equalsIgnoreCase(targetName)) {
                    target = e;
                    break;
                }
            }
        } else {
            // Attack nearest living entity in reach (4.5 blocks)
            AABB box = player.getBoundingBox().inflate(4.5);
            double closest = Double.MAX_VALUE;
            for (Entity e : player.level().getEntities(player, box)) {
                if (!(e instanceof LivingEntity)) continue;
                double d = player.distanceToSqr(e);
                if (d < closest) {
                    closest = d;
                    target = e;
                }
            }
        }

        if (target == null) return error("No target found");

        player.attack(target);

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("target", target instanceof ServerPlayer sp
                ? sp.getGameProfile().name()
                : target.getType().toShortString());
        result.addProperty("dist", Math.round(Math.sqrt(player.distanceToSqr(target)) * 10.0) / 10.0);
        return result;
    }

    public JsonObject useItem() {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        ItemStack held = player.getMainHandItem();
        player.gameMode.useItem(player, player.level(), held, InteractionHand.MAIN_HAND);

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("item", held.isEmpty() ? "empty" : held.getItem().toString());
        return result;
    }

    // ── Inventory ──

    public JsonObject setSlot(int slot) {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");
        if (slot < 0 || slot > 8) return error("Slot must be 0-8");

        player.getInventory().setSelectedSlot(slot);

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("slot", slot);
        ItemStack held = player.getMainHandItem();
        result.addProperty("item", held.isEmpty() ? "empty" : held.getItem().toString());
        return result;
    }

    public JsonObject getInventory() {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        JsonArray items = new JsonArray();
        var inv = player.getInventory();
        for (int i = 0; i < inv.getContainerSize(); i++) {
            ItemStack stack = inv.getItem(i);
            if (stack.isEmpty()) continue;
            JsonObject item = new JsonObject();
            item.addProperty("slot", i);
            item.addProperty("item", stack.getItem().toString());
            item.addProperty("count", stack.getCount());
            items.add(item);
        }

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("selected", inv.getSelectedSlot());
        result.add("items", items);
        return result;
    }

    public JsonObject getGui() {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        var menu = player.containerMenu;
        if (menu == player.inventoryMenu) return error("No GUI open");

        JsonArray slots = new JsonArray();
        for (int i = 0; i < menu.slots.size(); i++) {
            var slot = menu.slots.get(i);
            ItemStack stack = slot.getItem();
            if (stack.isEmpty()) continue;
            JsonObject item = new JsonObject();
            item.addProperty("slot", i);
            item.addProperty("item", stack.getItem().toString());
            item.addProperty("count", stack.getCount());
            if (stack.has(net.minecraft.core.component.DataComponents.CUSTOM_NAME)) {
                item.addProperty("name", stack.getHoverName().getString());
            }
            slots.add(item);
        }

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("type", menu.getType().toString());
        result.addProperty("size", menu.slots.size());
        result.add("slots", slots);
        return result;
    }

    public JsonObject clickSlot(int slotId, boolean rightClick) {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        var menu = player.containerMenu;
        int button = rightClick ? 1 : 0;
        // ClickType.PICKUP = normal click (left=pickup/place, right=pickup-one/place-one)
        net.minecraft.world.inventory.ClickType clickType = net.minecraft.world.inventory.ClickType.PICKUP;

        menu.clicked(slotId, button, clickType, player);
        menu.broadcastChanges();

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("slot", slotId);
        result.addProperty("button", rightClick ? "right" : "left");
        return result;
    }

    public JsonObject closeGui() {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        player.closeContainer();

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("message", "GUI closed");
        return result;
    }

    // ── Status ──

    public JsonObject getStatus() {
        ServerPlayer player = getPlayer();
        if (player == null) return error(BOT_NAME + " is not spawned");

        JsonObject result = new JsonObject();
        result.addProperty("ok", true);
        result.addProperty("name", BOT_NAME);
        result.add("pos", posJson(player.getX(), player.getY(), player.getZ()));
        result.addProperty("health", player.getHealth());
        result.addProperty("food", player.getFoodData().getFoodLevel());
        result.addProperty("yaw", player.getYRot());
        result.addProperty("pitch", player.getXRot());
        result.addProperty("world", player.level().getWorld().getName());
        result.addProperty("flying", player.getAbilities().flying);
        ItemStack held = player.getMainHandItem();
        result.addProperty("held", held.isEmpty() ? "empty" : held.getItem().toString());
        return result;
    }

    // ── HUD packet capture ──

    private void capturePacket(Packet<?> packet) {
        if (packet instanceof ClientboundSetTitleTextPacket p) {
            String text = p.text().getString();
            titleBuffer.addLast(text);
            while (titleBuffer.size() > HUD_BUFFER_SIZE) titleBuffer.pollFirst();
        } else if (packet instanceof ClientboundSetSubtitleTextPacket p) {
            lastSubtitle = p.text().getString();
        } else if (packet instanceof ClientboundSetActionBarTextPacket p) {
            String text = p.text().getString();
            actionBarBuffer.addLast(text);
            while (actionBarBuffer.size() > HUD_BUFFER_SIZE) actionBarBuffer.pollFirst();
        }
    }

    public JsonObject getHud() {
        JsonObject result = new JsonObject();
        result.addProperty("ok", true);

        JsonArray titles = new JsonArray();
        for (String t : titleBuffer) titles.add(t);
        result.add("titles", titles);
        result.addProperty("subtitle", lastSubtitle);

        JsonArray bars = new JsonArray();
        for (String a : actionBarBuffer) bars.add(a);
        result.add("actionbars", bars);

        return result;
    }

    // ── Accessors ──

    public boolean isSpawned() {
        return botManager.getBot(BOT_NAME) != null;
    }

    public ServerPlayer getPlayer() {
        BotContext ctx = botManager.getBot(BOT_NAME);
        return ctx != null ? ctx.serverPlayer() : null;
    }

    // ── Utilities ──

    static JsonObject posJson(double x, double y, double z) {
        JsonObject pos = new JsonObject();
        pos.addProperty("x", Math.round(x * 100.0) / 100.0);
        pos.addProperty("y", Math.round(y * 100.0) / 100.0);
        pos.addProperty("z", Math.round(z * 100.0) / 100.0);
        return pos;
    }

    static JsonObject error(String message) {
        JsonObject result = new JsonObject();
        result.addProperty("ok", false);
        result.addProperty("error", message);
        return result;
    }
}
