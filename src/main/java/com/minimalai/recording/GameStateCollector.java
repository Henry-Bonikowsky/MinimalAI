package com.minimalai.recording;

import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.passive.PassiveEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.util.hit.BlockHitResult;
import net.minecraft.util.hit.EntityHitResult;
import net.minecraft.util.hit.HitResult;
import net.minecraft.util.math.Vec3d;
import net.minecraft.world.RaycastContext;
import net.minecraft.world.World;

/**
 * Collects game state as 64 normalized float values for neural network input.
 *
 * Layout:
 * [0-7]   Player state: health, hunger, air, pos_y, vel_x, vel_y, vel_z, on_ground
 * [8-9]   Reserved (zeros) - rotation removed to prevent feedback loops
 * [10-15] Crosshair: block_type, entity_type, distance, hardness, break_progress, is_attacking
 * [16-55] Raycasts: 10 directions × 4 values (type, distance, solid, entity)
 * [56-59] Inventory: selected_slot, has_tool, has_weapon, inventory_full
 * [60-63] Movement: sprinting, sneaking, swimming, flying
 */
public class GameStateCollector {
    public static final int STATE_SIZE = 64;

    // Raycast directions (relative to player facing)
    private static final float[][] RAYCAST_ANGLES = {
        {0, 0},      // front
        {45, 0},     // front-right
        {90, 0},     // right
        {135, 0},    // back-right
        {180, 0},    // back
        {-135, 0},   // back-left
        {-90, 0},    // left
        {-45, 0},    // front-left
        {0, -90},    // up
        {0, 90}      // down
    };

    private static final float RAYCAST_DISTANCE = 20.0f;

    private final MinecraftClient client;
    private float blockBreakProgress = 0;
    private boolean isAttacking = false;

    public GameStateCollector() {
        this.client = MinecraftClient.getInstance();
    }

    public void setBlockBreakProgress(float progress) {
        this.blockBreakProgress = progress;
    }

    public void setIsAttacking(boolean attacking) {
        this.isAttacking = attacking;
    }

    public float[] collect() {
        float[] state = new float[STATE_SIZE];

        ClientPlayerEntity player = client.player;
        if (player == null || client.world == null) {
            return state; // Return zeros if no player
        }

        int idx = 0;

        // Player state (8 values)
        state[idx++] = player.getHealth() / 20.0f;  // Normalize to 0-1
        state[idx++] = player.getHungerManager().getFoodLevel() / 20.0f;
        state[idx++] = player.getAir() / (float) player.getMaxAir();
        state[idx++] = normalizeHeight(player.getY());
        state[idx++] = (float) clamp(player.getVelocity().x, -1, 1);
        state[idx++] = (float) clamp(player.getVelocity().y, -1, 1);
        state[idx++] = (float) clamp(player.getVelocity().z, -1, 1);
        state[idx++] = player.isOnGround() ? 1.0f : 0.0f;

        // Reserved (2 values) - previously rotation, removed to prevent feedback loop
        // Camera movement should be based on what the network sees (raycasts), not current orientation
        state[idx++] = 0.0f;
        state[idx++] = 0.0f;

        // Crosshair info (6 values)
        idx = collectCrosshair(state, idx, player);

        // Raycasts (40 values)
        idx = collectRaycasts(state, idx, player);

        // Inventory (4 values)
        idx = collectInventory(state, idx, player);

        // Movement state (4 values)
        state[idx++] = player.isSprinting() ? 1.0f : 0.0f;
        state[idx++] = player.isSneaking() ? 1.0f : 0.0f;
        state[idx++] = player.isSwimming() ? 1.0f : 0.0f;
        state[idx++] = player.getAbilities().flying ? 1.0f : 0.0f;

        return state;
    }

    private int collectCrosshair(float[] state, int idx, ClientPlayerEntity player) {
        HitResult hit = client.crosshairTarget;

        if (hit == null || hit.getType() == HitResult.Type.MISS) {
            // Looking at nothing
            state[idx++] = 0; // block type (air)
            state[idx++] = 0; // entity type (none)
            state[idx++] = 1; // distance (max)
            state[idx++] = 0; // hardness
            state[idx++] = blockBreakProgress;
            state[idx++] = isAttacking ? 1.0f : 0.0f;
        } else if (hit.getType() == HitResult.Type.BLOCK) {
            BlockHitResult blockHit = (BlockHitResult) hit;
            BlockState blockState = client.world.getBlockState(blockHit.getBlockPos());

            state[idx++] = categorizeBlock(blockState);
            state[idx++] = 0; // entity type (none)
            state[idx++] = (float) Math.min(hit.getPos().distanceTo(player.getEyePos()) / RAYCAST_DISTANCE, 1.0);
            state[idx++] = normalizeHardness(blockState.getHardness(client.world, blockHit.getBlockPos()));
            state[idx++] = blockBreakProgress;
            state[idx++] = isAttacking ? 1.0f : 0.0f;
        } else if (hit.getType() == HitResult.Type.ENTITY) {
            EntityHitResult entityHit = (EntityHitResult) hit;
            Entity entity = entityHit.getEntity();

            state[idx++] = 0; // block type (air - looking at entity)
            state[idx++] = categorizeEntity(entity);
            state[idx++] = (float) Math.min(hit.getPos().distanceTo(player.getEyePos()) / RAYCAST_DISTANCE, 1.0);
            state[idx++] = 0; // hardness (N/A for entities)
            state[idx++] = 0; // break progress (N/A)
            state[idx++] = isAttacking ? 1.0f : 0.0f;
        } else {
            // Fallback
            state[idx++] = 0;
            state[idx++] = 0;
            state[idx++] = 1;
            state[idx++] = 0;
            state[idx++] = 0;
            state[idx++] = 0;
        }

        return idx;
    }

    private int collectRaycasts(float[] state, int idx, ClientPlayerEntity player) {
        World world = client.world;
        Vec3d eyePos = player.getEyePos();
        float playerYaw = player.getYaw();
        float playerPitch = player.getPitch();

        for (float[] angles : RAYCAST_ANGLES) {
            float yawOffset = angles[0];
            float pitchOffset = angles[1];

            // Calculate direction
            float yaw = playerYaw + yawOffset;
            float pitch = playerPitch + pitchOffset;

            // Clamp pitch
            pitch = Math.max(-90, Math.min(90, pitch));

            // Convert to direction vector
            double yawRad = Math.toRadians(yaw);
            double pitchRad = Math.toRadians(pitch);

            double x = -Math.sin(yawRad) * Math.cos(pitchRad);
            double y = -Math.sin(pitchRad);
            double z = Math.cos(yawRad) * Math.cos(pitchRad);

            Vec3d direction = new Vec3d(x, y, z).normalize();
            Vec3d endPos = eyePos.add(direction.multiply(RAYCAST_DISTANCE));

            // Block raycast
            BlockHitResult blockHit = world.raycast(new RaycastContext(
                eyePos, endPos, RaycastContext.ShapeType.COLLIDER, RaycastContext.FluidHandling.NONE, player
            ));

            float blockType = 0;
            float distance = 1;
            float isSolid = 0;
            float hasEntity = 0;

            if (blockHit.getType() == HitResult.Type.BLOCK) {
                BlockState blockState = world.getBlockState(blockHit.getBlockPos());
                blockType = categorizeBlock(blockState);
                distance = (float) Math.min(blockHit.getPos().distanceTo(eyePos) / RAYCAST_DISTANCE, 1.0);
                isSolid = blockState.isSolidBlock(world, blockHit.getBlockPos()) ? 1.0f : 0.0f;
            }

            // Entity check in this direction
            for (Entity entity : world.getEntitiesByClass(LivingEntity.class,
                    player.getBoundingBox().expand(RAYCAST_DISTANCE), e -> e != player)) {
                Vec3d toEntity = entity.getPos().add(0, entity.getHeight() / 2, 0).subtract(eyePos);
                double dot = toEntity.normalize().dotProduct(direction);
                if (dot > 0.9) { // Within ~25 degrees
                    double entityDist = toEntity.length();
                    if (entityDist < RAYCAST_DISTANCE && entityDist / RAYCAST_DISTANCE < distance) {
                        hasEntity = categorizeEntity(entity);
                        break;
                    }
                }
            }

            state[idx++] = blockType;
            state[idx++] = distance;
            state[idx++] = isSolid;
            state[idx++] = hasEntity;
        }

        return idx;
    }

    private int collectInventory(float[] state, int idx, ClientPlayerEntity player) {
        state[idx++] = player.getInventory().getSelectedSlot() / 8.0f; // Normalize 0-8 to 0-1

        // Check for tools and weapons using item ID
        boolean hasTool = false;
        boolean hasWeapon = false;
        int filledSlots = 0;

        for (int i = 0; i < player.getInventory().size(); i++) {
            ItemStack stack = player.getInventory().getStack(i);
            if (!stack.isEmpty()) {
                filledSlots++;
                String itemId = Registries.ITEM.getId(stack.getItem()).toString().toLowerCase();

                // Tools: pickaxes, axes, shovels, hoes
                if (itemId.contains("pickaxe") || itemId.contains("axe") ||
                    itemId.contains("shovel") || itemId.contains("hoe")) {
                    hasTool = true;
                }
                // Weapons: swords, axes, bows, crossbows, tridents
                if (itemId.contains("sword") || itemId.contains("axe") ||
                    itemId.contains("bow") || itemId.contains("trident")) {
                    hasWeapon = true;
                }
            }
        }

        state[idx++] = hasTool ? 1.0f : 0.0f;
        state[idx++] = hasWeapon ? 1.0f : 0.0f;
        state[idx++] = filledSlots >= 36 ? 1.0f : 0.0f; // Main inventory full

        return idx;
    }

    private float categorizeBlock(BlockState blockState) {
        Block block = blockState.getBlock();

        if (blockState.isAir()) return 0.0f;

        String name = block.getTranslationKey().toLowerCase();

        // Ores (valuable)
        if (name.contains("ore")) return 0.9f;

        // Wood/logs
        if (name.contains("log") || name.contains("wood") || name.contains("plank")) return 0.7f;

        // Plants/crops
        if (name.contains("leaves") || name.contains("grass") || name.contains("flower") ||
            name.contains("crop") || name.contains("wheat") || name.contains("carrot")) return 0.3f;

        // Water/lava
        if (block == Blocks.WATER) return 0.2f;
        if (block == Blocks.LAVA) return 0.1f;

        // Default solid block
        return 0.5f;
    }

    private float categorizeEntity(Entity entity) {
        if (entity instanceof PlayerEntity) return 1.0f;
        if (entity instanceof HostileEntity) return 0.8f;
        if (entity instanceof PassiveEntity) return 0.4f;
        return 0.2f; // Other (items, projectiles, etc.)
    }

    private float normalizeHardness(float hardness) {
        if (hardness < 0) return 1.0f; // Unbreakable
        return (float) Math.min(hardness / 50.0, 1.0); // Obsidian is 50
    }

    private float normalizeHeight(double y) {
        // Minecraft world height is roughly -64 to 320
        return (float) clamp((y + 64) / 384.0, 0, 1);
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
}
