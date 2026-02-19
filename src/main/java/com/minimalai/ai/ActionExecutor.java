package com.minimalai.ai;

import net.minecraft.core.component.DataComponents;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.projectile.ThrownEnderpearl;
import net.minecraft.world.entity.projectile.ThrownSplashPotion;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.item.alchemy.PotionContents;
import net.minecraft.world.item.alchemy.Potions;
import net.minecraft.world.entity.MoverType;
import net.minecraft.world.phys.Vec3;
import org.bukkit.entity.Player;
import org.jetbrains.annotations.Nullable;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

import static com.minimalai.ai.ActionSpace.*;

/**
 * Converts a 35 multi-binary action vector into NMS operations on a
 * server-side {@link ServerPlayer} (typically a fake bot player).
 *
 * Designed to run once per server tick (50 ms). The caller provides the
 * current target and nearby entities; execute() returns the (possibly
 * updated) target based on target-selection actions.
 *
 * ArcaneSigils integration is optional: pass null if the plugin isn't loaded.
 */
public class ActionExecutor {

    private static final Logger LOGGER = Logger.getLogger("MinimalAI");

    /**
     * Minimal interface for activating ArcaneSigils abilities server-side.
     * Implement and pass to the constructor if ArcaneSigils is loaded.
     */
    @FunctionalInterface
    public interface ArcaneSigilsAPI {
        /**
         * Activate a sigil ability for the given player.
         *
         * @param player   the Bukkit player
         * @param slotIndex 0-based sigil slot index (0-11)
         * @return true if the ability fired, false if on cooldown / invalid
         */
        boolean activateAbility(Player player, int slotIndex);
    }

    /**
     * Optional interface for querying sigil cooldown state (for action masking).
     */
    public interface SigilCooldownQuery {
        /** @return true if the sigil slot is off cooldown and ready to fire */
        boolean isReady(Player player, int slotIndex);
    }

    // Approximate walk/sprint speeds in blocks/tick (matching sim values)
    private static final float WALK_SPEED = 0.1f;
    private static final float SPRINT_SPEED = 0.26f;

    // Vanilla melee reach and max angle for hit registration
    private static final double MELEE_REACH = 3.0;
    private static final double MELEE_ANGLE_COS = Math.cos(Math.toRadians(60)); // ~60° cone

    private final @Nullable ArcaneSigilsAPI sigilsApi;
    private final @Nullable SigilCooldownQuery sigilCooldowns;

    // Per-bot eating state (keyed by entity ID since ActionExecutor is shared)
    private static class EatingState {
        int ticksRemaining = 0;
        InteractionHand hand = null;
    }
    private final Map<Integer, EatingState> eatingStates = new ConcurrentHashMap<>();

    /**
     * @param sigilsApi      nullable; set if ArcaneSigils is loaded
     * @param sigilCooldowns nullable; used for action masking if available
     */
    public ActionExecutor(@Nullable ArcaneSigilsAPI sigilsApi,
                          @Nullable SigilCooldownQuery sigilCooldowns) {
        this.sigilsApi = sigilsApi;
        this.sigilCooldowns = sigilCooldowns;
    }

    public ActionExecutor(@Nullable ArcaneSigilsAPI sigilsApi) {
        this(sigilsApi, null);
    }

    public ActionExecutor() {
        this(null, null);
    }

    // ------------------------------------------------------------------
    //  Main execution
    // ------------------------------------------------------------------

    /**
     * Apply a 35-element multi-binary action vector to the bot player.
     *
     * @param bot             the fake ServerPlayer to control
     * @param actions         int[35] with 0 or 1 per action slot
     * @param target          current combat target (nullable)
     * @param nearbyEntities  entities sorted by distance for target selection
     * @return the (possibly updated) target entity
     */
    public @Nullable LivingEntity execute(ServerPlayer bot,
                                          int[] actions,
                                          @Nullable LivingEntity target,
                                          List<LivingEntity> nearbyEntities) {

        applyMovement(bot, actions);
        applyJump(bot, actions);
        applySneak(bot, actions);
        applySprint(bot, actions);
        applySprintReset(bot, actions);
        // Look BEFORE attack so the bot faces the target first
        applyLookIntent(bot, actions, target);
        applyAttack(bot, actions, target);
        applyBlock(bot, actions);
        applyEatGap(bot, actions);
        applyThrowPot(bot, actions);
        applyThrowPearl(bot, actions);
        // ACT_SWAP_WEAPON (13) is always masked -- no-op
        applySigils(bot, actions);

        tickEating(bot);

        return resolveTarget(actions, target, nearbyEntities);
    }

    // ------------------------------------------------------------------
    //  Movement (actions 0-3)
    // ------------------------------------------------------------------

    private void applyMovement(ServerPlayer bot, int[] actions) {
        float forward = actions[ACT_FORWARD] - actions[ACT_BACKWARD]; // -1, 0, or 1
        float strafe = actions[ACT_STRAFE_LEFT] - actions[ACT_STRAFE_RIGHT]; // -1, 0, or 1

        if (forward == 0 && strafe == 0) {
            // No movement input -- keep vertical momentum, zero horizontal
            Vec3 current = bot.getDeltaMovement();
            Vec3 verticalOnly = new Vec3(0, current.y, 0);
            bot.setDeltaMovement(verticalOnly);
            bot.move(MoverType.SELF, verticalOnly);
            return;
        }

        float yawRad = (float) Math.toRadians(bot.getYRot());
        float speed = bot.isSprinting() ? SPRINT_SPEED : WALK_SPEED;

        double dx = (-Math.sin(yawRad) * forward + Math.cos(yawRad) * strafe) * speed;
        double dz = (Math.cos(yawRad) * forward + Math.sin(yawRad) * strafe) * speed;

        Vec3 current = bot.getDeltaMovement();
        Vec3 movement = new Vec3(dx, current.y, dz);
        bot.setDeltaMovement(movement);
        // ServerPlayer movement is packet-driven; setDeltaMovement alone won't
        // change position. Explicitly apply the displacement with collision.
        bot.move(MoverType.SELF, movement);
    }

    // ------------------------------------------------------------------
    //  Jump (action 4)
    // ------------------------------------------------------------------

    private void applyJump(ServerPlayer bot, int[] actions) {
        if (actions[ACT_JUMP] == 1 && bot.onGround()) {
            bot.jumpFromGround();
        }
    }

    // ------------------------------------------------------------------
    //  Sneak (action 5)
    // ------------------------------------------------------------------

    private void applySneak(ServerPlayer bot, int[] actions) {
        bot.setShiftKeyDown(actions[ACT_SNEAK] == 1);
    }

    // ------------------------------------------------------------------
    //  Sprint (action 6)
    // ------------------------------------------------------------------

    private void applySprint(ServerPlayer bot, int[] actions) {
        bot.setSprinting(actions[ACT_SPRINT] == 1);
    }

    // ------------------------------------------------------------------
    //  Sprint Reset (action 12) -- same-tick toggle for KB boost
    // ------------------------------------------------------------------

    private void applySprintReset(ServerPlayer bot, int[] actions) {
        if (actions[ACT_SPRINT_RESET] == 1) {
            bot.setSprinting(false);
            bot.setSprinting(true);
        }
    }

    // ------------------------------------------------------------------
    //  Attack (action 7)
    // ------------------------------------------------------------------

    private void applyAttack(ServerPlayer bot, int[] actions,
                             @Nullable LivingEntity target) {
        if (actions[ACT_ATTACK] != 1 || target == null || !target.isAlive()) return;

        // Range check: vanilla player reach is ~3 blocks
        double dist = bot.distanceTo(target);
        if (dist > MELEE_REACH) return;

        // Angle check: bot must be roughly facing the target
        double dx = target.getX() - bot.getX();
        double dz = target.getZ() - bot.getZ();
        double horizDist = Math.sqrt(dx * dx + dz * dz);
        if (horizDist > 0.01) {
            float yawRad = (float) Math.toRadians(bot.getYRot());
            double lookX = -Math.sin(yawRad);
            double lookZ = Math.cos(yawRad);
            double dot = (lookX * dx + lookZ * dz) / horizDist;
            if (dot < MELEE_ANGLE_COS) return; // target is outside the ~60° cone
        }

        bot.attack(target); // vanilla damage calc, knockback, crits
    }

    // ------------------------------------------------------------------
    //  Block (action 8)
    // ------------------------------------------------------------------

    private void applyBlock(ServerPlayer bot, int[] actions) {
        if (actions[ACT_BLOCK] == 1) {
            // Try off-hand first (shield), fall back to main hand (1.8 sword block)
            ItemStack offHand = bot.getOffhandItem();
            if (offHand.is(Items.SHIELD)) {
                bot.startUsingItem(InteractionHand.OFF_HAND);
            } else {
                // 1.8-style: any sword in main hand can block
                ItemStack mainHand = bot.getMainHandItem();
                if (isSword(mainHand)) {
                    bot.startUsingItem(InteractionHand.MAIN_HAND);
                }
            }
        } else {
            // Release block if we were blocking
            if (bot.isUsingItem() && isBlockingItem(bot.getUseItem())) {
                bot.stopUsingItem();
            }
        }
    }

    // ------------------------------------------------------------------
    //  Eat Golden Apple (action 9)
    // ------------------------------------------------------------------

    private void applyEatGap(ServerPlayer bot, int[] actions) {
        if (actions[ACT_EAT_GAP] != 1) return;
        EatingState es = eatingStates.computeIfAbsent(bot.getId(), k -> new EatingState());
        if (es.ticksRemaining > 0) return; // already eating

        // Find a golden apple in inventory
        int slot = findItemSlot(bot, stack ->
                stack.is(Items.GOLDEN_APPLE) || stack.is(Items.ENCHANTED_GOLDEN_APPLE));
        if (slot == -1) return;

        // Move the apple to main hand and start using
        swapToSlot(bot, slot);
        bot.startUsingItem(InteractionHand.MAIN_HAND);
        es.ticksRemaining = 32; // golden apple use time
        es.hand = InteractionHand.MAIN_HAND;
    }

    /** Tick down eating state; stop using when done. */
    private void tickEating(ServerPlayer bot) {
        EatingState es = eatingStates.get(bot.getId());
        if (es == null || es.ticksRemaining <= 0) return;
        es.ticksRemaining--;
        if (es.ticksRemaining == 0) {
            // Let vanilla finish the use (completeUsingItem is called automatically
            // by the server when useItemRemainingTicks reaches 0).
            es.hand = null;
        }
    }

    // ------------------------------------------------------------------
    //  Throw Splash Health Potion (action 10)
    // ------------------------------------------------------------------

    private void applyThrowPot(ServerPlayer bot, int[] actions) {
        if (actions[ACT_THROW_POT] != 1) return;

        ServerLevel level = (ServerLevel) bot.level();
        int slot = findItemSlot(bot, stack ->
                stack.is(Items.SPLASH_POTION) && isHealingPotion(stack));
        if (slot == -1) return;

        ItemStack potStack = bot.getInventory().getItem(slot);

        // Create the thrown splash potion entity via EntityType constructor
        ThrownSplashPotion potion = new ThrownSplashPotion(EntityType.SPLASH_POTION, level);
        potion.setPos(bot.getX(), bot.getEyeY() - 0.1, bot.getZ());
        potion.setItem(potStack.copy());

        // Launch using vanilla shootFromRotation(Entity, pitch, yaw, pitchOffset, speed, inaccuracy)
        potion.shootFromRotation(bot, bot.getXRot(), bot.getYRot(), -20.0f, 0.5f, 1.0f);

        level.addFreshEntity(potion);

        // Consume the item
        potStack.shrink(1);
        if (potStack.isEmpty()) {
            bot.getInventory().setItem(slot, ItemStack.EMPTY);
        }
    }

    // ------------------------------------------------------------------
    //  Throw Ender Pearl (action 11)
    // ------------------------------------------------------------------

    private void applyThrowPearl(ServerPlayer bot, int[] actions) {
        if (actions[ACT_THROW_PEARL] != 1) return;

        ServerLevel level = (ServerLevel) bot.level();
        int slot = findItemSlot(bot, stack -> stack.is(Items.ENDER_PEARL));
        if (slot == -1) return;

        ItemStack pearlStack = bot.getInventory().getItem(slot);

        // Create via EntityType constructor
        ThrownEnderpearl pearl = new ThrownEnderpearl(EntityType.ENDER_PEARL, level);
        pearl.setPos(bot.getX(), bot.getEyeY() - 0.1, bot.getZ());

        // Launch using vanilla shootFromRotation(Entity, pitch, yaw, pitchOffset, speed, inaccuracy)
        pearl.shootFromRotation(bot, bot.getXRot(), bot.getYRot(), 0.0f, 1.5f, 1.0f);

        level.addFreshEntity(pearl);

        // Apply vanilla pearl cooldown (20 ticks = 1 second)
        bot.getCooldowns().addCooldown(pearlStack, 20);

        // Consume the item
        pearlStack.shrink(1);
        if (pearlStack.isEmpty()) {
            bot.getInventory().setItem(slot, ItemStack.EMPTY);
        }
    }

    // ------------------------------------------------------------------
    //  Sigil abilities (actions 14-25)
    // ------------------------------------------------------------------

    private void applySigils(ServerPlayer bot, int[] actions) {
        if (sigilsApi == null) return;

        Player bukkitPlayer = getBukkitPlayer(bot);
        if (bukkitPlayer == null) return;

        for (int i = 0; i < NUM_SIGIL_SLOTS; i++) {
            if (actions[ACT_SIGIL_0 + i] == 1) {
                sigilsApi.activateAbility(bukkitPlayer, i);
            }
        }
    }

    // ------------------------------------------------------------------
    //  Camera intent (actions 26-29)
    //  Priority: FACE_TARGET > FACE_AWAY > LOOK_DOWN_SELF > FACE_MOVEMENT
    // ------------------------------------------------------------------

    private void applyLookIntent(ServerPlayer bot, int[] actions,
                                  @Nullable LivingEntity target) {
        if (actions[ACT_FACE_TARGET] == 1 && target != null && target.isAlive()) {
            lookAt(bot, target.getX(), target.getEyeY(), target.getZ());
        } else if (actions[ACT_FACE_AWAY] == 1 && target != null && target.isAlive()) {
            // Look directly opposite of target
            float yawToTarget = getYawToward(bot, target.getX(), target.getZ());
            bot.setYRot(yawToTarget + 180f);
            bot.setXRot(0f); // level pitch for running
            bot.setYHeadRot(bot.getYRot());
        } else if (actions[ACT_LOOK_DOWN_SELF] == 1) {
            // Look at own feet for self-potting
            bot.setXRot(89f);
            bot.setYHeadRot(bot.getYRot());
        } else if (actions[ACT_FACE_MOVEMENT] == 1) {
            // Face movement direction
            Vec3 vel = bot.getDeltaMovement();
            if (vel.x * vel.x + vel.z * vel.z > 0.001) {
                float moveYaw = (float) (Math.toDegrees(Math.atan2(-vel.x, vel.z)));
                bot.setYRot(moveYaw);
                bot.setXRot(0f);
                bot.setYHeadRot(moveYaw);
            }
        }
        // If no intent is active, hold current look direction
    }

    private void lookAt(ServerPlayer bot, double x, double y, double z) {
        double dx = x - bot.getX();
        double dy = y - bot.getEyeY();
        double dz = z - bot.getZ();
        double dist = Math.sqrt(dx * dx + dz * dz);

        float yaw = (float) Math.toDegrees(Math.atan2(-dx, dz));
        float pitch = (float) -Math.toDegrees(Math.atan2(dy, dist));

        bot.setYRot(yaw);
        bot.setXRot(Math.clamp(pitch, -90f, 90f));
        bot.setYHeadRot(yaw);
    }

    private float getYawToward(ServerPlayer bot, double x, double z) {
        double dx = x - bot.getX();
        double dz = z - bot.getZ();
        return (float) Math.toDegrees(Math.atan2(-dx, dz));
    }

    // ------------------------------------------------------------------
    //  Target selection (actions 30-34)
    // ------------------------------------------------------------------

    private @Nullable LivingEntity resolveTarget(int[] actions,
                                                 @Nullable LivingEntity current,
                                                 List<LivingEntity> nearby) {
        for (int i = 0; i < NUM_TARGET_SLOTS; i++) {
            if (actions[ACT_TARGET_0 + i] == 1 && i < nearby.size()) {
                LivingEntity candidate = nearby.get(i);
                if (candidate.isAlive()) {
                    return candidate;
                }
            }
        }
        return current; // keep existing target
    }

    // ------------------------------------------------------------------
    //  Action masking
    // ------------------------------------------------------------------

    /**
     * Build a mask of currently valid actions.
     * 1.0 = valid, 0.0 = masked (should be zeroed out before softmax).
     *
     * @param bot the bot player to inspect
     * @return float[35] action mask
     */
    public float[] buildActionMask(ServerPlayer bot) {
        float[] mask = new float[NUM_ACTIONS];

        // Movement, sprint, sneak, look intents, sprint-reset are always valid
        mask[ACT_FORWARD] = 1f;
        mask[ACT_BACKWARD] = 1f;
        mask[ACT_STRAFE_LEFT] = 1f;
        mask[ACT_STRAFE_RIGHT] = 1f;
        mask[ACT_SNEAK] = 1f;
        mask[ACT_SPRINT] = 1f;
        mask[ACT_SPRINT_RESET] = 1f;

        mask[ACT_FACE_TARGET] = 1f;
        mask[ACT_FACE_AWAY] = 1f;
        mask[ACT_LOOK_DOWN_SELF] = 1f;
        mask[ACT_FACE_MOVEMENT] = 1f;

        // Jump: only if on ground
        mask[ACT_JUMP] = bot.onGround() ? 1f : 0f;

        // Attack: always available (target check is at execution time)
        mask[ACT_ATTACK] = 1f;

        // Block: need shield in off-hand or sword in main hand
        ItemStack offHand = bot.getOffhandItem();
        ItemStack mainHand = bot.getMainHandItem();
        mask[ACT_BLOCK] = (offHand.is(Items.SHIELD) || isSword(mainHand)) ? 1f : 0f;

        // Eat golden apple: must have one in inventory
        mask[ACT_EAT_GAP] = hasItem(bot, stack ->
                stack.is(Items.GOLDEN_APPLE) || stack.is(Items.ENCHANTED_GOLDEN_APPLE)) ? 1f : 0f;

        // Throw splash health pot
        mask[ACT_THROW_POT] = hasItem(bot, stack ->
                stack.is(Items.SPLASH_POTION) && isHealingPotion(stack)) ? 1f : 0f;

        // Throw ender pearl (must have item AND not on cooldown)
        boolean hasPearl = hasItem(bot, stack -> stack.is(Items.ENDER_PEARL));
        boolean pearlOnCooldown = bot.getCooldowns().isOnCooldown(new ItemStack(Items.ENDER_PEARL));
        mask[ACT_THROW_PEARL] = (hasPearl && !pearlOnCooldown) ? 1f : 0f;

        // Swap weapon: always masked
        mask[ACT_SWAP_WEAPON] = 0f;

        // Sigil slots
        Player bukkitPlayer = getBukkitPlayer(bot);
        for (int i = 0; i < NUM_SIGIL_SLOTS; i++) {
            if (sigilsApi == null || bukkitPlayer == null) {
                mask[ACT_SIGIL_0 + i] = 0f;
            } else if (sigilCooldowns != null) {
                mask[ACT_SIGIL_0 + i] = sigilCooldowns.isReady(bukkitPlayer, i) ? 1f : 0f;
            } else {
                // No cooldown query available; optimistically unmask
                mask[ACT_SIGIL_0 + i] = 1f;
            }
        }

        // Target selection: unmask slots that have a live entity
        // (we don't have the nearby list here, so unmask all 5;
        //  the caller can refine with nearby.size())
        for (int i = 0; i < NUM_TARGET_SLOTS; i++) {
            mask[ACT_TARGET_0 + i] = 1f;
        }

        return mask;
    }

    /**
     * Overload that also masks target slots based on actual nearby count.
     */
    public float[] buildActionMask(ServerPlayer bot, int nearbyCount) {
        float[] mask = buildActionMask(bot);
        for (int i = 0; i < NUM_TARGET_SLOTS; i++) {
            if (i >= nearbyCount) {
                mask[ACT_TARGET_0 + i] = 0f;
            }
        }
        return mask;
    }

    // ------------------------------------------------------------------
    //  Inventory helpers
    // ------------------------------------------------------------------

    @FunctionalInterface
    private interface ItemPredicate {
        boolean test(ItemStack stack);
    }

    /**
     * Find the first inventory slot (0-35) matching the predicate, or -1.
     */
    private int findItemSlot(ServerPlayer player, ItemPredicate pred) {
        // Search hotbar first (slots 0-8), then rest of inventory
        for (int i = 0; i < player.getInventory().getContainerSize(); i++) {
            ItemStack stack = player.getInventory().getItem(i);
            if (!stack.isEmpty() && pred.test(stack)) {
                return i;
            }
        }
        return -1;
    }

    private boolean hasItem(ServerPlayer player, ItemPredicate pred) {
        return findItemSlot(player, pred) != -1;
    }

    /**
     * Move the item at the given slot to the player's main hand (selected slot).
     * If the item is already in the hotbar, just select that slot.
     */
    private void swapToSlot(ServerPlayer player, int slot) {
        int selected = player.getBukkitEntity().getInventory().getHeldItemSlot();
        if (slot == selected) return; // already held

        if (slot < 9) {
            // Item is in hotbar, just change selected slot
            player.getBukkitEntity().getInventory().setHeldItemSlot(slot);
        } else {
            // Swap inventory slot with current selected hotbar slot
            ItemStack hotbarStack = player.getInventory().getItem(selected);
            ItemStack targetStack = player.getInventory().getItem(slot);
            player.getInventory().setItem(selected, targetStack);
            player.getInventory().setItem(slot, hotbarStack);
        }
    }

    // ------------------------------------------------------------------
    //  Item classification helpers
    // ------------------------------------------------------------------

    private boolean isSword(ItemStack stack) {
        if (stack.isEmpty()) return false;
        return stack.is(net.minecraft.tags.ItemTags.SWORDS);
    }

    private boolean isBlockingItem(ItemStack stack) {
        if (stack.isEmpty()) return false;
        return stack.is(Items.SHIELD) || isSword(stack);
    }

    private boolean isHealingPotion(ItemStack stack) {
        if (!stack.is(Items.SPLASH_POTION)) return false;
        PotionContents contents = stack.get(DataComponents.POTION_CONTENTS);
        if (contents == null) return false;
        // Check if it's an instant health potion
        return contents.potion().isPresent()
                && (contents.potion().get() == Potions.HEALING
                    || contents.potion().get() == Potions.STRONG_HEALING);
    }

    // ------------------------------------------------------------------
    //  Bukkit bridge
    // ------------------------------------------------------------------

    /**
     * Get the Bukkit Player handle for a ServerPlayer.
     * Returns null if the player has no Bukkit wrapper (shouldn't happen on Paper).
     */
    private @Nullable Player getBukkitPlayer(ServerPlayer nms) {
        try {
            return nms.getBukkitEntity();
        } catch (Exception e) {
            return null;
        }
    }

    // ------------------------------------------------------------------
    //  Debug / logging
    // ------------------------------------------------------------------

    /**
     * Human-readable summary of active actions for logging.
     */
    public static String describeActions(int[] actions) {
        String[] names = {
            "FWD", "BACK", "STRAFE_L", "STRAFE_R", "JUMP", "SNEAK", "SPRINT",
            "ATTACK", "BLOCK", "EAT_GAP", "THROW_POT", "THROW_PEARL",
            "SPRINT_RESET", "SWAP_WPN",
            "SIG0", "SIG1", "SIG2", "SIG3", "SIG4", "SIG5",
            "SIG6", "SIG7", "SIG8", "SIG9", "SIG10", "SIG11",
            "FACE_TGT", "FACE_AWAY", "LOOK_DN_SELF", "FACE_MOVE",
            "TGT0", "TGT1", "TGT2", "TGT3", "TGT4"
        };
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(actions.length, names.length); i++) {
            if (actions[i] == 1) {
                if (sb.length() > 0) sb.append(' ');
                sb.append(names[i]);
            }
        }
        return sb.length() > 0 ? sb.toString() : "IDLE";
    }
}
