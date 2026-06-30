package com.minimalai.ai;

import net.minecraft.core.Holder;
import net.minecraft.core.component.DataComponents;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.effect.MobEffects;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.ai.attributes.Attribute;
import net.minecraft.world.entity.ai.attributes.Attributes;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.item.Items;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import com.minimalai.integration.ArcaneSigilsAPI;
import org.jetbrains.annotations.Nullable;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static com.minimalai.ai.ObservationSpace.*;

/**
 * Maps live Minecraft server state to v2 observation tensors that match
 * the training environment (training/combat_sim/env.py).
 *
 * Uses Mojang-mapped NMS types provided by paperweight-userdev.
 */
public class ObservationBuilder {

    // Arena center coordinates (from config)
    private final double arenaCenterX;
    private final double arenaCenterZ;

    // Episode settings
    private int maxEpisodeTicks = 1800;
    private int episodeStartTick = 0;

    // Per-bot combat tracking: botName -> CombatState
    private final Map<String, CombatState> combatStates = new ConcurrentHashMap<>();

    // Optional ArcaneSigils API (direct Java API via reflection bridge)
    private final boolean sigilsAvailable;
    private final @Nullable ArcaneSigilsAPI sigilsApi;

    /**
     * Per-bot mutable combat state, updated from external event callbacks.
     */
    private static class CombatState {
        float damageDealtThisTick = 0f;
        float damageTakenThisTick = 0f;
        int comboCounter = 0;
        long lastHitTick = -100;
        float cpsEstimate = 10f;

        // CPS tracking: count hits in a rolling 1-second window
        private final long[] hitTimestamps = new long[20];
        private int hitIndex = 0;

        void recordHit(long tick) {
            hitTimestamps[hitIndex % hitTimestamps.length] = tick;
            hitIndex++;
            // Count hits in the last 20 ticks (1 second)
            int count = 0;
            for (long t : hitTimestamps) {
                if (tick - t <= 20 && t > 0) count++;
            }
            cpsEstimate = count;
        }

        void tickReset() {
            damageDealtThisTick = 0f;
            damageTakenThisTick = 0f;
        }
    }

    /**
     * Observation record holding all 6 tensor arrays that the neural network consumes.
     */
    public record Observation(
        float[] selfState,       // [SELF_STATE_DIM]
        float[] entityFeatures,  // [MAX_ENTITIES * ENTITY_FEATURE_DIM]
        float[] entityMask,      // [MAX_ENTITIES]
        float[] combatCtx,       // [COMBAT_CTX_DIM]
        float[] sigilState,      // [SIGIL_STATE_DIM]
        float[] envState         // [ENV_STATE_DIM]
    ) {}

    public ObservationBuilder(double arenaCenterX, double arenaCenterZ) {
        this(arenaCenterX, arenaCenterZ, null);
    }

    public ObservationBuilder(double arenaCenterX, double arenaCenterZ,
                              @Nullable ArcaneSigilsAPI sigilsApi) {
        this.arenaCenterX = arenaCenterX;
        this.arenaCenterZ = arenaCenterZ;
        this.sigilsApi = sigilsApi;
        this.sigilsAvailable = sigilsApi != null && sigilsApi.isAvailable();
    }

    // ------------------------------------------------------------------
    //  External event callbacks (called from damage listeners)
    // ------------------------------------------------------------------

    public void onDamageDealt(String botName, float amount) {
        CombatState s = combatStates.computeIfAbsent(botName, k -> new CombatState());
        s.damageDealtThisTick += amount;
        s.comboCounter++;
        s.lastHitTick = getCurrentTick();
        s.recordHit(getCurrentTick());
    }

    public void onDamageTaken(String botName, float amount) {
        CombatState s = combatStates.computeIfAbsent(botName, k -> new CombatState());
        s.damageTakenThisTick += amount;
        s.comboCounter = 0; // reset combo on being hit
    }

    /**
     * Called every server tick per bot. Resets per-tick accumulators.
     */
    public void onTick(String botName) {
        CombatState s = combatStates.get(botName);
        if (s != null) {
            s.tickReset();
        }
    }

    /** Get the current tick using the server's global tick counter. */
    private int getCurrentTick() {
        return org.bukkit.Bukkit.getCurrentTick();
    }

    public void setMaxEpisodeTicks(int ticks) {
        this.maxEpisodeTicks = ticks;
    }

    public void setEpisodeStartTick(int tick) {
        this.episodeStartTick = tick;
    }

    public void resetCombatState(String botName) {
        combatStates.remove(botName);
    }

    // ------------------------------------------------------------------
    //  Main observation builder
    // ------------------------------------------------------------------

    /**
     * Build the full observation for a bot, optionally relative to a target.
     *
     * @param bot    the ServerPlayer being controlled by the AI
     * @param target the primary combat target (may be null)
     * @return Observation containing all 6 float arrays
     */
    public Observation build(ServerPlayer bot, ServerPlayer target) {
        float[] selfState = buildSelfState(bot);
        float[] entityFeatures = new float[MAX_ENTITIES * ENTITY_FEATURE_DIM];
        float[] entityMask = new float[MAX_ENTITIES];
        buildEntityFeatures(bot, target, entityFeatures, entityMask);
        float[] combatCtx = buildCombatCtx(bot, target);
        float[] sigilState = buildSigilState(bot);
        float[] envState = buildEnvState(bot, target);
        return new Observation(selfState, entityFeatures, entityMask, combatCtx, sigilState, envState);
    }

    // ------------------------------------------------------------------
    //  self_state[38]
    // ------------------------------------------------------------------

    private float[] buildSelfState(ServerPlayer p) {
        float[] s = new float[SELF_STATE_DIM];

        // [0] health / 20
        s[0] = p.getHealth() / 20f;
        // [1] maxHealth / 20
        s[1] = (float) p.getMaxHealth() / 20f;
        // [2] absorption / 20
        s[2] = p.getAbsorptionAmount() / 20f;
        // [3] armor / 20
        s[3] = (float) p.getArmorValue() / 20f;
        // [4] armorToughness / 20
        s[4] = (float) getAttributeSafe(p, Attributes.ARMOR_TOUGHNESS) / 20f;
        // [5-7] velocity
        Vec3 vel = p.getDeltaMovement();
        s[5] = (float) vel.x;
        s[6] = (float) vel.y;
        s[7] = (float) vel.z;
        // [8] yaw / 180
        s[8] = p.getYRot() / 180f;
        // [9] pitch / 90
        s[9] = p.getXRot() / 90f;
        // [10] onGround
        s[10] = p.onGround() ? 1f : 0f;
        // [11] isSprinting
        s[11] = p.isSprinting() ? 1f : 0f;
        // [12] isSneaking
        s[12] = p.isShiftKeyDown() ? 1f : 0f;
        // [13] isBlocking (shield/sword)
        s[13] = p.isBlocking() ? 1f : 0f;
        // [14] isUsingItem (eating)
        s[14] = p.isUsingItem() ? 1f : 0f;
        // [15] fallDistance / 10
        s[15] = (float) p.fallDistance / 10f;
        // [16-18] weapon type one-hot (sword=0, axe=1, other=2)
        int wt = getWeaponType(p);
        s[16] = wt == 0 ? 1f : 0f;
        s[17] = wt == 1 ? 1f : 0f;
        s[18] = wt == 2 ? 1f : 0f;
        // [19] attack damage / 20
        s[19] = (float) getAttributeSafe(p, Attributes.ATTACK_DAMAGE) / 20f;
        // [20] golden apples / 64
        s[20] = countItem(p, Items.GOLDEN_APPLE) / 64f;
        // [21] health pots / 64
        s[21] = countItem(p, Items.SPLASH_POTION) / 64f;
        // [22] ender pearls / 16
        s[22] = countItem(p, Items.ENDER_PEARL) / 16f;
        // [23] totems / 1
        s[23] = Math.min(countItem(p, Items.TOTEM_OF_UNDYING), 1);
        // [24-32] status effect flags
        s[24] = p.hasEffect(MobEffects.SPEED) ? 1f : 0f;
        s[25] = p.hasEffect(MobEffects.STRENGTH) ? 1f : 0f;
        s[26] = p.hasEffect(MobEffects.RESISTANCE) ? 1f : 0f;
        s[27] = p.hasEffect(MobEffects.FIRE_RESISTANCE) ? 1f : 0f;
        s[28] = p.hasEffect(MobEffects.REGENERATION) ? 1f : 0f;
        s[29] = p.hasEffect(MobEffects.POISON) ? 1f : 0f;
        s[30] = p.hasEffect(MobEffects.WITHER) ? 1f : 0f;
        s[31] = p.hasEffect(MobEffects.SLOWNESS) ? 1f : 0f;
        s[32] = p.hasEffect(MobEffects.WEAKNESS) ? 1f : 0f;
        // [33] hurtTime / 10 (invulnerability frames)
        s[33] = p.hurtTime / 10f;
        // [34-37] ArcaneSigils state
        if (sigilsAvailable) {
            float[] sigils = readSigilCombatState(p);
            s[34] = sigils[0]; // damageAmp
            s[35] = sigils[1]; // damageReduction
            s[36] = sigils[2]; // kbCharges / 10
            s[37] = sigils[3]; // invulnHits / 10
        }
        // else already zeros

        return s;
    }

    // ------------------------------------------------------------------
    //  entity_features[MAX_ENTITIES x 24] + entity_mask[MAX_ENTITIES]
    // ------------------------------------------------------------------

    private void buildEntityFeatures(ServerPlayer bot, ServerPlayer target,
                                     float[] features, float[] mask) {
        List<LivingEntity> nearby = getNearbyEntities(bot, 30.0);
        int slot = 0;

        for (LivingEntity entity : nearby) {
            if (slot >= MAX_ENTITIES) break;

            int base = slot * ENTITY_FEATURE_DIM;

            // [0] alliance
            features[base] = getAlliance(bot, entity);

            // [1-3] relative position / 30
            double dx = entity.getX() - bot.getX();
            double dy = entity.getY() - bot.getY();
            double dz = entity.getZ() - bot.getZ();
            features[base + 1] = (float) (dx / 30.0);
            features[base + 2] = (float) (dy / 30.0);
            features[base + 3] = (float) (dz / 30.0);

            // [4] distance / 30
            double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            features[base + 4] = (float) (dist / 30.0);

            // [5] health / 20
            features[base + 5] = entity.getHealth() / 20f;

            // [6] maxHealth / 20
            features[base + 6] = (float) entity.getMaxHealth() / 20f;

            // [7] armor / 20
            features[base + 7] = (float) entity.getArmorValue() / 20f;

            // [8-10] velocity
            Vec3 eVel = entity.getDeltaMovement();
            features[base + 8] = (float) eVel.x;
            features[base + 9] = (float) eVel.y;
            features[base + 10] = (float) eVel.z;

            // [11] yaw / 180
            features[base + 11] = entity.getYRot() / 180f;

            // [12] onGround
            features[base + 12] = entity.onGround() ? 1f : 0f;

            // [13] isSprinting
            features[base + 13] = entity.isSprinting() ? 1f : 0f;

            // [14] isBlocking
            features[base + 14] = entity.isBlocking() ? 1f : 0f;

            // [15] isUsingItem
            features[base + 15] = entity.isUsingItem() ? 1f : 0f;

            // [16-18] weapon type one-hot
            int ewt = getWeaponTypeForEntity(entity);
            features[base + 16] = ewt == 0 ? 1f : 0f;
            features[base + 17] = ewt == 1 ? 1f : 0f;
            features[base + 18] = ewt == 2 ? 1f : 0f;

            // [19] attack damage / 20
            features[base + 19] = (float) getEntityAttackDamage(entity) / 20f;

            // [20] hurtTime / 10
            features[base + 20] = entity.hurtTime / 10f;

            // [21] is current target
            features[base + 21] = (target != null && entity.getId() == target.getId()) ? 1f : 0f;

            // [22] has pharaoh mark (ArcaneSigils)
            if (sigilsAvailable) {
                features[base + 22] = readPharaohMark(bot, entity) ? 1f : 0f;
            }

            // [23] distance to bot's target / 30
            if (target != null) {
                double tDist = entity.distanceTo(target);
                features[base + 23] = (float) (tDist / 30.0);
            }

            mask[slot] = 1f;
            slot++;
        }
        // Remaining slots stay zero (padding), mask stays 0
    }

    // ------------------------------------------------------------------
    //  combat_ctx[26]
    // ------------------------------------------------------------------

    private float[] buildCombatCtx(ServerPlayer bot, ServerPlayer target) {
        float[] ctx = new float[COMBAT_CTX_DIM];
        CombatState cs = combatStates.computeIfAbsent(bot.getScoreboardName(), k -> new CombatState());

        // [0-2] weapon type one-hot
        int wt = getWeaponType(bot);
        ctx[0] = wt == 0 ? 1f : 0f;
        ctx[1] = wt == 1 ? 1f : 0f;
        ctx[2] = wt == 2 ? 1f : 0f;

        // [3] golden apples / 64
        ctx[3] = countItem(bot, Items.GOLDEN_APPLE) / 64f;
        // [4] health pots / 64
        ctx[4] = countItem(bot, Items.SPLASH_POTION) / 64f;
        // [5] ender pearls / 16
        ctx[5] = countItem(bot, Items.ENDER_PEARL) / 16f;
        // [6] totems / 1
        ctx[6] = Math.min(countItem(bot, Items.TOTEM_OF_UNDYING), 1);
        // [7] damage dealt this tick / 20
        ctx[7] = cs.damageDealtThisTick / 20f;
        // [8] damage taken this tick / 20
        ctx[8] = cs.damageTakenThisTick / 20f;
        // [9] combo counter / 10
        ctx[9] = cs.comboCounter / 10f;
        // [10] time since last hit / 100
        ctx[10] = Math.min((getCurrentTick() - cs.lastHitTick) / 100f, 1f);
        // [11] CPS estimate / 20
        ctx[11] = cs.cpsEstimate / 20f;
        // [12] sprint state
        ctx[12] = bot.isSprinting() ? 1f : 0f;
        // [13] block state
        ctx[13] = bot.isBlocking() ? 1f : 0f;
        // [14] eating state
        boolean isEating = bot.isUsingItem() && bot.getUseItem().has(DataComponents.FOOD);
        ctx[14] = isEating ? 1f : 0f;
        // [15] eating ticks remaining / 32
        ctx[15] = isEating ? bot.getUseItemRemainingTicks() / 32f : 0f;
        // [16-24] status effects
        ctx[16] = bot.hasEffect(MobEffects.SPEED) ? 1f : 0f;
        ctx[17] = bot.hasEffect(MobEffects.STRENGTH) ? 1f : 0f;
        ctx[18] = bot.hasEffect(MobEffects.RESISTANCE) ? 1f : 0f;
        ctx[19] = bot.hasEffect(MobEffects.FIRE_RESISTANCE) ? 1f : 0f;
        ctx[20] = bot.hasEffect(MobEffects.REGENERATION) ? 1f : 0f;
        ctx[21] = bot.hasEffect(MobEffects.POISON) ? 1f : 0f;
        ctx[22] = bot.hasEffect(MobEffects.WITHER) ? 1f : 0f;
        ctx[23] = bot.hasEffect(MobEffects.SLOWNESS) ? 1f : 0f;
        ctx[24] = bot.hasEffect(MobEffects.WEAKNESS) ? 1f : 0f;
        // [25] pearl cooldown / 300
        // Ender pearl cooldown is tracked by item cooldown on the player
        ItemStack pearlStack = new ItemStack(Items.ENDER_PEARL);
        ctx[25] = bot.getCooldowns().isOnCooldown(pearlStack)
            ? bot.getCooldowns().getCooldownPercent(pearlStack, 0f)
            : 0f;

        return ctx;
    }

    // ------------------------------------------------------------------
    //  sigil_state[48] = 12 slots x 4 values
    // ------------------------------------------------------------------

    private float[] buildSigilState(ServerPlayer bot) {
        float[] state = new float[SIGIL_STATE_DIM];

        if (!sigilsAvailable || sigilsApi == null) return state; // all zeros

        org.bukkit.entity.Player bukkitPlayer = bot.getBukkitEntity();

        for (int i = 0; i < NUM_SIGIL_SLOTS; i++) {
            int base = i * 4;

            // [0] ready ? 1 : 0
            state[base] = sigilsApi.isSigilReady(bukkitPlayer, i) ? 1f : 0f;

            // [1] cooldown progress (0 = ready, 1 = full cd)
            state[base + 1] = (float) sigilsApi.getCooldownProgress(bukkitPlayer, i);

            // [2] tier / 3
            int tier = sigilsApi.getTier(bukkitPlayer, i);
            state[base + 2] = tier / 3f;

            // [3] cooldown remaining seconds / 30 (normalized)
            double cdRemaining = sigilsApi.getCooldownRemaining(bukkitPlayer, i);
            state[base + 3] = Math.min((float) (cdRemaining / 30.0), 1f);
        }

        return state;
    }

    // ------------------------------------------------------------------
    //  env_state[8]
    // ------------------------------------------------------------------

    private float[] buildEnvState(ServerPlayer bot, ServerPlayer target) {
        float[] env = new float[ENV_STATE_DIM];

        // [0-1] position relative to arena center
        env[0] = (float) ((bot.getX() - arenaCenterX) / 100.0);
        env[1] = (float) ((bot.getZ() - arenaCenterZ) / 100.0);

        // [2] y position / 320
        env[2] = (float) (bot.getY() / 320.0);

        // [3] height diff to target / 30
        if (target != null && target.isAlive()) {
            env[3] = (float) ((bot.getY() - target.getY()) / 30.0);
        }

        // [4] in attack range (distance < 3.5)
        if (target != null && target.isAlive()) {
            double dist = bot.distanceTo(target);
            env[4] = dist < 3.5 ? 1f : 0f;
        }

        // [5] onGround
        env[5] = bot.onGround() ? 1f : 0f;

        // [6] alive enemies ratio
        ServerLevel level = (ServerLevel) bot.level();
        List<ServerPlayer> allPlayers = level.players();
        int totalEnemies = 0;
        int aliveEnemies = 0;
        for (ServerPlayer other : allPlayers) {
            if (other.getId() == bot.getId()) continue;
            if (getAlliance(bot, other) < 0) {
                totalEnemies++;
                if (other.isAlive()) {
                    aliveEnemies++;
                }
            }
        }
        env[6] = totalEnemies > 0 ? (float) aliveEnemies / totalEnemies : 0f;

        // [7] episode progress
        int episodeTick = getCurrentTick() - episodeStartTick;
        env[7] = maxEpisodeTicks > 0 ? (float) episodeTick / maxEpisodeTicks : 0f;

        return env;
    }

    // ------------------------------------------------------------------
    //  Helper: weapon type (0=sword, 1=axe, 2=other)
    // ------------------------------------------------------------------

    private static int getWeaponType(ServerPlayer p) {
        return classifyWeapon(p.getMainHandItem());
    }

    private static int getWeaponTypeForEntity(LivingEntity entity) {
        return classifyWeapon(entity.getMainHandItem());
    }

    /**
     * Classify a held item as sword (0), axe (1), or other (2)
     * using registry key inspection since SwordItem/AxeItem classes
     * no longer exist in 1.21.10.
     */
    private static int classifyWeapon(ItemStack stack) {
        if (stack.isEmpty()) return 2;
        ResourceLocation key = BuiltInRegistries.ITEM.getKey(stack.getItem());
        String path = key.getPath();
        if (path.endsWith("_sword") || path.equals("sword")) return 0;
        if (path.endsWith("_axe") || path.equals("axe")) return 1;
        return 2;
    }

    // ------------------------------------------------------------------
    //  Helper: count item across entire inventory
    // ------------------------------------------------------------------

    private static int countItem(ServerPlayer p, Item targetItem) {
        int count = 0;
        for (int i = 0; i < p.getInventory().getContainerSize(); i++) {
            ItemStack stack = p.getInventory().getItem(i);
            if (stack.is(targetItem)) {
                count += stack.getCount();
            }
        }
        // Check offhand
        ItemStack offhand = p.getOffhandItem();
        if (offhand.is(targetItem)) {
            count += offhand.getCount();
        }
        return count;
    }

    // ------------------------------------------------------------------
    //  Helper: nearby living entities sorted by distance
    // ------------------------------------------------------------------

    private List<LivingEntity> getNearbyEntities(ServerPlayer bot, double range) {
        AABB box = bot.getBoundingBox().inflate(range);
        ServerLevel level = (ServerLevel) bot.level();
        List<LivingEntity> entities = level.getEntitiesOfClass(
            LivingEntity.class, box,
            e -> e != bot && e.isAlive() && !e.isSpectator()
        );
        entities.sort(Comparator.comparingDouble(e -> e.distanceToSqr(bot)));
        return entities;
    }

    // ------------------------------------------------------------------
    //  Helper: alliance (-1 enemy, 0 neutral, 1 ally)
    // ------------------------------------------------------------------

    private float getAlliance(ServerPlayer bot, LivingEntity entity) {
        if (!(entity instanceof Player otherPlayer)) {
            // Non-player entities default to neutral
            return 0f;
        }
        // Same team = ally, different team or no team = enemy
        if (bot.getTeam() != null && bot.getTeam().isAlliedTo(otherPlayer.getTeam())) {
            return 1f;
        }
        if (bot.getTeam() == null && otherPlayer.getTeam() == null) {
            // No teams: treat other players as enemies
            return -1f;
        }
        return -1f;
    }

    // ------------------------------------------------------------------
    //  Helper: safe attribute read
    // ------------------------------------------------------------------

    private static double getAttributeSafe(LivingEntity entity, Holder<Attribute> attribute) {
        var inst = entity.getAttribute(attribute);
        return inst != null ? inst.getValue() : 0.0;
    }

    private static double getEntityAttackDamage(LivingEntity entity) {
        var inst = entity.getAttribute(Attributes.ATTACK_DAMAGE);
        if (inst != null) return inst.getValue();
        // Fallback: estimate from held item
        int wt = classifyWeapon(entity.getMainHandItem());
        if (wt == 0) return 7.0; // sword
        if (wt == 1) return 9.0; // axe
        return 1.0;
    }

    // ------------------------------------------------------------------
    //  ArcaneSigils integration via Bukkit scoreboard
    // ------------------------------------------------------------------

    /**
     * Read sigil combat state from API.
     * Returns [damageAmp, damageReduction, kbCharges/10, invulnHits/10].
     */
    private float[] readSigilCombatState(ServerPlayer nmsPlayer) {
        float[] result = new float[4];
        if (sigilsApi == null) return result;

        org.bukkit.entity.Player bukkitPlayer = nmsPlayer.getBukkitEntity();

        result[0] = (float) sigilsApi.getDamageAmplifier(bukkitPlayer);
        result[1] = (float) sigilsApi.getDamageReduction(bukkitPlayer);
        result[2] = sigilsApi.getKingsBraceCharges(bukkitPlayer) / 10f;
        result[3] = sigilsApi.getInvulnHits(bukkitPlayer) / 10f;
        return result;
    }

    /**
     * Check if an entity has a Pharaoh mark placed by the bot.
     */
    private boolean readPharaohMark(ServerPlayer bot, LivingEntity entity) {
        if (sigilsApi == null) return false;
        if (!(entity instanceof ServerPlayer targetNms)) return false;

        org.bukkit.entity.Player botBukkit = bot.getBukkitEntity();
        org.bukkit.entity.Player targetBukkit = targetNms.getBukkitEntity();

        return sigilsApi.isMarked(targetBukkit, botBukkit);
    }
}
