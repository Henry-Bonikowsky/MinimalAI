package com.minimalai.training;

import com.minimalai.ai.ObservationBuilder;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.jetbrains.annotations.Nullable;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * Listens to Bukkit events and computes per-tick rewards for each registered bot.
 * Reward formula mirrors {@code training/combat_sim/env.py._calculate_reward()}.
 */
public class RewardComputer implements Listener {

    private static final Logger LOG = Logger.getLogger("MinimalAI");

    // --- Configurable reward scales (matching Python sim defaults) ---
    private final float killReward;
    private final float deathPenalty;
    private final float damageDealtScale;
    private final float damageTakenScale;
    private final float whiffPenalty;
    private final float iFrameWastePenalty;
    private final float potGoodReward;
    private final float potBadPenalty;
    private final float gapGoodReward;
    private final float sprintResetReward;
    private final float approachScale;
    private final float healthAdvantageScale;

    // Optional observation builder to forward damage events for combat features
    private @Nullable ObservationBuilder obsBuilder;

    // Physics recorders to forward damage/KB events for recording sessions
    private final Map<String, PhysicsRecorder> recorders = new ConcurrentHashMap<>();

    // --- Per-bot tracking ---
    private final Map<String, Float> pendingRewards = new ConcurrentHashMap<>();
    private final Map<String, Float> previousHealth = new ConcurrentHashMap<>();
    private final Map<String, Float> previousTargetHealth = new ConcurrentHashMap<>();
    private final Map<String, Float> previousDistance = new ConcurrentHashMap<>();
    private final Set<String> registeredBots = ConcurrentHashMap.newKeySet();

    /**
     * Construct with default reward scales matching the Python training sim.
     */
    public RewardComputer() {
        this(5.0f, -3.0f, 0.5f, 0.1f, -0.05f, -0.03f,
             0.3f, -0.2f, 0.1f, 0.15f, 0.05f, 0.02f);
    }

    /**
     * Construct with explicit reward scales.
     *
     * @param killReward           reward for killing the target (+5.0)
     * @param deathPenalty         penalty for dying (-3.0)
     * @param damageDealtScale     multiplier for dealt / MAX_HP reward (0.5)
     * @param damageTakenScale     multiplier for taken / MAX_HP penalty (0.1)
     * @param whiffPenalty         penalty for attacking out of range (-0.05)
     * @param iFrameWastePenalty   penalty for hitting during i-frames (-0.03)
     * @param potGoodReward        reward for pot below 50% HP (+0.3)
     * @param potBadPenalty        penalty for pot above 80% HP (-0.2)
     * @param gapGoodReward        reward for gap below 70% HP (+0.1)
     * @param sprintResetReward    reward for sprint-reset within 4 blocks (+0.15)
     * @param approachScale        scale for approach potential shaping (0.05)
     * @param healthAdvantageScale scale for health-advantage shaping (0.02)
     */
    public RewardComputer(float killReward, float deathPenalty,
                          float damageDealtScale, float damageTakenScale,
                          float whiffPenalty, float iFrameWastePenalty,
                          float potGoodReward, float potBadPenalty,
                          float gapGoodReward, float sprintResetReward,
                          float approachScale, float healthAdvantageScale) {
        this.killReward = killReward;
        this.deathPenalty = deathPenalty;
        this.damageDealtScale = damageDealtScale;
        this.damageTakenScale = damageTakenScale;
        this.whiffPenalty = whiffPenalty;
        this.iFrameWastePenalty = iFrameWastePenalty;
        this.potGoodReward = potGoodReward;
        this.potBadPenalty = potBadPenalty;
        this.gapGoodReward = gapGoodReward;
        this.sprintResetReward = sprintResetReward;
        this.approachScale = approachScale;
        this.healthAdvantageScale = healthAdvantageScale;
    }

    // ----------------------------------------------------------------
    //  Bot registration
    // ----------------------------------------------------------------

    public void registerBot(String name) {
        registeredBots.add(name);
        pendingRewards.put(name, 0.0f);
        previousHealth.put(name, 20.0f);
        previousTargetHealth.put(name, 20.0f);
        previousDistance.put(name, 30.0f);
        LOG.info("[RewardComputer] Registered bot: " + name);
    }

    public void unregisterBot(String name) {
        registeredBots.remove(name);
        pendingRewards.remove(name);
        previousHealth.remove(name);
        previousTargetHealth.remove(name);
        previousDistance.remove(name);
    }

    public boolean isRegistered(String name) {
        return registeredBots.contains(name);
    }

    /**
     * Set the observation builder to forward damage events for combat features.
     */
    public void setObservationBuilder(@Nullable ObservationBuilder obsBuilder) {
        this.obsBuilder = obsBuilder;
    }

    public void registerRecorder(String name, PhysicsRecorder recorder) {
        recorders.put(name, recorder);
    }

    public void unregisterRecorder(String name) {
        recorders.remove(name);
    }

    // ----------------------------------------------------------------
    //  Bukkit event handlers
    // ----------------------------------------------------------------

    @EventHandler
    public void onEntityDamage(EntityDamageByEntityEvent e) {
        // Damage dealt by a registered bot
        if (e.getDamager() instanceof Player attacker && registeredBots.contains(attacker.getName())) {
            float dealt = (float) e.getFinalDamage();
            float reward = (dealt / 20.0f) * damageDealtScale;
            addReward(attacker.getName(), reward);
            if (obsBuilder != null) {
                obsBuilder.onDamageDealt(attacker.getName(), dealt);
            }
            PhysicsRecorder rec = recorders.get(attacker.getName());
            if (rec != null) {
                rec.onDamageDealt(dealt);
            }
        }

        // Damage received by a registered bot
        if (e.getEntity() instanceof Player victim && registeredBots.contains(victim.getName())) {
            float taken = (float) e.getFinalDamage();
            float penalty = -(taken / 20.0f) * damageTakenScale;
            addReward(victim.getName(), penalty);
            if (obsBuilder != null) {
                obsBuilder.onDamageTaken(victim.getName(), taken);
            }
            PhysicsRecorder rec = recorders.get(victim.getName());
            if (rec != null) {
                // Extract knockback from the velocity change
                Player victimPlayer = victim;
                rec.onDamageTaken(taken,
                        victimPlayer.getVelocity().getX(),
                        victimPlayer.getVelocity().getY(),
                        victimPlayer.getVelocity().getZ());
            }
        }
    }

    @EventHandler
    public void onPlayerDeath(PlayerDeathEvent e) {
        Player dead = e.getEntity();

        // Bot died
        if (registeredBots.contains(dead.getName())) {
            addReward(dead.getName(), deathPenalty);
        }

        // Bot got the kill
        Player killer = dead.getKiller();
        if (killer != null && registeredBots.contains(killer.getName())) {
            addReward(killer.getName(), killReward);
        }
    }

    // ----------------------------------------------------------------
    //  Manual reward additions (called by BotBrain / tick logic)
    // ----------------------------------------------------------------

    /**
     * Add arbitrary reward to a bot's pending accumulator.
     */
    public void addReward(String botName, float amount) {
        if (!registeredBots.contains(botName)) return;
        pendingRewards.merge(botName, amount, Float::sum);
    }

    /**
     * Consume (read + reset) the accumulated reward for a bot this tick.
     *
     * @return accumulated reward since last consume, clamped to [-5, 5]
     */
    public float consumeReward(String botName) {
        Float reward = pendingRewards.put(botName, 0.0f);
        if (reward == null) return 0.0f;
        return Math.max(-5.0f, Math.min(5.0f, reward));
    }

    // ----------------------------------------------------------------
    //  Potential-based shaping (called each tick by BotBrain)
    // ----------------------------------------------------------------

    /**
     * Update potential-based reward shaping for approach + health advantage.
     *
     * @param botName       the bot name
     * @param myHealth      bot's current health (0-20)
     * @param targetHealth  target's current health (0-20)
     * @param distToTarget  distance to current target
     */
    public void updatePotentialShaping(String botName, float myHealth, float targetHealth, float distToTarget) {
        if (!registeredBots.contains(botName)) return;

        float prevDist = previousDistance.getOrDefault(botName, 30.0f);
        float prevMyHp = previousHealth.getOrDefault(botName, 20.0f);
        float prevTargetHp = previousTargetHealth.getOrDefault(botName, 20.0f);

        // Approach shaping: reward getting closer
        float approachReward = approachScale * (prevDist - distToTarget) / 30.0f;

        // Health advantage shaping: reward gaining HP edge
        float currentAdvantage = myHealth - targetHealth;
        float previousAdvantage = prevMyHp - prevTargetHp;
        float advantageReward = healthAdvantageScale * (currentAdvantage - previousAdvantage);

        addReward(botName, approachReward + advantageReward);

        previousHealth.put(botName, myHealth);
        previousTargetHealth.put(botName, targetHealth);
        previousDistance.put(botName, distToTarget);
    }

    // ----------------------------------------------------------------
    //  Action quality callbacks (called by BotBrain per action)
    // ----------------------------------------------------------------

    /**
     * Called when a bot attacks. Applies whiff / i-frame-waste penalties.
     *
     * @param botName     the bot name
     * @param hit         whether the attack connected
     * @param iFrameWaste whether the target had active invulnerability ticks
     */
    public void onBotAttack(String botName, boolean hit, boolean iFrameWaste) {
        if (!registeredBots.contains(botName)) return;

        if (!hit) {
            addReward(botName, whiffPenalty);
        } else if (iFrameWaste) {
            addReward(botName, iFrameWastePenalty);
        }
    }

    /**
     * Called when a bot uses a splash health potion.
     *
     * @param botName            the bot name
     * @param currentHealthRatio health / maxHealth (0.0 to 1.0)
     */
    public void onBotUsePot(String botName, float currentHealthRatio) {
        if (!registeredBots.contains(botName)) return;

        if (currentHealthRatio < 0.5f) {
            addReward(botName, potGoodReward);
        } else if (currentHealthRatio > 0.8f) {
            addReward(botName, potBadPenalty);
        }
    }

    /**
     * Called when a bot eats a golden apple.
     *
     * @param botName            the bot name
     * @param currentHealthRatio health / maxHealth (0.0 to 1.0)
     */
    public void onBotUseGap(String botName, float currentHealthRatio) {
        if (!registeredBots.contains(botName)) return;

        if (currentHealthRatio < 0.7f) {
            addReward(botName, gapGoodReward);
        }
    }

    /**
     * Called when a bot performs a sprint-reset.
     *
     * @param botName      the bot name
     * @param distToTarget distance to current target in blocks
     */
    public void onBotSprintReset(String botName, float distToTarget) {
        if (!registeredBots.contains(botName)) return;

        if (distToTarget <= 4.0f) {
            addReward(botName, sprintResetReward);
        }
    }
}
