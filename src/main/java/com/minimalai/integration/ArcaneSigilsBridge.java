package com.minimalai.integration;

import org.bukkit.Bukkit;
import org.bukkit.entity.LivingEntity;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;
import org.bukkit.scoreboard.Objective;
import org.bukkit.scoreboard.Score;
import org.bukkit.scoreboard.Scoreboard;

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Bridge to ArcaneSigils plugin.
 *
 * Strategy: Try direct API via reflection (ArcaneSigils.getAPI()), fall back
 * to scoreboard queries, fall back to command dispatch.
 *
 * Once ArcaneSigils exposes its API jar as a compileOnly dependency,
 * the reflection can be replaced with direct calls.
 */
public class ArcaneSigilsBridge implements ArcaneSigilsAPI {

    private static final int MAX_SLOTS = 12;

    private final Logger logger;
    private final boolean pluginPresent;
    private Object nativeApi; // com.miracle.arcanesigils.api.ArcaneSigilsAPI via reflection

    public ArcaneSigilsBridge(Logger logger) {
        this.logger = logger;
        Plugin sigils = Bukkit.getPluginManager().getPlugin("ArcaneSigils");
        this.pluginPresent = sigils != null && sigils.isEnabled();

        if (pluginPresent) {
            // Try to get native API via reflection
            try {
                Method getApi = sigils.getClass().getMethod("getAPI");
                nativeApi = getApi.invoke(null);
                logger.info("ArcaneSigils API connected directly");
            } catch (Exception e) {
                nativeApi = null;
                logger.info("ArcaneSigils detected - using scoreboard fallback");
            }
        } else {
            logger.info("ArcaneSigils not found - sigil features disabled");
        }
    }

    @Override
    public boolean isAvailable() {
        return pluginPresent;
    }

    @Override
    public List<SigilInfo> getEquippedSigils(Player player) {
        if (!pluginPresent) return Collections.emptyList();
        // TODO: Implement via native API or scoreboard
        return Collections.emptyList();
    }

    @Override
    public boolean isSigilReady(Player player, int bindSlot) {
        if (!pluginPresent) return false;
        return getScoreboardValue(player, "as_sigil_" + bindSlot + "_ready") == 1;
    }

    @Override
    public double getCooldownProgress(Player player, int bindSlot) {
        if (!pluginPresent) return 0.0;
        return getScoreboardValue(player, "as_sigil_" + bindSlot + "_cd") / 100.0;
    }

    @Override
    public double getCooldownRemaining(Player player, int bindSlot) {
        if (!pluginPresent) return 0.0;
        return getScoreboardValue(player, "as_sigil_" + bindSlot + "_cd_rem") / 20.0;
    }

    @Override
    public double getMaxCooldown(Player player, int bindSlot) {
        if (!pluginPresent) return 0.0;
        return getScoreboardValue(player, "as_sigil_" + bindSlot + "_cd_max") / 20.0;
    }

    @Override
    public int getTier(Player player, int bindSlot) {
        if (!pluginPresent) return 0;
        return getScoreboardValue(player, "as_sigil_" + bindSlot + "_tier");
    }

    @Override
    public String getSigilType(Player player, int bindSlot) {
        if (!pluginPresent) return "empty";
        // TODO: Map from scoreboard int code to string ID once ArcaneSigils exposes this
        return "unknown";
    }

    @Override
    public String getActivationType(Player player, int bindSlot) {
        if (!pluginPresent) return "passive";
        return "ability"; // TODO: Read from API/scoreboard
    }

    @Override
    public boolean activateAbility(Player player, int bindSlot) {
        if (!pluginPresent) return false;
        if (!isSigilReady(player, bindSlot)) return false;
        try {
            return Bukkit.dispatchCommand(Bukkit.getConsoleSender(),
                    "arcanesigils activate " + player.getName() + " " + bindSlot);
        } catch (Exception e) {
            logger.warning("Failed to dispatch sigil activation: " + e.getMessage());
            return false;
        }
    }

    @Override
    public double getDamageAmplifier(Player player) {
        if (!pluginPresent) return 1.0;
        int raw = getScoreboardValue(player, "as_dmg_amp");
        return raw == 0 ? 1.0 : raw / 100.0;
    }

    @Override
    public double getDamageReduction(Player player) {
        if (!pluginPresent) return 1.0;
        int raw = getScoreboardValue(player, "as_dmg_red");
        return raw == 0 ? 1.0 : raw / 100.0;
    }

    @Override
    public int getKingsBraceCharges(Player player) {
        if (!pluginPresent) return 0;
        return getScoreboardValue(player, "as_kb_charges");
    }

    @Override
    public int getInvulnHits(Player player) {
        if (!pluginPresent) return 0;
        return getScoreboardValue(player, "as_invuln");
    }

    @Override
    public boolean isMarked(Player target, Player attacker) {
        if (!pluginPresent) return false;
        return getScoreboardValue(target, "as_marked_by_" + attacker.getName()) > 0;
    }

    @Override
    public boolean hasMark(LivingEntity entity, String markName) {
        if (!pluginPresent) return false;
        if (!(entity instanceof Player p)) return false;
        return getScoreboardValue(p, "as_mark_" + markName.toLowerCase()) > 0;
    }

    @Override
    public List<MarkInfo> getActiveMarks(LivingEntity entity) {
        if (!pluginPresent) return Collections.emptyList();
        // TODO: Implement via native API
        return Collections.emptyList();
    }

    @Override
    public double getMarkDamageMultiplier(LivingEntity entity) {
        if (!pluginPresent) return 1.0;
        if (!(entity instanceof Player p)) return 1.0;
        int raw = getScoreboardValue(p, "as_mark_dmg_mult");
        return raw == 0 ? 1.0 : raw / 100.0;
    }

    @Override
    public LivingEntity getSelectedTarget(Player player) {
        // TODO: Implement via native API
        return null;
    }

    @Override
    public LivingEntity getLastVictim(Player player) {
        // TODO: Implement via native API
        return null;
    }

    private int getScoreboardValue(Player player, String objectiveName) {
        try {
            Scoreboard scoreboard = Bukkit.getScoreboardManager().getMainScoreboard();
            Objective objective = scoreboard.getObjective(objectiveName);
            if (objective == null) return 0;
            Score score = objective.getScore(player);
            return score.isScoreSet() ? score.getScore() : 0;
        } catch (Exception e) {
            return 0;
        }
    }
}
