package com.minimalai.integration;

import org.bukkit.Bukkit;
import org.bukkit.entity.LivingEntity;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Bridge to ArcaneSigils plugin via reflection against its real Java API.
 *
 * Uses ArmorSetsPlugin.getAPI() to get the native ArcaneSigilsAPI instance,
 * then forwards all calls via cached reflection Method handles.
 */
public class ArcaneSigilsBridge implements ArcaneSigilsAPI {

    private final Logger logger;
    private final boolean pluginPresent;
    private Object nativeApi; // com.miracle.arcanesigils.api.ArcaneSigilsAPI

    // Cached method handles for hot-path calls
    private Method mIsSigilReady;
    private Method mGetCooldownProgress;
    private Method mGetCooldownRemaining;
    private Method mGetMaxCooldown;
    private Method mGetTier;
    private Method mGetSigilType;
    private Method mGetActivationType;
    private Method mActivateAbility;
    private Method mGetDamageAmplifier;
    private Method mGetDamageReduction;
    private Method mGetKingsBraceCharges;
    private Method mGetInvulnHits;
    private Method mIsMarked;
    private Method mHasMark;
    private Method mGetActiveMarks;
    private Method mGetMarkDamageMultiplier;
    private Method mGetSelectedTarget;
    private Method mGetLastVictim;
    private Method mGetEquippedSigils;
    private Method mRegisterBotSigils;
    private Method mUnregisterBotSigils;

    public ArcaneSigilsBridge(Logger logger) {
        this.logger = logger;
        Plugin sigils = Bukkit.getPluginManager().getPlugin("ArcaneSigils");
        this.pluginPresent = sigils != null && sigils.isEnabled();

        if (pluginPresent) {
            try {
                // ArmorSetsPlugin.getAPI() is static
                Method getApi = sigils.getClass().getMethod("getAPI");
                nativeApi = getApi.invoke(null);
                if (nativeApi != null) {
                    cacheMethodHandles(nativeApi.getClass());
                    logger.info("ArcaneSigils API connected via reflection");
                } else {
                    logger.warning("ArcaneSigils getAPI() returned null");
                }
            } catch (Exception e) {
                nativeApi = null;
                logger.warning("Failed to connect ArcaneSigils API: " + e.getMessage());
            }
        } else {
            logger.info("ArcaneSigils not found - sigil features disabled");
        }
    }

    private void cacheMethodHandles(Class<?> apiClass) {
        try {
            mIsSigilReady = apiClass.getMethod("isSigilReady", Player.class, int.class);
            mGetCooldownProgress = apiClass.getMethod("getCooldownProgress", Player.class, int.class);
            mGetCooldownRemaining = apiClass.getMethod("getCooldownRemaining", Player.class, int.class);
            mGetMaxCooldown = apiClass.getMethod("getMaxCooldown", Player.class, int.class);
            mGetTier = apiClass.getMethod("getTier", Player.class, int.class);
            mGetSigilType = apiClass.getMethod("getSigilType", Player.class, int.class);
            mGetActivationType = apiClass.getMethod("getActivationType", Player.class, int.class);
            mActivateAbility = apiClass.getMethod("activateAbility", Player.class, int.class);
            mGetDamageAmplifier = apiClass.getMethod("getDamageAmplifier", Player.class);
            mGetDamageReduction = apiClass.getMethod("getDamageReduction", Player.class);
            mGetKingsBraceCharges = apiClass.getMethod("getKingsBraceCharges", Player.class);
            mGetInvulnHits = apiClass.getMethod("getInvulnHits", Player.class);
            mIsMarked = apiClass.getMethod("isMarked", Player.class, Player.class);
            mHasMark = apiClass.getMethod("hasMark", LivingEntity.class, String.class);
            mGetActiveMarks = apiClass.getMethod("getActiveMarks", LivingEntity.class);
            mGetMarkDamageMultiplier = apiClass.getMethod("getMarkDamageMultiplier", LivingEntity.class);
            mGetSelectedTarget = apiClass.getMethod("getSelectedTarget", Player.class);
            mGetLastVictim = apiClass.getMethod("getLastVictim", Player.class);
            mGetEquippedSigils = apiClass.getMethod("getEquippedSigils", Player.class);
            mRegisterBotSigils = apiClass.getMethod("registerBotSigils", Player.class, java.util.List.class);
            mUnregisterBotSigils = apiClass.getMethod("unregisterBotSigils", Player.class);
        } catch (NoSuchMethodException e) {
            logger.warning("ArcaneSigils API method not found: " + e.getMessage());
            nativeApi = null;
        }
    }

    @Override
    public boolean isAvailable() {
        return pluginPresent && nativeApi != null;
    }

    @Override
    public List<SigilInfo> getEquippedSigils(Player player) {
        if (!isAvailable()) return Collections.emptyList();
        // TODO: Map native SigilInfo records to our SigilInfo records
        return Collections.emptyList();
    }

    @Override
    public boolean isSigilReady(Player player, int bindSlot) {
        if (!isAvailable()) return false;
        try {
            return (boolean) mIsSigilReady.invoke(nativeApi, player, bindSlot);
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public double getCooldownProgress(Player player, int bindSlot) {
        if (!isAvailable()) return 0.0;
        try {
            return (double) mGetCooldownProgress.invoke(nativeApi, player, bindSlot);
        } catch (Exception e) {
            return 0.0;
        }
    }

    @Override
    public double getCooldownRemaining(Player player, int bindSlot) {
        if (!isAvailable()) return 0.0;
        try {
            return (double) mGetCooldownRemaining.invoke(nativeApi, player, bindSlot);
        } catch (Exception e) {
            return 0.0;
        }
    }

    @Override
    public double getMaxCooldown(Player player, int bindSlot) {
        if (!isAvailable()) return 0.0;
        try {
            return (double) mGetMaxCooldown.invoke(nativeApi, player, bindSlot);
        } catch (Exception e) {
            return 0.0;
        }
    }

    @Override
    public int getTier(Player player, int bindSlot) {
        if (!isAvailable()) return 0;
        try {
            return (int) mGetTier.invoke(nativeApi, player, bindSlot);
        } catch (Exception e) {
            return 0;
        }
    }

    @Override
    public String getSigilType(Player player, int bindSlot) {
        if (!isAvailable()) return "empty";
        try {
            return (String) mGetSigilType.invoke(nativeApi, player, bindSlot);
        } catch (Exception e) {
            return "empty";
        }
    }

    @Override
    public String getActivationType(Player player, int bindSlot) {
        if (!isAvailable()) return "passive";
        try {
            return (String) mGetActivationType.invoke(nativeApi, player, bindSlot);
        } catch (Exception e) {
            return "passive";
        }
    }

    @Override
    public boolean activateAbility(Player player, int bindSlot) {
        if (!isAvailable()) return false;
        try {
            return (boolean) mActivateAbility.invoke(nativeApi, player, bindSlot);
        } catch (Exception e) {
            logger.warning("Failed to activate sigil ability: " + e.getMessage());
            return false;
        }
    }

    @Override
    public double getDamageAmplifier(Player player) {
        if (!isAvailable()) return 1.0;
        try {
            return (double) mGetDamageAmplifier.invoke(nativeApi, player);
        } catch (Exception e) {
            return 1.0;
        }
    }

    @Override
    public double getDamageReduction(Player player) {
        if (!isAvailable()) return 1.0;
        try {
            return (double) mGetDamageReduction.invoke(nativeApi, player);
        } catch (Exception e) {
            return 1.0;
        }
    }

    @Override
    public int getKingsBraceCharges(Player player) {
        if (!isAvailable()) return 0;
        try {
            return (int) mGetKingsBraceCharges.invoke(nativeApi, player);
        } catch (Exception e) {
            return 0;
        }
    }

    @Override
    public int getInvulnHits(Player player) {
        if (!isAvailable()) return 0;
        try {
            return (int) mGetInvulnHits.invoke(nativeApi, player);
        } catch (Exception e) {
            return 0;
        }
    }

    @Override
    public boolean isMarked(Player target, Player attacker) {
        if (!isAvailable()) return false;
        try {
            return (boolean) mIsMarked.invoke(nativeApi, target, attacker);
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public boolean hasMark(LivingEntity entity, String markName) {
        if (!isAvailable()) return false;
        try {
            return (boolean) mHasMark.invoke(nativeApi, entity, markName);
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<MarkInfo> getActiveMarks(LivingEntity entity) {
        if (!isAvailable()) return Collections.emptyList();
        // TODO: Map native MarkInfo to our MarkInfo
        return Collections.emptyList();
    }

    @Override
    public double getMarkDamageMultiplier(LivingEntity entity) {
        if (!isAvailable()) return 1.0;
        try {
            return (double) mGetMarkDamageMultiplier.invoke(nativeApi, entity);
        } catch (Exception e) {
            return 1.0;
        }
    }

    @Override
    public LivingEntity getSelectedTarget(Player player) {
        if (!isAvailable()) return null;
        try {
            return (LivingEntity) mGetSelectedTarget.invoke(nativeApi, player);
        } catch (Exception e) {
            return null;
        }
    }

    @Override
    public void registerBotSigils(Player player, java.util.List<String> sigilIds) {
        if (!isAvailable()) return;
        try {
            mRegisterBotSigils.invoke(nativeApi, player, sigilIds);
            logger.info("Registered " + sigilIds.size() + " virtual sigils for bot " + player.getName());
        } catch (Exception e) {
            logger.warning("Failed to register bot sigils: " + e.getMessage());
        }
    }

    @Override
    public void unregisterBotSigils(Player player) {
        if (!isAvailable()) return;
        try {
            mUnregisterBotSigils.invoke(nativeApi, player);
        } catch (Exception e) {
            logger.warning("Failed to unregister bot sigils: " + e.getMessage());
        }
    }

    @Override
    public LivingEntity getLastVictim(Player player) {
        if (!isAvailable()) return null;
        try {
            return (LivingEntity) mGetLastVictim.invoke(nativeApi, player);
        } catch (Exception e) {
            return null;
        }
    }
}
