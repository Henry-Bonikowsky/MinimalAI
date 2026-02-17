package com.minimalai.integration;

import org.bukkit.entity.LivingEntity;
import org.bukkit.entity.Player;

import java.util.List;
import java.util.UUID;

/**
 * Interface defining what MinimalAI needs from ArcaneSigils.
 * Mirrors com.miracle.arcanesigils.api.ArcaneSigilsAPI.
 */
public interface ArcaneSigilsAPI {

    record SigilInfo(String id, String name, int tier, String slot, String activationType) {}

    record MarkInfo(String name, double multiplier, long expiryTimeMs, UUID ownerUUID) {}

    // --- Equipped Sigils ---
    List<SigilInfo> getEquippedSigils(Player player);

    // --- Bind Slot Queries ---
    boolean isSigilReady(Player player, int bindSlot);
    double getCooldownProgress(Player player, int bindSlot);
    double getCooldownRemaining(Player player, int bindSlot);
    double getMaxCooldown(Player player, int bindSlot);
    int getTier(Player player, int bindSlot);
    String getSigilType(Player player, int bindSlot);
    String getActivationType(Player player, int bindSlot);

    // --- Ability Activation ---
    boolean activateAbility(Player player, int bindSlot);

    // --- Combat State ---
    double getDamageAmplifier(Player player);
    double getDamageReduction(Player player);

    // --- Specific Sigil State ---
    int getKingsBraceCharges(Player player);
    int getInvulnHits(Player player);

    // --- Marks ---
    boolean isMarked(Player target, Player attacker);
    boolean hasMark(LivingEntity entity, String markName);
    List<MarkInfo> getActiveMarks(LivingEntity entity);
    double getMarkDamageMultiplier(LivingEntity entity);

    // --- Targets ---
    LivingEntity getSelectedTarget(Player player);
    LivingEntity getLastVictim(Player player);

    // --- Availability ---
    boolean isAvailable();
}
