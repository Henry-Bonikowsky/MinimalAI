package com.minimalai.training;

import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.phys.Vec3;
import org.jetbrains.annotations.Nullable;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

/**
 * Records per-tick physics state to CSV for system identification.
 *
 * Each tick captures the state BEFORE and AFTER action execution,
 * allowing a Python script to derive exact physics constants
 * (speed, friction, gravity, knockback, etc.) from real server data.
 */
public class PhysicsRecorder implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger("MinimalAI");
    private static final String HEADER =
            "tick,bot," +
            "pre_x,pre_y,pre_z,pre_vx,pre_vy,pre_vz," +
            "pre_yaw,pre_pitch,pre_onGround,pre_sprinting,pre_sneaking," +
            "act_fwd,act_back,act_left,act_right,act_jump,act_sneak,act_sprint," +
            "act_attack,act_block," +
            "post_x,post_y,post_z,post_vx,post_vy,post_vz," +
            "post_yaw,post_pitch,post_onGround,post_sprinting,post_sneaking," +
            "dmg_dealt,dmg_taken,kb_x,kb_y,kb_z," +
            "target_dist,target_dx,target_dy,target_dz";

    private final BufferedWriter writer;
    private final Path filePath;
    private int tickCounter = 0;
    private boolean closed = false;

    // Per-tick damage accumulators (set by external callbacks)
    private float dmgDealt = 0f;
    private float dmgTaken = 0f;
    private double kbX = 0, kbY = 0, kbZ = 0;

    public PhysicsRecorder(Path outputFile) throws IOException {
        this.filePath = outputFile;
        Files.createDirectories(outputFile.getParent());
        this.writer = Files.newBufferedWriter(outputFile);
        writer.write(HEADER);
        writer.newLine();
        LOG.info("[PhysicsRecorder] Recording to " + outputFile);
    }

    /**
     * Snapshot of a player's physics state at one instant.
     */
    public record StateSnapshot(
            double x, double y, double z,
            double vx, double vy, double vz,
            float yaw, float pitch,
            boolean onGround, boolean sprinting, boolean sneaking
    ) {
        public static StateSnapshot capture(ServerPlayer bot) {
            Vec3 vel = bot.getDeltaMovement();
            return new StateSnapshot(
                    bot.getX(), bot.getY(), bot.getZ(),
                    vel.x, vel.y, vel.z,
                    bot.getYRot(), bot.getXRot(),
                    bot.onGround(), bot.isSprinting(), bot.isShiftKeyDown()
            );
        }
    }

    /**
     * Record one tick of physics data.
     *
     * @param botName  bot identifier
     * @param pre      state snapshot BEFORE action execution
     * @param post     state snapshot AFTER action execution
     * @param actions  the 35-element action vector (we extract movement-relevant ones)
     * @param target   the combat target (nullable, for distance/direction data)
     * @param botPlayer the bot's ServerPlayer (for target-relative calculations)
     */
    public void record(String botName, StateSnapshot pre, StateSnapshot post,
                       int[] actions, @Nullable LivingEntity target,
                       ServerPlayer botPlayer) {
        if (closed) return;

        try {
            StringBuilder sb = new StringBuilder(512);
            sb.append(tickCounter++).append(',').append(botName).append(',');

            // Pre-action state
            appendState(sb, pre);

            // Actions (movement-relevant subset)
            sb.append(actions[0]).append(',');  // fwd
            sb.append(actions[1]).append(',');  // back
            sb.append(actions[2]).append(',');  // strafe left
            sb.append(actions[3]).append(',');  // strafe right
            sb.append(actions[4]).append(',');  // jump
            sb.append(actions[5]).append(',');  // sneak
            sb.append(actions[6]).append(',');  // sprint
            sb.append(actions[7]).append(',');  // attack
            sb.append(actions[8]).append(',');  // block

            // Post-action state
            appendState(sb, post);

            // Combat data
            sb.append(fmt(dmgDealt)).append(',');
            sb.append(fmt(dmgTaken)).append(',');
            sb.append(fmt(kbX)).append(',');
            sb.append(fmt(kbY)).append(',');
            sb.append(fmt(kbZ)).append(',');

            // Target data
            if (target != null && target.isAlive()) {
                double dx = target.getX() - botPlayer.getX();
                double dy = target.getY() - botPlayer.getY();
                double dz = target.getZ() - botPlayer.getZ();
                double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                sb.append(fmt(dist)).append(',');
                sb.append(fmt(dx)).append(',');
                sb.append(fmt(dy)).append(',');
                sb.append(fmt(dz));
            } else {
                sb.append(",,,,");
            }

            writer.write(sb.toString());
            writer.newLine();

            // Reset per-tick accumulators
            dmgDealt = 0f;
            dmgTaken = 0f;
            kbX = kbY = kbZ = 0;

        } catch (IOException e) {
            LOG.warning("[PhysicsRecorder] Write error: " + e.getMessage());
        }
    }

    /** Called from damage event listeners. */
    public void onDamageDealt(float amount) {
        dmgDealt += amount;
    }

    /** Called from damage event listeners. */
    public void onDamageTaken(float amount, double knockbackX, double knockbackY, double knockbackZ) {
        dmgTaken += amount;
        kbX += knockbackX;
        kbY += knockbackY;
        kbZ += knockbackZ;
    }

    public int getTickCount() {
        return tickCounter;
    }

    public Path getFilePath() {
        return filePath;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            writer.flush();
            writer.close();
            LOG.info("[PhysicsRecorder] Saved " + tickCounter + " ticks to " + filePath);
        } catch (IOException e) {
            LOG.warning("[PhysicsRecorder] Error closing: " + e.getMessage());
        }
    }

    private void appendState(StringBuilder sb, StateSnapshot s) {
        sb.append(fmt(s.x)).append(',');
        sb.append(fmt(s.y)).append(',');
        sb.append(fmt(s.z)).append(',');
        sb.append(fmt(s.vx)).append(',');
        sb.append(fmt(s.vy)).append(',');
        sb.append(fmt(s.vz)).append(',');
        sb.append(fmt(s.yaw)).append(',');
        sb.append(fmt(s.pitch)).append(',');
        sb.append(s.onGround ? 1 : 0).append(',');
        sb.append(s.sprinting ? 1 : 0).append(',');
        sb.append(s.sneaking ? 1 : 0).append(',');
    }

    private static String fmt(double v) {
        return String.format("%.6f", v);
    }
}
