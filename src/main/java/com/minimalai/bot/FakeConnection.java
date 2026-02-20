package com.minimalai.bot;

import io.netty.channel.Channel;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.channel.ChannelOutboundHandlerAdapter;
import io.netty.channel.ChannelPromise;
import io.netty.channel.embedded.EmbeddedChannel;
import net.minecraft.network.Connection;
import net.minecraft.network.PacketSendListener;
import net.minecraft.network.protocol.Packet;
import net.minecraft.network.protocol.PacketFlow;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.network.CommonListenerCookie;
import net.minecraft.server.network.ServerGamePacketListenerImpl;

import java.net.InetSocketAddress;

/**
 * No-op network plumbing for fake (bot) players.
 *
 * Adds dummy pipeline handlers ("encoder", "decoder", "splitter", "prepender")
 * so plugins like ProtocolLib, TAB, and ModelEngine don't crash when they
 * try to inject into or write through the pipeline.
 */
public class FakeConnection {

    private final Connection connection;
    private final ServerGamePacketListenerImpl packetListener;

    private FakeConnection(Connection connection, ServerGamePacketListenerImpl packetListener) {
        this.connection = connection;
        this.packetListener = packetListener;
    }

    public static FakeConnection create(MinecraftServer server, ServerPlayer player, CommonListenerCookie cookie) {
        Connection connection = new Connection(PacketFlow.SERVERBOUND);

        EmbeddedChannel channel = new EmbeddedChannel();

        // Add dummy named handlers that plugins expect to find in the pipeline.
        // ProtocolLib/PacketEvents inject "before encoder" / "after decoder" — without
        // these named handlers they throw NoSuchElementException.
        channel.pipeline().addLast("splitter", new ChannelInboundHandlerAdapter());
        channel.pipeline().addLast("decoder", new ChannelInboundHandlerAdapter());
        channel.pipeline().addLast("prepender", new ChannelOutboundHandlerAdapter() {
            @Override
            public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise) {
                promise.setSuccess();
            }
        });
        channel.pipeline().addLast("encoder", new ChannelOutboundHandlerAdapter() {
            @Override
            public void write(ChannelHandlerContext ctx, Object msg, ChannelPromise promise) {
                promise.setSuccess();
            }
        });
        channel.pipeline().addLast("packet_handler", connection);

        connection.channel = channel;
        connection.address = new InetSocketAddress("127.0.0.1", 0);

        ServerGamePacketListenerImpl listener = new NoOpPacketListener(server, connection, player, cookie);
        player.connection = listener;

        return new FakeConnection(connection, listener);
    }

    /**
     * Reassign the NoOpPacketListener onto the player AND the Connection
     * AFTER placeNewPlayer(). placeNewPlayer creates its own
     * ServerGamePacketListenerImpl and registers it on both player.connection
     * and connection.packetListener. We must replace BOTH references so the
     * real SGPLI's tick() (keepalive checks, movement validation, anti-cheat
     * position resets) never runs.
     */
    public void reattach(ServerPlayer player) {
        player.connection = packetListener;
        // Replace the listener on the Connection object itself —
        // same call placeNewPlayer uses internally.
        connection.setupInboundProtocol(
                net.minecraft.network.protocol.game.GameProtocols.SERVERBOUND_TEMPLATE
                        .bind(net.minecraft.network.RegistryFriendlyByteBuf.decorator(
                                net.minecraft.server.MinecraftServer.getServer().registryAccess()),
                                packetListener),
                packetListener
        );
    }

    public Connection connection() {
        return connection;
    }

    public ServerGamePacketListenerImpl packetListener() {
        return packetListener;
    }

    private static class NoOpPacketListener extends ServerGamePacketListenerImpl {

        public NoOpPacketListener(MinecraftServer server, Connection connection,
                                  ServerPlayer player, CommonListenerCookie cookie) {
            super(server, connection, player, cookie);
        }

        @Override
        public void tick() {
            // Skip keepalive/network checks (would disconnect the fake player),
            // but still tick the player for movement, combat cooldowns, potion
            // effects, etc. Vanilla's aiStep() → travel() processes xxa/zza input,
            // gravity, knockback, and friction.
            this.player.doTick();
        }

        @Override
        public void send(Packet<?> packet) {
            // Silently drop
        }

        // Overload for packets with send listener (silently drop)
        public void send(Packet<?> packet, PacketSendListener listener) {
            // Silently drop — the channel encoder no-ops already handle this,
            // but this catches any direct calls with a PacketSendListener arg.
        }
    }
}
