package com.minimalai.bot;

import io.netty.channel.Channel;
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
 * The server requires every ServerPlayer to have a valid Connection and
 * packet listener so it doesn't NPE when broadcasting packets.  We give
 * it an EmbeddedChannel (Netty in-memory) and a listener that silently
 * drops every outbound packet.
 */
public class FakeConnection {

    private final Connection connection;
    private final ServerGamePacketListenerImpl packetListener;

    private FakeConnection(Connection connection, ServerGamePacketListenerImpl packetListener) {
        this.connection = connection;
        this.packetListener = packetListener;
    }

    /**
     * Build a fully wired FakeConnection for the given bot player.
     *
     * @param server the dedicated server instance
     * @param player the fake ServerPlayer that needs networking
     * @param cookie the CommonListenerCookie used during login
     * @return a FakeConnection whose {@link #connection()} can be passed to placeNewPlayer
     */
    public static FakeConnection create(MinecraftServer server, ServerPlayer player, CommonListenerCookie cookie) {
        // 1. Create a Connection backed by an in-memory Netty channel.
        //    PacketFlow.SERVERBOUND means "we receive from client" which is
        //    the perspective the server expects for an incoming connection.
        Connection connection = new Connection(PacketFlow.SERVERBOUND);

        // 2. Attach an EmbeddedChannel so Netty internals don't NPE.
        //    EmbeddedChannel is a fully functional in-memory channel.
        Channel channel = new EmbeddedChannel();
        connection.channel = channel;
        // Give it a remote address so logging / anti-cheat doesn't explode
        connection.address = new InetSocketAddress("127.0.0.1", 0);

        // TODO: In some Paper builds connection.channel is private/final.
        //       If direct field assignment fails, use reflection:
        //
        //   Field channelField = Connection.class.getDeclaredField("channel");
        //   channelField.setAccessible(true);
        //   channelField.set(connection, channel);
        //
        //   Field addressField = Connection.class.getDeclaredField("address");
        //   addressField.setAccessible(true);
        //   addressField.set(connection, new InetSocketAddress("127.0.0.1", 0));

        // 3. Build the packet listener.
        //    ServerGamePacketListenerImpl is the main in-game packet handler.
        //    We override send() to no-op so outbound packets are silently dropped.
        //
        // TODO: The ServerGamePacketListenerImpl constructor signature may vary
        //       across Paper versions. Typical 1.21.x signature:
        //       (MinecraftServer, Connection, ServerPlayer, CommonListenerCookie)
        ServerGamePacketListenerImpl listener = new NoOpPacketListener(server, connection, player, cookie);

        // Wire the listener onto the connection
        player.connection = listener;

        return new FakeConnection(connection, listener);
    }

    public Connection connection() {
        return connection;
    }

    public ServerGamePacketListenerImpl packetListener() {
        return packetListener;
    }

    /**
     * Packet listener that drops all outbound traffic.
     * Keeps the bot "connected" without generating network I/O.
     */
    private static class NoOpPacketListener extends ServerGamePacketListenerImpl {

        // TODO: Verify this constructor matches your Paper 1.21.10 build.
        //       If Paper adds/removes parameters, adjust here.
        public NoOpPacketListener(MinecraftServer server, Connection connection,
                                  ServerPlayer player, CommonListenerCookie cookie) {
            super(server, connection, player, cookie);
        }

        @Override
        public void send(Packet<?> packet) {
            // Silently drop -- no real client to receive this
        }

        // send(Packet, PacketSendListener) may not exist in all Paper builds.
        // If it does, uncomment this @Override.
        public void send(Packet<?> packet, PacketSendListener listener) {
            // Silently drop
        }

        // TODO: If the server kicks bots for "timed out", override the
        //       tick/keepalive methods to no-op as well:
        //
        //   @Override
        //   public void handleKeepAlive(ServerboundKeepAlivePacket pkt) { }
        //
        //   @Override
        //   public void tick() {
        //       // Call super.tick() but catch/suppress timeout disconnect
        //   }
    }
}
