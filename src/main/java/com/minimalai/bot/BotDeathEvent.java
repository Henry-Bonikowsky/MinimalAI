package com.minimalai.bot;

import org.bukkit.Location;
import org.bukkit.event.Event;
import org.bukkit.event.HandlerList;

/**
 * Fired when a MinimalAI bot dies and is about to be despawned.
 * Other plugins can listen to this and call FakePlayerManager.spawn()
 * to implement their own respawn loop.
 */
public class BotDeathEvent extends Event {

    private static final HandlerList HANDLERS = new HandlerList();

    private final String botName;
    private final Location deathLocation;
    private final int totalDeaths;

    public BotDeathEvent(String botName, Location deathLocation, int totalDeaths) {
        this.botName = botName;
        this.deathLocation = deathLocation;
        this.totalDeaths = totalDeaths;
    }

    public String getBotName() { return botName; }
    public Location getDeathLocation() { return deathLocation; }
    public int getTotalDeaths() { return totalDeaths; }

    @Override
    public HandlerList getHandlers() { return HANDLERS; }
    public static HandlerList getHandlerList() { return HANDLERS; }
}
