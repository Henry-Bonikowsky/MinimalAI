package com.minimalai;

import com.minimalai.ui.OverlayRenderer;
import net.fabricmc.api.ClientModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MinimalAI implements ClientModInitializer {
    public static final String MOD_ID = "minimalai";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitializeClient() {
        LOGGER.info("MinimalAI initializing...");

        ModKeybinds.register();
        OverlayRenderer.init();

        LOGGER.info("MinimalAI initialized");
    }
}
