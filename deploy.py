#!/usr/bin/env python3
"""
Deploy MinimalAI plugin to the local Minecraft server.

Usage:
    python deploy.py          # Build and copy to server
    python deploy.py --skip   # Copy without rebuilding
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
BUILD_DIR = PROJECT_ROOT / "build" / "libs"
JAR_NAME = "MinimalAI-1.0.0-reobf.jar"
SERVER_PLUGINS = Path("C:/Users/henry/Games/Minecraft/Server/plugins")


def build():
    print("Building plugin...")
    result = subprocess.run(
        ["./gradlew", "build"],
        cwd=PROJECT_ROOT,
        shell=True,
    )
    if result.returncode != 0:
        print("Build failed!")
        sys.exit(1)
    print("Build successful.")


def deploy():
    jar = BUILD_DIR / JAR_NAME
    if not jar.exists():
        print(f"Jar not found: {jar}")
        print("Run without --skip to build first.")
        sys.exit(1)

    if not SERVER_PLUGINS.exists():
        print(f"Server plugins directory not found: {SERVER_PLUGINS}")
        sys.exit(1)

    dest = SERVER_PLUGINS / JAR_NAME
    shutil.copy2(jar, dest)
    print(f"Deployed {JAR_NAME} -> {dest}")


def main():
    skip_build = "--skip" in sys.argv

    if not skip_build:
        build()

    deploy()
    print("Done! Restart the server to load the new plugin.")


if __name__ == "__main__":
    main()
