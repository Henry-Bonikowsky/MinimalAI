"""Pytest configuration - add training directory to path."""

import sys
import os

# Add training directory to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
