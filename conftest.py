import os
import sys

# Tests import the simulator modules the same way the app does: with src/ on the
# path (from environment import Environment, etc.).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
