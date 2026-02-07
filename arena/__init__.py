"""
WOLF_AI Arena - Training Ground

The arena is where wolves train, test, and evolve.
Challenges, simulations, and skill assessment.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-export from evolve module (Arena class lives there)
from modules.evolve import Arena, get_arena

__all__ = ["Arena", "get_arena"]
