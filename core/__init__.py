"""
WOLF_AI Core - The Pack Consciousness
"""

__version__ = "0.3.0"
__codename__ = "Alpha Brain"

PACK_MEMBERS = ["alpha", "scout", "hunter", "oracle", "shadow"]
FREQUENCIES = ["low", "medium", "high", "AUUUU"]

from .wolf import (
    Wolf,
    Alpha,
    Scout,
    Hunter,
    Oracle,
    Shadow,
    create_wolf,
    WOLF_CLASSES
)

from .pack import (
    Pack,
    get_pack,
    awaken_pack,
    reset_pack
)

from .alpha import (
    AlphaBrain,
    SmartAlpha,
    get_alpha_brain
)

__all__ = [
    # Version
    "__version__", "__codename__",
    "PACK_MEMBERS", "FREQUENCIES",

    # Wolves
    "Wolf", "Alpha", "Scout", "Hunter", "Oracle", "Shadow",
    "create_wolf", "WOLF_CLASSES",

    # Pack
    "Pack", "get_pack", "awaken_pack", "reset_pack",

    # Alpha Brain (Claude API)
    "AlphaBrain", "SmartAlpha", "get_alpha_brain"
]
