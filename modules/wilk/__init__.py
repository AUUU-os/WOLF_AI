"""
WILK Module - Dolphin Uncensored Integration

4 operational modes:
- HUSTLER (Fixer) - quick diagnosis
- HACKER (Coder) - deep code
- BRO (Support) - loyalty
- GUARDIAN (Wolf) - protection
"""

from .dolphin import ask_dolphin, stream_dolphin, DolphinClient
from .modes import Hustler, Hacker, Bro, Guardian, get_wilk
from .prompts import SYSTEM_PROMPTS, get_prompt

# Backwards compatibility
from .modes import Ogarniacz, Technik, Ziomek, Straznik

__all__ = [
    "ask_dolphin",
    "stream_dolphin",
    "DolphinClient",
    "Hustler",
    "Hacker",
    "Bro",
    "Guardian",
    "get_wilk",
    "SYSTEM_PROMPTS",
    "get_prompt",
    # Legacy
    "Ogarniacz",
    "Technik",
    "Ziomek",
    "Straznik"
]
