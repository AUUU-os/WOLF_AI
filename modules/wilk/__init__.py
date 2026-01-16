"""
WILK Module - Dolphin Uncensored Integration

4 tryby operacyjne:
- OGARNIACZ (Hustler) - szybka diagnoza
- TECHNIK (Hacker) - głęboki kod
- ZIOMEK (Bro) - lojalność
- STRAŻNIK (Wolf) - ochrona
"""

from .dolphin import ask_dolphin, stream_dolphin, DolphinClient
from .modes import Ogarniacz, Technik, Ziomek, Straznik, get_wilk
from .prompts import SYSTEM_PROMPTS, get_prompt

__all__ = [
    "ask_dolphin",
    "stream_dolphin",
    "DolphinClient",
    "Ogarniacz",
    "Technik",
    "Ziomek",
    "Straznik",
    "get_wilk",
    "SYSTEM_PROMPTS",
    "get_prompt"
]
