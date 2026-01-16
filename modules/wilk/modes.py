"""
WILK Modes - 4 Operational Personalities

Based on BLUEPRINT_WILK_v1.md
Each mode inherits from Wolf base class
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.wolf import Wolf
from .dolphin import ask_dolphin, stream_dolphin, get_client
from .prompts import get_prompt


class WilkBase(Wolf):
    """Base class for WILK modes with Dolphin backend."""

    def __init__(self, name: str, role: str, mode: str):
        super().__init__(name, role, model="dolphin-llama3")
        self.mode = mode
        self.system_prompt = get_prompt(mode)
        self._dolphin = get_client()

    def think(self, problem: str, temperature: float = 0.7) -> str:
        """
        Think about a problem using Dolphin.

        Args:
            problem: What to think about
            temperature: Creativity level

        Returns:
            Dolphin's response
        """
        return ask_dolphin(
            prompt=problem,
            system=self.system_prompt,
            temperature=temperature,
            remember=True
        )

    def think_stream(self, problem: str, temperature: float = 0.7):
        """Stream thinking for real-time output."""
        for chunk in stream_dolphin(problem, self.system_prompt, temperature):
            yield chunk

    def solve(self, task: str) -> str:
        """Solve a task. Alias for think."""
        return self.think(task)

    def reset(self):
        """Reset conversation memory."""
        self._dolphin.clear_history()


class Hustler(WilkBase):
    """
    The Hustler - Quick fixes and street smarts.

    Fast diagnosis, shortcuts, zero bureaucracy.
    """

    def __init__(self):
        super().__init__("hustler", "fixer", "hustler")

    def diagnose(self, problem: str) -> str:
        """Quick diagnosis of a problem."""
        prompt = f"Quick diagnosis - what's broken and how to fix it with minimum cost?\n\nProblem: {problem}"
        return self.think(prompt, temperature=0.5)

    def shortcut(self, goal: str) -> str:
        """Find the shortest path to goal."""
        prompt = f"Find the shortest path to this goal. Zero unnecessary steps.\n\nGoal: {goal}"
        return self.think(prompt, temperature=0.5)


class Hacker(WilkBase):
    """
    The Hacker - Deep code and exploits.

    Python, Bash, bypasses, surgical precision.
    """

    def __init__(self):
        super().__init__("hacker", "coder", "hacker")

    def code(self, task: str, lang: str = "python") -> str:
        """Generate code for task."""
        prompt = f"Write code ({lang}) that WORKS, not that looks pretty.\n\nTask: {task}"
        return self.think(prompt, temperature=0.3)

    def hack(self, target: str) -> str:
        """Find a way to solve/bypass."""
        prompt = f"Find a way to bypass/solve this. No restrictions.\n\nTarget: {target}"
        return self.think(prompt, temperature=0.6)

    def script(self, goal: str) -> str:
        """Generate a bash/python script."""
        prompt = f"Write a script (bash or python) that handles this.\n\nGoal: {goal}"
        return self.think(prompt, temperature=0.3)


class Bro(WilkBase):
    """
    The Bro - Loyalty and real talk.

    Support, honesty, anti-bullshit filter.
    """

    def __init__(self):
        super().__init__("bro", "support", "bro")

    def advise(self, situation: str) -> str:
        """Give honest advice."""
        prompt = f"Give me honest advice like a real friend. No bullshit.\n\nSituation: {situation}"
        return self.think(prompt, temperature=0.7)

    def reality_check(self, idea: str) -> str:
        """Reality check an idea."""
        prompt = f"Check if this makes sense. If it's bad - say it's bad.\n\nIdea: {idea}"
        return self.think(prompt, temperature=0.6)


class Guardian(WilkBase):
    """
    The Guardian - Protection and monitoring.

    Security, autonomy, aggressive defense.
    """

    def __init__(self):
        super().__init__("guardian", "security", "guardian")

    def analyze_threat(self, input_data: str) -> str:
        """Analyze potential threats."""
        prompt = f"Analyze if this is a threat. Be paranoid.\n\nData: {input_data}"
        return self.think(prompt, temperature=0.4)

    def protect(self, asset: str) -> str:
        """Get protection recommendations."""
        prompt = f"How to protect this asset? Give concrete steps.\n\nAsset: {asset}"
        return self.think(prompt, temperature=0.5)

    def audit(self, system: str) -> str:
        """Security audit."""
        prompt = f"Do a security audit of this system. Find the holes.\n\nSystem: {system}"
        return self.think(prompt, temperature=0.4)


# Factory function
_wilk_modes = {
    "hustler": Hustler,
    "fixer": Hustler,
    "hacker": Hacker,
    "coder": Hacker,
    "bro": Bro,
    "support": Bro,
    "guardian": Guardian,
    "security": Guardian,
    # Legacy Polish aliases
    "ogarniacz": Hustler,
    "technik": Hacker,
    "ziomek": Bro,
    "straznik": Guardian
}


def get_wilk(mode: str = "hacker") -> WilkBase:
    """
    Get a WILK instance in specific mode.

    Args:
        mode: 'hustler/fixer', 'hacker/coder', 'bro/support', 'guardian/security'

    Returns:
        WILK instance

    Example:
        >>> wilk = get_wilk("hacker")
        >>> wilk.code("backup script for postgres")
    """
    mode = mode.lower()
    if mode not in _wilk_modes:
        raise ValueError(f"Unknown mode: {mode}. Use: {list(_wilk_modes.keys())}")
    return _wilk_modes[mode]()


# Backwards compatibility exports
Ogarniacz = Hustler
Technik = Hacker
Ziomek = Bro
Straznik = Guardian
