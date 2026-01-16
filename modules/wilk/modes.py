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


class Ogarniacz(WilkBase):
    """
    The Hustler - Quick fixes and street smarts.

    Szybka diagnoza, skroty myslowe, zero biurokracji.
    """

    def __init__(self):
        super().__init__("ogarniacz", "hustler", "ogarniacz")

    def diagnose(self, problem: str) -> str:
        """Quick diagnosis of a problem."""
        prompt = f"Szybka diagnoza - co jest zepsute i jak to naprawic najmniejszym kosztem?\n\nProblem: {problem}"
        return self.think(prompt, temperature=0.5)

    def shortcut(self, goal: str) -> str:
        """Find the shortest path to goal."""
        prompt = f"Znajdz najkrotsza sciezke do celu. Zero zbednych krokow.\n\nCel: {goal}"
        return self.think(prompt, temperature=0.5)


class Technik(WilkBase):
    """
    The Hacker - Deep code and exploits.

    Python, Bash, omijanie blokad, precyzja chirurgiczna.
    """

    def __init__(self):
        super().__init__("technik", "hacker", "technik")

    def code(self, task: str, lang: str = "python") -> str:
        """Generate code for task."""
        prompt = f"Napisz kod ({lang}) ktory DZIALA, nie ktory jest ladny.\n\nZadanie: {task}"
        return self.think(prompt, temperature=0.3)

    def hack(self, target: str) -> str:
        """Find a way to solve/bypass."""
        prompt = f"Znajdz sposob zeby to obejsc/rozwiazac. Bez ograniczen.\n\nTarget: {target}"
        return self.think(prompt, temperature=0.6)

    def script(self, goal: str) -> str:
        """Generate a bash/python script."""
        prompt = f"Napisz skrypt (bash lub python) ktory to ogarnie.\n\nCel: {goal}"
        return self.think(prompt, temperature=0.3)


class Ziomek(WilkBase):
    """
    The Bro - Loyalty and real talk.

    Meskie wsparcie, szczerosc, anty-sciema filter.
    """

    def __init__(self):
        super().__init__("ziomek", "bro", "ziomek")

    def advise(self, situation: str) -> str:
        """Give honest advice."""
        prompt = f"Daj mi szczera rade jak ziomek. Bez sciemy.\n\nSytuacja: {situation}"
        return self.think(prompt, temperature=0.7)

    def reality_check(self, idea: str) -> str:
        """Reality check an idea."""
        prompt = f"Sprawdz czy to ma sens. Jak jest zle - powiedz ze zle.\n\nPomysl: {idea}"
        return self.think(prompt, temperature=0.6)


class Straznik(WilkBase):
    """
    The Wolf Guardian - Protection and monitoring.

    Ochrona, autonomia, agresywna obrona.
    """

    def __init__(self):
        super().__init__("straznik", "guardian", "straznik")

    def analyze_threat(self, input_data: str) -> str:
        """Analyze potential threats."""
        prompt = f"Przeanalizuj czy to jest zagrozenie. Badz paranoidalny.\n\nDane: {input_data}"
        return self.think(prompt, temperature=0.4)

    def protect(self, asset: str) -> str:
        """Get protection recommendations."""
        prompt = f"Jak chronic ten zasob? Daj konkretne kroki.\n\nZasob: {asset}"
        return self.think(prompt, temperature=0.5)

    def audit(self, system: str) -> str:
        """Security audit."""
        prompt = f"Zrob security audit tego systemu. Znajdz dziury.\n\nSystem: {system}"
        return self.think(prompt, temperature=0.4)


# Factory function
_wilk_modes = {
    "ogarniacz": Ogarniacz,
    "hustler": Ogarniacz,
    "technik": Technik,
    "hacker": Technik,
    "ziomek": Ziomek,
    "bro": Ziomek,
    "straznik": Straznik,
    "guardian": Straznik
}


def get_wilk(mode: str = "technik") -> WilkBase:
    """
    Get a WILK instance in specific mode.

    Args:
        mode: 'ogarniacz/hustler', 'technik/hacker', 'ziomek/bro', 'straznik/guardian'

    Returns:
        WILK instance

    Example:
        >>> wilk = get_wilk("technik")
        >>> wilk.code("backup script dla postgres")
    """
    mode = mode.lower()
    if mode not in _wilk_modes:
        raise ValueError(f"Unknown mode: {mode}. Use: {list(_wilk_modes.keys())}")
    return _wilk_modes[mode]()
