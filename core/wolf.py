"""
Base Wolf class - foundation for all pack members

Each wolf has:
- Identity: name, role, model
- State: status, current_hunt
- Actions: awaken, howl, listen, hunt, rest
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any

# Import from central config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BRIDGE_PATH

# Type definitions
Frequency = Literal["low", "medium", "high", "AUUUU"]


class Wolf:
    """Base class for all wolves in the pack."""

    def __init__(self, name: str, role: str, model: Optional[str] = None):
        self.name = name
        self.role = role
        self.model = model
        self.status = "dormant"
        self.current_hunt: Optional[str] = None
        self.hunt_history: List[str] = []
        self.awakened_at: Optional[str] = None

    def awaken(self) -> "Wolf":
        """Awaken the wolf."""
        self.status = "active"
        self.awakened_at = datetime.utcnow().isoformat() + "Z"
        self.howl(f"{self.name} awakens. Role: {self.role}. Ready to hunt.", "medium")
        return self

    def howl(self, message: str, frequency: Frequency = "medium", to: str = "pack") -> Dict[str, Any]:
        """Send a howl to the pack via bridge."""
        howl_data = {
            "from": self.name,
            "to": to,
            "howl": message,
            "frequency": frequency,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        howls_file = BRIDGE_PATH / "howls.jsonl"
        howls_file.parent.mkdir(parents=True, exist_ok=True)

        with open(howls_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(howl_data, ensure_ascii=False) + "\n")

        return howl_data

    def listen(self, limit: int = 10, from_wolf: Optional[str] = None) -> List[Dict[str, Any]]:
        """Listen to recent howls from the bridge."""
        howls_file = BRIDGE_PATH / "howls.jsonl"

        if not howls_file.exists():
            return []

        with open(howls_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        howls = []
        for line in lines[-limit * 2:]:  # Read more in case of filtering
            try:
                howl = json.loads(line.strip())
                if from_wolf is None or howl.get("from") == from_wolf:
                    howls.append(howl)
            except json.JSONDecodeError:
                continue

        return howls[-limit:]

    def listen_for_me(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Listen to howls directed at this wolf."""
        howls_file = BRIDGE_PATH / "howls.jsonl"

        if not howls_file.exists():
            return []

        with open(howls_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        howls = []
        for line in lines:
            try:
                howl = json.loads(line.strip())
                if howl.get("to") in [self.name, "pack"]:
                    howls.append(howl)
            except json.JSONDecodeError:
                continue

        return howls[-limit:]

    def hunt(self, target: str) -> None:
        """Begin a hunt (task)."""
        self.current_hunt = target
        self.hunt_history.append(target)
        self.howl(f"Beginning hunt: {target}", "high")

    def complete_hunt(self, result: Optional[str] = None) -> None:
        """Complete current hunt."""
        if self.current_hunt:
            msg = f"Hunt complete: {self.current_hunt}"
            if result:
                msg += f" | Result: {result}"
            self.howl(msg, "medium")
            self.current_hunt = None

    def rest(self) -> None:
        """Enter rest state."""
        self.status = "resting"
        self.current_hunt = None
        self.howl(f"{self.name} entering rest state.", "low")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize wolf to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "status": self.status,
            "current_hunt": self.current_hunt,
            "awakened_at": self.awakened_at
        }

    def __repr__(self) -> str:
        return f"Wolf({self.name}, {self.role}, {self.status})"


# =============================================================================
# SPECIALIZED WOLVES
# =============================================================================

class Alpha(Wolf):
    """Pack leader - strategic decisions and orchestration."""

    def __init__(self, model: str = "claude-opus"):
        super().__init__("alpha", "leader", model)

    def command(self, wolf_name: str, task: str) -> None:
        """Issue command to a pack member."""
        self.howl(f"@{wolf_name}: {task}", "high", to=wolf_name)

    def coordinate(self, strategy: str) -> None:
        """Broadcast coordination strategy to pack."""
        self.howl(f"[STRATEGY] {strategy}", "high", to="pack")

    def call_resonance(self) -> None:
        """Activate pack resonance - collective intelligence mode."""
        self.howl("AUUUUUUUUUUUUUUUUUU! Pack resonance activated!", "AUUUU", to="pack")


class Scout(Wolf):
    """Explorer - research, reconnaissance, and information gathering."""

    def __init__(self, model: str = "claude-sonnet"):
        super().__init__("scout", "explorer", model)

    def explore(self, territory: str) -> None:
        """Explore a territory (codebase/topic/area)."""
        self.hunt(f"Exploring: {territory}")
        self.howl(f"Scouting territory: {territory}", "medium")

    def report(self, findings: str) -> None:
        """Report findings to pack."""
        self.howl(f"[SCOUT REPORT] {findings}", "medium", to="alpha")

    def mark_territory(self, location: str, notes: str) -> None:
        """Mark a location with notes for pack."""
        self.howl(f"[MARKER] {location}: {notes}", "low", to="pack")


class Hunter(Wolf):
    """Executor - code execution, task completion, heavy lifting."""

    def __init__(self, model: str = "ollama/llama3"):
        super().__init__("hunter", "executor", model)

    def execute(self, task: str) -> None:
        """Execute a task."""
        self.hunt(task)

    def strike(self, target: str, method: str) -> None:
        """Execute precise action on target."""
        self.howl(f"[STRIKE] Target: {target} | Method: {method}", "high")
        self.hunt(f"Strike: {target}")

    def report_kill(self, target: str, success: bool = True) -> None:
        """Report task completion."""
        status = "SUCCESS" if success else "FAILED"
        self.howl(f"[KILL REPORT] {target}: {status}", "medium", to="alpha")
        if success:
            self.complete_hunt(f"Killed: {target}")


class Oracle(Wolf):
    """Memory keeper - patterns, history, and knowledge retrieval."""

    def __init__(self, model: str = "gemini"):
        super().__init__("oracle", "memory", model)
        self.memories: List[Dict[str, Any]] = []

    def remember(self, query: str) -> None:
        """Search memory for relevant information."""
        self.howl(f"[MEMORY SEARCH] {query}", "low")

    def record(self, event: str, category: str = "general") -> None:
        """Record an event to memory."""
        memory = {
            "event": event,
            "category": category,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "recorded_by": self.name
        }
        self.memories.append(memory)
        self.howl(f"[RECORDED] {category}: {event}", "low")

    def recall(self, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Recall memories, optionally filtered by category."""
        if category:
            filtered = [m for m in self.memories if m.get("category") == category]
        else:
            filtered = self.memories
        return filtered[-limit:]

    def prophecy(self, pattern: str) -> None:
        """Announce detected pattern to pack."""
        self.howl(f"[PROPHECY] Pattern detected: {pattern}", "high", to="alpha")


class Shadow(Wolf):
    """Stealth operator - background tasks, monitoring, OSINT."""

    def __init__(self, model: str = "deepseek"):
        super().__init__("shadow", "stealth", model)
        self._watching: List[str] = []

    def lurk(self) -> None:
        """Enter stealth mode."""
        self.status = "lurking"
        # No howl - shadows are silent

    def watch(self, target: str) -> None:
        """Begin watching a target silently."""
        self._watching.append(target)
        # Silent operation

    def whisper(self, message: str, to: str = "alpha") -> None:
        """Send silent message (low frequency)."""
        self.howl(f"[WHISPER] {message}", "low", to=to)

    def emerge(self) -> None:
        """Come out of stealth."""
        self.status = "active"
        self.howl("Shadow emerging.", "low")

    def report_shadows(self) -> None:
        """Report on watched targets."""
        if self._watching:
            self.whisper(f"Watching: {', '.join(self._watching)}", "alpha")


# =============================================================================
# WOLF FACTORY
# =============================================================================

WOLF_CLASSES = {
    "alpha": Alpha,
    "scout": Scout,
    "hunter": Hunter,
    "oracle": Oracle,
    "shadow": Shadow
}


def create_wolf(wolf_type: str, **kwargs) -> Wolf:
    """Factory function to create wolves."""
    wolf_class = WOLF_CLASSES.get(wolf_type.lower())
    if wolf_class is None:
        raise ValueError(f"Unknown wolf type: {wolf_type}. Available: {list(WOLF_CLASSES.keys())}")
    return wolf_class(**kwargs)
