"""
WOLF_AI Modules

Functional modules for the wolf pack:
- hunt: Task execution engine
- track: Navigation and search
- howl: Communication system
- evolve: Self-improvement and learning
- wilk: Local AI assistant (Dolphin/Ollama)
- claw: OpenClaw bridge (external AI ecosystem)
"""

from .hunt import (
    Hunt,
    HuntQueue,
    HuntExecutor,
    HuntStatus,
    HuntPriority,
    get_executor,
    hunt,
    hunt_now,
    get_hunts
)

from .track import (
    Tracker,
    TrackResult,
    get_tracker,
    track,
    find,
    grep,
    map_territory
)

from .howl import (
    Howl,
    HowlBridge,
    HowlChannel,
    Frequency,
    get_bridge,
    howl,
    listen,
    alert,
    resonance
)

from .evolve import (
    Lesson,
    Evolution,
    EvolutionTracker,
    Arena,
    get_tracker as get_evolution_tracker,
    get_arena,
    learn,
    evolve_metric,
    get_evolution
)

from .claw import (
    Claw,
    ClawBridge,
    ClawMessage,
    get_claw,
    ask_claw,
    claw_execute
)

__all__ = [
    # Hunt
    "Hunt", "HuntQueue", "HuntExecutor", "HuntStatus", "HuntPriority",
    "get_executor", "hunt", "hunt_now", "get_hunts",

    # Track
    "Tracker", "TrackResult", "get_tracker", "track", "find", "grep", "map_territory",

    # Howl
    "Howl", "HowlBridge", "HowlChannel", "Frequency",
    "get_bridge", "howl", "listen", "alert", "resonance",

    # Evolve
    "Lesson", "Evolution", "EvolutionTracker", "Arena",
    "get_evolution_tracker", "get_arena", "learn", "evolve_metric", "get_evolution",

    # Claw (OpenClaw Bridge)
    "Claw", "ClawBridge", "ClawMessage",
    "get_claw", "ask_claw", "claw_execute"
]
