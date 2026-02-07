"""
Howl Module - Advanced communication for WOLF_AI

Handles:
- Message routing and filtering
- Broadcast channels
- Priority messaging
- Message history and search
- Notification system
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BRIDGE_PATH


class Frequency(Enum):
    """Message frequency/priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUUUU = "AUUUU"  # Pack-wide alert


@dataclass
class Howl:
    """Represents a single howl (message)."""
    from_wolf: str
    to: str
    message: str
    frequency: Frequency = Frequency.MEDIUM
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_wolf,
            "to": self.to,
            "howl": self.message,
            "frequency": self.frequency.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Howl":
        return cls(
            from_wolf=data.get("from", "unknown"),
            to=data.get("to", "pack"),
            message=data.get("howl", ""),
            frequency=Frequency(data.get("frequency", "medium")),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


class HowlChannel:
    """A communication channel for howls."""

    def __init__(self, name: str):
        self.name = name
        self.subscribers: List[Callable[[Howl], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[Howl], None]):
        """Subscribe to this channel."""
        with self._lock:
            self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Howl], None]):
        """Unsubscribe from this channel."""
        with self._lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)

    def broadcast(self, howl: Howl):
        """Broadcast a howl to all subscribers."""
        with self._lock:
            for callback in self.subscribers:
                try:
                    callback(howl)
                except Exception as e:
                    print(f"[HowlChannel:{self.name}] Callback error: {e}")


class HowlBridge:
    """Central howl management system."""

    def __init__(self):
        self.howls_file = BRIDGE_PATH / "howls.jsonl"
        self.channels: Dict[str, HowlChannel] = {}
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        """Ensure howls file exists."""
        self.howls_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.howls_file.exists():
            self.howls_file.touch()

    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================

    def howl(self, from_wolf: str, message: str, to: str = "pack",
             frequency: str = "medium", tags: List[str] = None,
             metadata: Dict[str, Any] = None) -> Howl:
        """Send a howl."""
        h = Howl(
            from_wolf=from_wolf,
            to=to,
            message=message,
            frequency=Frequency(frequency),
            tags=tags or [],
            metadata=metadata or {}
        )

        # Write to file
        with self._lock:
            with open(self.howls_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(h.to_dict(), ensure_ascii=False) + "\n")

        # Broadcast to channels
        self._broadcast(h)

        return h

    def _broadcast(self, howl: Howl):
        """Broadcast howl to relevant channels."""
        # Broadcast to specific channel
        if howl.to in self.channels:
            self.channels[howl.to].broadcast(howl)

        # Always broadcast to 'pack' channel
        if "pack" in self.channels and howl.to != "pack":
            self.channels["pack"].broadcast(howl)

        # AUUUU goes to all channels
        if howl.frequency == Frequency.AUUUU:
            for channel in self.channels.values():
                channel.broadcast(howl)

    # =========================================================================
    # READING HOWLS
    # =========================================================================

    def listen(self, limit: int = 50, from_wolf: Optional[str] = None,
               to: Optional[str] = None, frequency: Optional[str] = None,
               since: Optional[datetime] = None,
               tags: Optional[List[str]] = None) -> List[Howl]:
        """Listen to howls with filters."""
        howls = self._read_all()

        # Apply filters
        if from_wolf:
            howls = [h for h in howls if h.from_wolf == from_wolf]

        if to:
            howls = [h for h in howls if h.to == to or h.to == "pack"]

        if frequency:
            howls = [h for h in howls if h.frequency.value == frequency]

        if since:
            since_str = since.isoformat()
            howls = [h for h in howls if h.timestamp >= since_str]

        if tags:
            howls = [h for h in howls if any(t in h.tags for t in tags)]

        return howls[-limit:]

    def _read_all(self) -> List[Howl]:
        """Read all howls from file."""
        if not self.howls_file.exists():
            return []

        howls = []
        with open(self.howls_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    howls.append(Howl.from_dict(data))
                except json.JSONDecodeError:
                    continue

        return howls

    def get_recent(self, hours: int = 24) -> List[Howl]:
        """Get howls from the last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)
        return self.listen(limit=1000, since=since)

    def search(self, query: str, limit: int = 50) -> List[Howl]:
        """Search howls by content."""
        howls = self._read_all()
        query_lower = query.lower()

        matches = [
            h for h in howls
            if query_lower in h.message.lower()
        ]

        return matches[-limit:]

    # =========================================================================
    # CHANNELS
    # =========================================================================

    def create_channel(self, name: str) -> HowlChannel:
        """Create or get a channel."""
        if name not in self.channels:
            self.channels[name] = HowlChannel(name)
        return self.channels[name]

    def subscribe(self, channel_name: str, callback: Callable[[Howl], None]):
        """Subscribe to a channel."""
        channel = self.create_channel(channel_name)
        channel.subscribe(callback)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def alert(self, from_wolf: str, message: str) -> Howl:
        """Send high-priority alert to pack."""
        return self.howl(from_wolf, f"[ALERT] {message}", "pack", "high")

    def resonance(self, from_wolf: str, message: str) -> Howl:
        """Send AUUUU resonance call to all."""
        return self.howl(from_wolf, f"AUUUUUUUUUUUUUUUUUU! {message}",
                        "world", "AUUUU")

    def whisper(self, from_wolf: str, to: str, message: str) -> Howl:
        """Send low-priority private message."""
        return self.howl(from_wolf, message, to, "low")

    def report(self, from_wolf: str, category: str, content: str) -> Howl:
        """Send a report howl."""
        return self.howl(
            from_wolf,
            f"[{category.upper()}] {content}",
            "alpha",
            "medium",
            tags=[category, "report"]
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get howl statistics."""
        howls = self._read_all()

        stats = {
            "total_howls": len(howls),
            "by_wolf": {},
            "by_frequency": {f.value: 0 for f in Frequency},
            "last_24h": 0,
            "last_hour": 0
        }

        now = datetime.utcnow()
        hour_ago = (now - timedelta(hours=1)).isoformat()
        day_ago = (now - timedelta(hours=24)).isoformat()

        for h in howls:
            # By wolf
            if h.from_wolf not in stats["by_wolf"]:
                stats["by_wolf"][h.from_wolf] = 0
            stats["by_wolf"][h.from_wolf] += 1

            # By frequency
            stats["by_frequency"][h.frequency.value] += 1

            # Time-based
            if h.timestamp >= day_ago:
                stats["last_24h"] += 1
            if h.timestamp >= hour_ago:
                stats["last_hour"] += 1

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_bridge: Optional[HowlBridge] = None


def get_bridge() -> HowlBridge:
    """Get or create the howl bridge."""
    global _bridge
    if _bridge is None:
        _bridge = HowlBridge()
    return _bridge


def howl(from_wolf: str, message: str, to: str = "pack",
         frequency: str = "medium") -> Dict[str, Any]:
    """Quick howl function."""
    bridge = get_bridge()
    h = bridge.howl(from_wolf, message, to, frequency)
    return h.to_dict()


def listen(limit: int = 20, from_wolf: Optional[str] = None) -> List[Dict[str, Any]]:
    """Listen to recent howls."""
    bridge = get_bridge()
    howls = bridge.listen(limit=limit, from_wolf=from_wolf)
    return [h.to_dict() for h in howls]


def alert(from_wolf: str, message: str) -> Dict[str, Any]:
    """Send an alert."""
    bridge = get_bridge()
    h = bridge.alert(from_wolf, message)
    return h.to_dict()


def resonance(message: str) -> Dict[str, Any]:
    """Send pack-wide resonance."""
    bridge = get_bridge()
    h = bridge.resonance("pack", message)
    return h.to_dict()
