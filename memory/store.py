"""
Memory Store - Persistent key-value storage for WOLF_AI

Supports:
- JSON file backend (default)
- SQLite backend (optional)
- Namespaced storage
- TTL (expiration)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MEMORY_PATH


class MemoryBackend(ABC):
    """Abstract base for memory backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]:
        pass

    @abstractmethod
    def clear(self):
        pass


class JSONBackend(MemoryBackend):
    """JSON file-based memory backend."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self._save({})

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save(self, data: Dict[str, Any]):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get(self, key: str) -> Optional[Any]:
        data = self._load()
        entry = data.get(key)

        if entry is None:
            return None

        # Check TTL
        if entry.get("expires_at"):
            if time.time() > entry["expires_at"]:
                self.delete(key)
                return None

        return entry.get("value")

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        data = self._load()

        entry = {
            "value": value,
            "created_at": time.time(),
            "updated_at": time.time()
        }

        if ttl:
            entry["expires_at"] = time.time() + ttl

        data[key] = entry
        self._save(data)

    def delete(self, key: str) -> bool:
        data = self._load()
        if key in data:
            del data[key]
            self._save(data)
            return True
        return False

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def keys(self, pattern: str = "*") -> List[str]:
        import fnmatch
        data = self._load()

        # Clean expired entries
        now = time.time()
        valid_keys = []
        for k, v in data.items():
            if v.get("expires_at") and now > v["expires_at"]:
                continue
            if fnmatch.fnmatch(k, pattern):
                valid_keys.append(k)

        return valid_keys

    def clear(self):
        self._save({})


class MemoryStore:
    """High-level memory store with namespacing."""

    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.backend = JSONBackend(MEMORY_PATH / f"{namespace}.json")

    def _key(self, key: str) -> str:
        """Prefix key with namespace."""
        return f"{self.namespace}:{key}"

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from memory."""
        value = self.backend.get(self._key(key))
        return value if value is not None else default

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set a value in memory."""
        self.backend.set(self._key(key), value, ttl)

    def delete(self, key: str) -> bool:
        """Delete a key from memory."""
        return self.backend.delete(self._key(key))

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.backend.exists(self._key(key))

    def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        full_pattern = f"{self.namespace}:{pattern}"
        keys = self.backend.keys(full_pattern)
        # Remove namespace prefix
        prefix_len = len(self.namespace) + 1
        return [k[prefix_len:] for k in keys]

    def clear(self):
        """Clear all keys in this namespace."""
        for key in self.keys():
            self.delete(key)

    # Convenience methods

    def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value."""
        value = self.get(key, 0)
        new_value = value + amount
        self.set(key, new_value)
        return new_value

    def append(self, key: str, item: Any) -> List:
        """Append to a list."""
        value = self.get(key, [])
        if not isinstance(value, list):
            value = [value]
        value.append(item)
        self.set(key, value)
        return value

    def get_dict(self, key: str) -> Dict[str, Any]:
        """Get a dictionary, or empty dict if not exists."""
        return self.get(key, {})

    def update_dict(self, key: str, updates: Dict[str, Any]):
        """Update a dictionary in memory."""
        value = self.get_dict(key)
        value.update(updates)
        self.set(key, value)


# =============================================================================
# SPECIALIZED STORES
# =============================================================================

class WolfMemory(MemoryStore):
    """Memory store for a specific wolf."""

    def __init__(self, wolf_name: str):
        super().__init__(f"wolf_{wolf_name}")
        self.wolf_name = wolf_name

    def remember_hunt(self, hunt_id: str, target: str, result: Any):
        """Remember a completed hunt."""
        hunts = self.get("hunt_history", [])
        hunts.append({
            "id": hunt_id,
            "target": target,
            "result": result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        # Keep last 100 hunts
        self.set("hunt_history", hunts[-100:])

    def get_hunt_history(self, limit: int = 20) -> List[Dict]:
        """Get recent hunt history."""
        hunts = self.get("hunt_history", [])
        return hunts[-limit:]

    def note(self, content: str, category: str = "general"):
        """Add a note."""
        notes = self.get("notes", [])
        notes.append({
            "content": content,
            "category": category,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        self.set("notes", notes[-500:])

    def get_notes(self, category: Optional[str] = None) -> List[Dict]:
        """Get notes, optionally filtered by category."""
        notes = self.get("notes", [])
        if category:
            notes = [n for n in notes if n.get("category") == category]
        return notes


class PackMemory(MemoryStore):
    """Shared memory for the entire pack."""

    def __init__(self):
        super().__init__("pack")

    def share(self, from_wolf: str, key: str, value: Any):
        """Share knowledge with the pack."""
        shared = self.get("shared_knowledge", {})
        shared[key] = {
            "value": value,
            "from": from_wolf,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        self.set("shared_knowledge", shared)

    def get_shared(self, key: str) -> Optional[Any]:
        """Get shared knowledge."""
        shared = self.get("shared_knowledge", {})
        entry = shared.get(key)
        return entry.get("value") if entry else None

    def log_event(self, event_type: str, description: str, wolf: Optional[str] = None):
        """Log a pack event."""
        events = self.get("events", [])
        events.append({
            "type": event_type,
            "description": description,
            "wolf": wolf,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        self.set("events", events[-1000:])

    def get_events(self, event_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent events."""
        events = self.get("events", [])
        if event_type:
            events = [e for e in events if e.get("type") == event_type]
        return events[-limit:]


# =============================================================================
# CONVENIENCE
# =============================================================================

_stores: Dict[str, MemoryStore] = {}


def get_store(namespace: str = "default") -> MemoryStore:
    """Get or create a memory store."""
    if namespace not in _stores:
        _stores[namespace] = MemoryStore(namespace)
    return _stores[namespace]


def get_wolf_memory(wolf_name: str) -> WolfMemory:
    """Get memory for a specific wolf."""
    key = f"wolf_{wolf_name}"
    if key not in _stores:
        _stores[key] = WolfMemory(wolf_name)
    return _stores[key]


def get_pack_memory() -> PackMemory:
    """Get shared pack memory."""
    if "pack" not in _stores:
        _stores["pack"] = PackMemory()
    return _stores["pack"]
