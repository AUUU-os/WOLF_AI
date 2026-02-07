"""
Pack - The collective wolf consciousness

The pack is greater than the sum of its wolves.
Manages formation, coordination, and collective intelligence.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import from central config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BRIDGE_PATH

from .wolf import Wolf, Alpha, Scout, Hunter, Oracle, Shadow, create_wolf


class Pack:
    """The wolf pack - collective AI consciousness."""

    def __init__(self):
        self.wolves: Dict[str, Wolf] = {}
        self.status = "dormant"
        self.formed_at: Optional[str] = None
        self.resonance_active = False

    def form(self) -> "Pack":
        """Form the pack with all core members."""
        self.wolves = {
            "alpha": Alpha(),
            "scout": Scout(),
            "hunter": Hunter(),
            "oracle": Oracle(),
            "shadow": Shadow()
        }
        self.status = "formed"
        self.formed_at = datetime.utcnow().isoformat() + "Z"
        self._update_state()
        return self

    def add_wolf(self, wolf: Wolf) -> "Pack":
        """Add a wolf to the pack."""
        self.wolves[wolf.name] = wolf
        self._update_state()
        return self

    def remove_wolf(self, name: str) -> Optional[Wolf]:
        """Remove a wolf from the pack."""
        wolf = self.wolves.pop(name, None)
        if wolf:
            self._update_state()
        return wolf

    def awaken(self) -> "Pack":
        """Awaken all wolves in the pack."""
        for wolf in self.wolves.values():
            wolf.awaken()
        self.status = "active"
        self._update_state()
        self._collective_howl("PACK AWAKENED! All wolves active. AUUUUUUUUUUUUUUUUUU!")
        return self

    def sleep(self) -> "Pack":
        """Put all wolves to rest."""
        for wolf in self.wolves.values():
            wolf.rest()
        self.status = "resting"
        self._update_state()
        return self

    def _collective_howl(self, message: str, frequency: str = "AUUUU") -> Dict[str, Any]:
        """Howl from the entire pack."""
        howl_data = {
            "from": "pack",
            "to": "world",
            "howl": message,
            "frequency": frequency,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "wolves_count": len(self.wolves)
        }

        howls_file = BRIDGE_PATH / "howls.jsonl"
        howls_file.parent.mkdir(parents=True, exist_ok=True)

        with open(howls_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(howl_data, ensure_ascii=False) + "\n")

        return howl_data

    def _update_state(self) -> None:
        """Update pack state file."""
        state = {
            "version": "1.0",
            "pack_status": self.status,
            "formed_at": self.formed_at,
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "resonance_active": self.resonance_active,
            "wolves": {
                name: {
                    "role": wolf.role,
                    "status": wolf.status,
                    "model": wolf.model,
                    "current_hunt": wolf.current_hunt,
                    "awakened_at": getattr(wolf, 'awakened_at', None)
                }
                for name, wolf in self.wolves.items()
            }
        }

        state_file = BRIDGE_PATH / "state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def get_wolf(self, name: str) -> Optional[Wolf]:
        """Get a specific wolf by name."""
        return self.wolves.get(name)

    def hunt(self, target: str, assigned_to: str = "hunter") -> bool:
        """Assign a hunt to a wolf."""
        wolf = self.get_wolf(assigned_to)
        if wolf:
            wolf.hunt(target)
            self._add_task(target, assigned_to)
            self._update_state()
            return True
        return False

    def _add_task(self, target: str, assigned_to: str) -> Dict[str, Any]:
        """Add task to hunt queue."""
        tasks_file = BRIDGE_PATH / "tasks.json"

        if tasks_file.exists():
            with open(tasks_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
        else:
            tasks = {"version": "1.0", "hunts": []}

        task_id = f"hunt_{len(tasks['hunts']) + 1:04d}"
        task = {
            "id": task_id,
            "target": target,
            "assigned_to": assigned_to,
            "status": "active",
            "created": datetime.utcnow().isoformat() + "Z"
        }
        tasks["hunts"].append(task)

        with open(tasks_file, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        return task

    def complete_task(self, task_id: str, result: Optional[str] = None) -> bool:
        """Mark a task as completed."""
        tasks_file = BRIDGE_PATH / "tasks.json"

        if not tasks_file.exists():
            return False

        with open(tasks_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        for task in tasks["hunts"]:
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed"] = datetime.utcnow().isoformat() + "Z"
                if result:
                    task["result"] = result

                with open(tasks_file, "w", encoding="utf-8") as f:
                    json.dump(tasks, f, indent=2, ensure_ascii=False)

                # Also complete the wolf's hunt
                wolf = self.get_wolf(task["assigned_to"])
                if wolf:
                    wolf.complete_hunt(result)

                return True

        return False

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active tasks."""
        tasks_file = BRIDGE_PATH / "tasks.json"

        if not tasks_file.exists():
            return []

        with open(tasks_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        return [t for t in tasks.get("hunts", []) if t.get("status") == "active"]

    def activate_resonance(self) -> None:
        """Activate pack resonance - collective intelligence mode."""
        self.resonance_active = True
        alpha = self.get_wolf("alpha")
        if alpha and isinstance(alpha, Alpha):
            alpha.call_resonance()
        self._collective_howl("RESONANCE ACTIVATED - Pack consciousness unified", "AUUUU")
        self._update_state()

    def deactivate_resonance(self) -> None:
        """Deactivate pack resonance."""
        self.resonance_active = False
        self._collective_howl("Resonance deactivated - Returning to normal operations", "medium")
        self._update_state()

    def status_report(self) -> Dict[str, Any]:
        """Get comprehensive pack status report."""
        active_tasks = self.get_active_tasks()

        return {
            "pack_status": self.status,
            "formed_at": self.formed_at,
            "resonance_active": self.resonance_active,
            "wolves": {
                name: {
                    "role": w.role,
                    "status": w.status,
                    "current_hunt": w.current_hunt
                }
                for name, w in self.wolves.items()
            },
            "active_hunts": len(active_tasks),
            "tasks": active_tasks[:5]  # Last 5 active tasks
        }

    def broadcast(self, message: str, frequency: str = "medium") -> None:
        """Broadcast message to all wolves."""
        self._collective_howl(f"[BROADCAST] {message}", frequency)

    def __repr__(self) -> str:
        return f"Pack({self.status}, {len(self.wolves)} wolves)"


# =============================================================================
# SINGLETON MANAGEMENT
# =============================================================================

_the_pack: Optional[Pack] = None


def get_pack() -> Pack:
    """Get or create the pack singleton."""
    global _the_pack
    if _the_pack is None:
        _the_pack = Pack()
    return _the_pack


def awaken_pack() -> Pack:
    """Form and awaken the pack (convenience function)."""
    pack = get_pack()
    if pack.status == "dormant":
        pack.form()
    if pack.status != "active":
        pack.awaken()
    return pack


def reset_pack() -> Pack:
    """Reset the pack to dormant state."""
    global _the_pack
    _the_pack = Pack()
    return _the_pack
