"""
Hunt Module - Task execution engine for WOLF_AI

Handles:
- Task queuing and prioritization
- Task execution with timeout/retry
- Result tracking and reporting
- Concurrent hunt coordination
"""

import json
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from dataclasses import dataclass, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BRIDGE_PATH, LOGS_PATH


class HuntStatus(Enum):
    """Status of a hunt."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HuntPriority(Enum):
    """Priority levels for hunts."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Hunt:
    """Represents a single hunt (task)."""
    id: str
    target: str
    assigned_to: str
    priority: HuntPriority = HuntPriority.MEDIUM
    status: HuntStatus = HuntStatus.PENDING
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    started: Optional[str] = None
    completed: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    timeout: int = 300  # seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "assigned_to": self.assigned_to,
            "priority": self.priority.value,
            "status": self.status.value,
            "created": self.created,
            "started": self.started,
            "completed": self.completed,
            "result": self.result,
            "error": self.error,
            "retries": self.retries
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Hunt":
        return cls(
            id=data["id"],
            target=data["target"],
            assigned_to=data.get("assigned_to", "hunter"),
            priority=HuntPriority(data.get("priority", 2)),
            status=HuntStatus(data.get("status", "pending")),
            created=data.get("created", datetime.utcnow().isoformat() + "Z"),
            started=data.get("started"),
            completed=data.get("completed"),
            result=data.get("result"),
            error=data.get("error"),
            retries=data.get("retries", 0)
        )


class HuntQueue:
    """Manages the hunt queue with priority ordering."""

    def __init__(self):
        self.hunts: Dict[str, Hunt] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load hunts from tasks.json."""
        tasks_file = BRIDGE_PATH / "tasks.json"
        if tasks_file.exists():
            with open(tasks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for hunt_data in data.get("hunts", []):
                    hunt = Hunt.from_dict(hunt_data)
                    self.hunts[hunt.id] = hunt

    def _save(self):
        """Save hunts to tasks.json."""
        tasks_file = BRIDGE_PATH / "tasks.json"
        data = {
            "version": "1.0",
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "hunts": [h.to_dict() for h in self.hunts.values()]
        }
        with open(tasks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add(self, target: str, assigned_to: str = "hunter",
            priority: HuntPriority = HuntPriority.MEDIUM) -> Hunt:
        """Add a new hunt to the queue."""
        with self._lock:
            hunt_id = f"hunt_{len(self.hunts) + 1:04d}_{int(time.time())}"
            hunt = Hunt(
                id=hunt_id,
                target=target,
                assigned_to=assigned_to,
                priority=priority
            )
            self.hunts[hunt_id] = hunt
            self._save()
            return hunt

    def get(self, hunt_id: str) -> Optional[Hunt]:
        """Get a hunt by ID."""
        return self.hunts.get(hunt_id)

    def get_pending(self) -> List[Hunt]:
        """Get all pending hunts, sorted by priority."""
        pending = [h for h in self.hunts.values() if h.status == HuntStatus.PENDING]
        return sorted(pending, key=lambda h: h.priority.value, reverse=True)

    def get_active(self) -> List[Hunt]:
        """Get all active hunts."""
        return [h for h in self.hunts.values() if h.status == HuntStatus.ACTIVE]

    def start(self, hunt_id: str) -> Optional[Hunt]:
        """Mark a hunt as started."""
        with self._lock:
            hunt = self.hunts.get(hunt_id)
            if hunt and hunt.status == HuntStatus.PENDING:
                hunt.status = HuntStatus.ACTIVE
                hunt.started = datetime.utcnow().isoformat() + "Z"
                self._save()
                return hunt
        return None

    def complete(self, hunt_id: str, result: Optional[str] = None) -> Optional[Hunt]:
        """Mark a hunt as completed."""
        with self._lock:
            hunt = self.hunts.get(hunt_id)
            if hunt:
                hunt.status = HuntStatus.COMPLETED
                hunt.completed = datetime.utcnow().isoformat() + "Z"
                hunt.result = result
                self._save()
                return hunt
        return None

    def fail(self, hunt_id: str, error: str) -> Optional[Hunt]:
        """Mark a hunt as failed."""
        with self._lock:
            hunt = self.hunts.get(hunt_id)
            if hunt:
                hunt.retries += 1
                if hunt.retries >= hunt.max_retries:
                    hunt.status = HuntStatus.FAILED
                    hunt.completed = datetime.utcnow().isoformat() + "Z"
                else:
                    hunt.status = HuntStatus.PENDING  # Retry
                hunt.error = error
                self._save()
                return hunt
        return None

    def cancel(self, hunt_id: str) -> Optional[Hunt]:
        """Cancel a hunt."""
        with self._lock:
            hunt = self.hunts.get(hunt_id)
            if hunt and hunt.status in [HuntStatus.PENDING, HuntStatus.ACTIVE]:
                hunt.status = HuntStatus.CANCELLED
                hunt.completed = datetime.utcnow().isoformat() + "Z"
                self._save()
                return hunt
        return None


class HuntExecutor:
    """Executes hunts with various strategies."""

    def __init__(self):
        self.queue = HuntQueue()
        self._running = False
        self._executor_thread: Optional[threading.Thread] = None
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, hunt_type: str, handler: Callable):
        """Register a handler for a specific hunt type."""
        self.handlers[hunt_type] = handler

    def execute_shell(self, command: str, timeout: int = 300) -> Dict[str, Any]:
        """Execute a shell command hunt."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout expired"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code hunt."""
        try:
            # Create a restricted namespace
            namespace = {"__builtins__": __builtins__}
            exec(code, namespace)
            return {"success": True, "namespace": str(namespace.keys())}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute_hunt(self, hunt: Hunt) -> Dict[str, Any]:
        """Execute a single hunt."""
        target = hunt.target

        # Log start
        self._log_hunt(hunt, "started")

        # Determine hunt type and execute
        if target.startswith("shell:"):
            command = target[6:].strip()
            result = self.execute_shell(command, hunt.timeout)
        elif target.startswith("python:"):
            code = target[7:].strip()
            result = self.execute_python(code)
        elif target.startswith("file:"):
            # File operation
            result = self._execute_file_hunt(target[5:].strip())
        else:
            # Default: treat as description, log it
            result = {"success": True, "note": f"Hunt registered: {target}"}

        # Log completion
        self._log_hunt(hunt, "completed", result)

        return result

    def _execute_file_hunt(self, path: str) -> Dict[str, Any]:
        """Execute a file-related hunt."""
        try:
            p = Path(path)
            if p.exists():
                if p.is_file():
                    content = p.read_text(encoding="utf-8")[:1000]
                    return {"success": True, "type": "file", "preview": content}
                else:
                    files = list(p.iterdir())[:20]
                    return {"success": True, "type": "directory", "contents": [str(f) for f in files]}
            else:
                return {"success": False, "error": f"Path not found: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _log_hunt(self, hunt: Hunt, status: str, result: Optional[Dict] = None):
        """Log hunt activity."""
        log_file = LOGS_PATH / "hunts.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        log_entry = {
            "hunt_id": hunt.id,
            "target": hunt.target,
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "result": result
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def process_next(self) -> Optional[Dict[str, Any]]:
        """Process the next pending hunt."""
        pending = self.queue.get_pending()
        if not pending:
            return None

        hunt = pending[0]
        self.queue.start(hunt.id)

        try:
            result = self.execute_hunt(hunt)
            if result.get("success"):
                self.queue.complete(hunt.id, json.dumps(result))
            else:
                self.queue.fail(hunt.id, result.get("error", "Unknown error"))
            return result
        except Exception as e:
            self.queue.fail(hunt.id, str(e))
            return {"success": False, "error": str(e)}

    def start_daemon(self, interval: float = 1.0):
        """Start background hunt processor."""
        self._running = True

        def daemon_loop():
            while self._running:
                self.process_next()
                time.sleep(interval)

        self._executor_thread = threading.Thread(target=daemon_loop, daemon=True)
        self._executor_thread.start()

    def stop_daemon(self):
        """Stop the background processor."""
        self._running = False
        if self._executor_thread:
            self._executor_thread.join(timeout=5)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_executor: Optional[HuntExecutor] = None


def get_executor() -> HuntExecutor:
    """Get or create the hunt executor."""
    global _executor
    if _executor is None:
        _executor = HuntExecutor()
    return _executor


def hunt(target: str, priority: str = "medium", assigned_to: str = "hunter") -> Hunt:
    """Quick function to add a hunt."""
    executor = get_executor()
    prio = HuntPriority[priority.upper()]
    return executor.queue.add(target, assigned_to, prio)


def hunt_now(target: str) -> Dict[str, Any]:
    """Execute a hunt immediately (blocking)."""
    executor = get_executor()
    h = executor.queue.add(target, "hunter", HuntPriority.CRITICAL)
    executor.queue.start(h.id)
    result = executor.execute_hunt(h)
    if result.get("success"):
        executor.queue.complete(h.id, json.dumps(result))
    else:
        executor.queue.fail(h.id, result.get("error", "Unknown"))
    return result


def get_hunts(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get hunts, optionally filtered by status."""
    executor = get_executor()
    hunts = list(executor.queue.hunts.values())

    if status:
        hunts = [h for h in hunts if h.status.value == status]

    return [h.to_dict() for h in hunts]
