"""
Pack Analytics - Metrics and performance tracking

Track pack performance, hunt success rates, and system health.

Usage:
    from modules.analytics import get_analytics

    analytics = get_analytics()
    analytics.track_event("hunt_completed", {"target": "bug fix", "wolf": "hunter"})
    summary = analytics.get_summary()
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIDGE_PATH


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    timestamp: str
    tags: Dict[str, str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Event:
    """A tracked event."""
    name: str
    data: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


class Analytics:
    """
    Pack analytics and metrics system.

    Tracks:
    - Howl counts by frequency
    - Hunt success/failure rates
    - Wolf activity
    - API requests
    - Sandbox executions
    - Alpha brain usage
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (BRIDGE_PATH / "analytics")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_file = self.data_dir / "metrics.jsonl"
        self._events_file = self.data_dir / "events.jsonl"

        # In-memory counters for fast access
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()

        self._load_counters()

    def _load_counters(self) -> None:
        """Load counters from disk."""
        counters_file = self.data_dir / "counters.json"
        if counters_file.exists():
            try:
                with open(counters_file) as f:
                    data = json.load(f)
                    self._counters.update(data.get("counters", {}))
                    self._gauges.update(data.get("gauges", {}))
            except Exception:
                pass

    def _save_counters(self) -> None:
        """Save counters to disk."""
        counters_file = self.data_dir / "counters.json"
        with open(counters_file, "w") as f:
            json.dump({
                "counters": dict(self._counters),
                "gauges": self._gauges,
                "updated": datetime.utcnow().isoformat() + "Z"
            }, f, indent=2)

    def increment(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> int:
        """Increment a counter."""
        with self._lock:
            full_name = self._make_name(name, tags)
            self._counters[full_name] += value
            self._save_counters()
            return self._counters[full_name]

    def gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge value."""
        with self._lock:
            full_name = self._make_name(name, tags)
            self._gauges[full_name] = value
            self._save_counters()

    def _make_name(self, name: str, tags: Dict[str, str] = None) -> str:
        """Create metric name with tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a metric data point."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow().isoformat() + "Z",
            tags=tags or {}
        )

        with open(self._metrics_file, "a") as f:
            f.write(json.dumps(metric.to_dict()) + "\n")

    def track_event(self, name: str, data: Dict[str, Any] = None) -> None:
        """Track an event."""
        event = Event(
            name=name,
            data=data or {},
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

        with open(self._events_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

        # Also increment counter
        self.increment(f"events.{name}")

    # =========================================================================
    # WOLF-SPECIFIC TRACKING
    # =========================================================================

    def track_howl(self, frequency: str, from_wolf: str, to: str) -> None:
        """Track a howl."""
        self.increment("howls.total")
        self.increment(f"howls.frequency.{frequency}")
        self.increment(f"howls.from.{from_wolf}")
        self.track_event("howl", {
            "frequency": frequency,
            "from": from_wolf,
            "to": to
        })

    def track_hunt(self, target: str, wolf: str, success: bool, duration: float = 0) -> None:
        """Track a hunt."""
        self.increment("hunts.total")
        if success:
            self.increment("hunts.success")
        else:
            self.increment("hunts.failed")
        self.increment(f"hunts.wolf.{wolf}")

        if duration > 0:
            self.record_metric("hunt.duration", duration, {"wolf": wolf})

        self.track_event("hunt", {
            "target": target[:100],
            "wolf": wolf,
            "success": success,
            "duration": duration
        })

    def track_alpha_thought(self, thought_type: str, model: str, tokens: int = 0) -> None:
        """Track Alpha brain usage."""
        self.increment("alpha.thoughts.total")
        self.increment(f"alpha.thoughts.type.{thought_type}")
        self.increment(f"alpha.model.{model}")

        if tokens > 0:
            self.increment("alpha.tokens.total", tokens)
            self.record_metric("alpha.tokens", tokens, {"type": thought_type})

        self.track_event("alpha_thought", {
            "type": thought_type,
            "model": model,
            "tokens": tokens
        })

    def track_sandbox_execution(self, language: str, success: bool, duration: float) -> None:
        """Track sandbox execution."""
        self.increment("sandbox.total")
        if success:
            self.increment("sandbox.success")
        else:
            self.increment("sandbox.failed")
        self.increment(f"sandbox.language.{language}")

        self.record_metric("sandbox.duration", duration, {"language": language})

        self.track_event("sandbox_execution", {
            "language": language,
            "success": success,
            "duration": duration
        })

    def track_api_request(self, endpoint: str, method: str, status_code: int, duration: float) -> None:
        """Track API request."""
        self.increment("api.requests.total")
        self.increment(f"api.endpoint.{endpoint.replace('/', '_')}")
        self.increment(f"api.status.{status_code}")

        self.record_metric("api.duration", duration, {"endpoint": endpoint})

    # =========================================================================
    # SUMMARIES
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get analytics summary."""
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)

        total_howls = counters.get("howls.total", 0)
        total_hunts = counters.get("hunts.total", 0)
        successful_hunts = counters.get("hunts.success", 0)
        failed_hunts = counters.get("hunts.failed", 0)

        hunt_success_rate = (successful_hunts / total_hunts * 100) if total_hunts > 0 else 0

        total_sandbox = counters.get("sandbox.total", 0)
        sandbox_success = counters.get("sandbox.success", 0)
        sandbox_rate = (sandbox_success / total_sandbox * 100) if total_sandbox > 0 else 0

        alpha_thoughts = counters.get("alpha.thoughts.total", 0)
        alpha_tokens = counters.get("alpha.tokens.total", 0)

        return {
            "howls": {
                "total": total_howls,
                "by_frequency": {
                    "low": counters.get("howls.frequency.low", 0),
                    "medium": counters.get("howls.frequency.medium", 0),
                    "high": counters.get("howls.frequency.high", 0),
                    "AUUUU": counters.get("howls.frequency.AUUUU", 0)
                }
            },
            "hunts": {
                "total": total_hunts,
                "success": successful_hunts,
                "failed": failed_hunts,
                "success_rate": f"{hunt_success_rate:.1f}%"
            },
            "alpha_brain": {
                "total_thoughts": alpha_thoughts,
                "total_tokens": alpha_tokens
            },
            "sandbox": {
                "total": total_sandbox,
                "success": sandbox_success,
                "failed": counters.get("sandbox.failed", 0),
                "success_rate": f"{sandbox_rate:.1f}%"
            },
            "api": {
                "total_requests": counters.get("api.requests.total", 0)
            }
        }

    def get_recent_events(self, limit: int = 20, event_type: str = None) -> List[Dict]:
        """Get recent events."""
        if not self._events_file.exists():
            return []

        events = []
        with open(self._events_file) as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if event_type and event.get("name") != event_type:
                        continue
                    events.append(event)
                except:
                    continue

        return events[-limit:]

    def get_wolf_stats(self, wolf: str) -> Dict[str, Any]:
        """Get stats for a specific wolf."""
        with self._lock:
            hunts = self._counters.get(f"hunts.wolf.{wolf}", 0)
            howls = self._counters.get(f"howls.from.{wolf}", 0)

        return {
            "wolf": wolf,
            "hunts_completed": hunts,
            "howls_sent": howls
        }

    def reset_counters(self) -> None:
        """Reset all counters (use with caution)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._save_counters()


# =============================================================================
# SINGLETON
# =============================================================================

_analytics: Optional[Analytics] = None


def get_analytics() -> Analytics:
    """Get or create analytics singleton."""
    global _analytics
    if _analytics is None:
        _analytics = Analytics()
    return _analytics


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="WOLF_AI Analytics")
    parser.add_argument("action", choices=["summary", "events", "wolf", "reset"])
    parser.add_argument("--wolf", help="Wolf name for wolf stats")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--type", help="Event type filter")
    args = parser.parse_args()

    analytics = get_analytics()

    if args.action == "summary":
        summary = analytics.get_summary()
        print(json.dumps(summary, indent=2))

    elif args.action == "events":
        events = analytics.get_recent_events(args.limit, args.type)
        for event in events:
            print(json.dumps(event))

    elif args.action == "wolf":
        if not args.wolf:
            print("Error: --wolf required")
            return
        stats = analytics.get_wolf_stats(args.wolf)
        print(json.dumps(stats, indent=2))

    elif args.action == "reset":
        confirm = input("Are you sure? This will reset all counters (yes/no): ")
        if confirm.lower() == "yes":
            analytics.reset_counters()
            print("Counters reset.")


if __name__ == "__main__":
    main()
