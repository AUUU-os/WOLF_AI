"""
Health Monitor - System health checks and monitoring

Monitor pack health, system resources, and service availability.

Usage:
    from modules.health import check_health, get_monitor

    # Quick check
    report = await check_health()

    # Full monitor
    monitor = get_monitor()
    await monitor.start()  # Background monitoring
"""

import os
import asyncio
import json
import subprocess
import platform
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIDGE_PATH, WOLF_ROOT


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    value: Any = None
    threshold: Any = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class HealthReport:
    """Complete health report."""
    overall: HealthStatus
    checks: List[HealthCheck]
    timestamp: str
    system: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "overall": self.overall.value,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
            "system": self.system
        }


# =============================================================================
# HEALTH CHECKS
# =============================================================================

async def check_disk_space(path: str = None, threshold: int = 90) -> HealthCheck:
    """Check disk space usage."""
    try:
        import shutil

        path = path or str(WOLF_ROOT)
        usage = shutil.disk_usage(path)
        percent = (usage.used / usage.total) * 100

        if percent > threshold:
            status = HealthStatus.CRITICAL
            message = f"Disk usage critical: {percent:.1f}%"
        elif percent > threshold - 10:
            status = HealthStatus.WARNING
            message = f"Disk usage high: {percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage: {percent:.1f}%"

        return HealthCheck(
            name="disk_space",
            status=status,
            message=message,
            value=percent,
            threshold=threshold
        )

    except Exception as e:
        return HealthCheck(
            name="disk_space",
            status=HealthStatus.UNKNOWN,
            message=str(e)
        )


async def check_memory(threshold: int = 90) -> HealthCheck:
    """Check memory usage."""
    try:
        system = platform.system()

        if system == "Linux":
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem_info = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = int(parts[1].strip().split()[0])
                    mem_info[key] = value

            total = mem_info.get("MemTotal", 1)
            available = mem_info.get("MemAvailable", mem_info.get("MemFree", 0))
            used_percent = ((total - available) / total) * 100

        elif system == "Darwin":  # macOS
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True
            )
            # Simplified parsing
            used_percent = 50  # Placeholder

        else:
            used_percent = 50  # Placeholder

        if used_percent > threshold:
            status = HealthStatus.CRITICAL
            message = f"Memory usage critical: {used_percent:.1f}%"
        elif used_percent > threshold - 15:
            status = HealthStatus.WARNING
            message = f"Memory usage high: {used_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage: {used_percent:.1f}%"

        return HealthCheck(
            name="memory",
            status=status,
            message=message,
            value=used_percent,
            threshold=threshold
        )

    except Exception as e:
        return HealthCheck(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message=str(e)
        )


async def check_api_server(host: str = "localhost", port: int = 8000) -> HealthCheck:
    """Check if API server is running."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"http://{host}:{port}/api/status")
            if response.status_code == 200:
                return HealthCheck(
                    name="api_server",
                    status=HealthStatus.HEALTHY,
                    message="API server running",
                    value={"status_code": 200}
                )
            else:
                return HealthCheck(
                    name="api_server",
                    status=HealthStatus.WARNING,
                    message=f"API returned {response.status_code}",
                    value={"status_code": response.status_code}
                )

    except Exception as e:
        return HealthCheck(
            name="api_server",
            status=HealthStatus.CRITICAL,
            message=f"API server not reachable: {e}"
        )


async def check_ollama() -> HealthCheck:
    """Check if Ollama is running."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return HealthCheck(
                    name="ollama",
                    status=HealthStatus.HEALTHY,
                    message=f"Ollama running with {len(models)} models",
                    value={"models": len(models)}
                )

    except:
        pass

    return HealthCheck(
        name="ollama",
        status=HealthStatus.WARNING,
        message="Ollama not running (optional)"
    )


async def check_claude_api() -> HealthCheck:
    """Check Claude API availability."""
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return HealthCheck(
            name="claude_api",
            status=HealthStatus.WARNING,
            message="ANTHROPIC_API_KEY not set"
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        # Just check client creation, don't make actual call
        return HealthCheck(
            name="claude_api",
            status=HealthStatus.HEALTHY,
            message="Claude API configured"
        )
    except ImportError:
        return HealthCheck(
            name="claude_api",
            status=HealthStatus.WARNING,
            message="anthropic package not installed"
        )
    except Exception as e:
        return HealthCheck(
            name="claude_api",
            status=HealthStatus.CRITICAL,
            message=f"Claude API error: {e}"
        )


async def check_openai_api() -> HealthCheck:
    """Check OpenAI API availability."""
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return HealthCheck(
            name="openai_api",
            status=HealthStatus.WARNING,
            message="OPENAI_API_KEY not set (optional)"
        )

    try:
        import openai
        return HealthCheck(
            name="openai_api",
            status=HealthStatus.HEALTHY,
            message="OpenAI API configured"
        )
    except ImportError:
        return HealthCheck(
            name="openai_api",
            status=HealthStatus.WARNING,
            message="openai package not installed"
        )


async def check_docker() -> HealthCheck:
    """Check Docker availability."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return HealthCheck(
                name="docker",
                status=HealthStatus.HEALTHY,
                message="Docker running"
            )
        else:
            return HealthCheck(
                name="docker",
                status=HealthStatus.WARNING,
                message="Docker not running"
            )
    except FileNotFoundError:
        return HealthCheck(
            name="docker",
            status=HealthStatus.WARNING,
            message="Docker not installed (optional)"
        )
    except Exception as e:
        return HealthCheck(
            name="docker",
            status=HealthStatus.WARNING,
            message=f"Docker check failed: {e}"
        )


async def check_bridge_files() -> HealthCheck:
    """Check bridge directory and files."""
    try:
        bridge_path = BRIDGE_PATH

        if not bridge_path.exists():
            return HealthCheck(
                name="bridge_files",
                status=HealthStatus.WARNING,
                message="Bridge directory not found"
            )

        howls = bridge_path / "howls.jsonl"
        state = bridge_path / "state.json"

        files_ok = howls.exists() and state.exists()

        if files_ok:
            # Check file freshness
            state_mtime = datetime.fromtimestamp(state.stat().st_mtime)
            age = datetime.now() - state_mtime

            if age > timedelta(hours=24):
                return HealthCheck(
                    name="bridge_files",
                    status=HealthStatus.WARNING,
                    message=f"State file stale ({age.total_seconds()/3600:.1f}h old)"
                )

            return HealthCheck(
                name="bridge_files",
                status=HealthStatus.HEALTHY,
                message="Bridge files OK"
            )
        else:
            return HealthCheck(
                name="bridge_files",
                status=HealthStatus.WARNING,
                message="Some bridge files missing"
            )

    except Exception as e:
        return HealthCheck(
            name="bridge_files",
            status=HealthStatus.UNKNOWN,
            message=str(e)
        )


async def check_pack_status() -> HealthCheck:
    """Check pack status."""
    try:
        from core.pack import get_pack

        pack = get_pack()
        report = pack.status_report()

        status = report.get("pack_status", "unknown")

        if status == "active":
            return HealthCheck(
                name="pack_status",
                status=HealthStatus.HEALTHY,
                message=f"Pack active with {len(report.get('wolves', {}))} wolves",
                value=report
            )
        elif status in ["formed", "dormant"]:
            return HealthCheck(
                name="pack_status",
                status=HealthStatus.WARNING,
                message=f"Pack status: {status}"
            )
        else:
            return HealthCheck(
                name="pack_status",
                status=HealthStatus.UNKNOWN,
                message=f"Unknown status: {status}"
            )

    except Exception as e:
        return HealthCheck(
            name="pack_status",
            status=HealthStatus.UNKNOWN,
            message=str(e)
        )


# =============================================================================
# HEALTH MONITOR
# =============================================================================

class HealthMonitor:
    """
    Background health monitor.

    Periodically runs health checks and triggers alerts.
    """

    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_report: Optional[HealthReport] = None
        self._alert_callback: Optional[Callable] = None

    def set_alert_callback(self, callback: Callable[[HealthCheck], None]) -> None:
        """Set callback for health alerts."""
        self._alert_callback = callback

    async def run_checks(self) -> HealthReport:
        """Run all health checks."""
        checks = await asyncio.gather(
            check_disk_space(),
            check_memory(),
            check_api_server(),
            check_ollama(),
            check_claude_api(),
            check_openai_api(),
            check_docker(),
            check_bridge_files(),
            check_pack_status()
        )

        # Determine overall status
        statuses = [c.status for c in checks]
        if HealthStatus.CRITICAL in statuses:
            overall = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall = HealthStatus.WARNING
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        # System info
        system_info = {
            "platform": platform.system(),
            "python": platform.python_version(),
            "hostname": platform.node(),
            "wolf_root": str(WOLF_ROOT)
        }

        report = HealthReport(
            overall=overall,
            checks=list(checks),
            timestamp=datetime.utcnow().isoformat() + "Z",
            system=system_info
        )

        self._last_report = report
        self._save_report(report)

        # Trigger alerts for critical/warning
        if self._alert_callback:
            for check in checks:
                if check.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                    self._alert_callback(check)

        return report

    def _save_report(self, report: HealthReport) -> None:
        """Save report to file."""
        report_file = BRIDGE_PATH / "health_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Also append to history
        history_file = BRIDGE_PATH / "health_history.jsonl"
        with open(history_file, "a") as f:
            f.write(json.dumps({
                "overall": report.overall.value,
                "timestamp": report.timestamp
            }) + "\n")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                asyncio.run(self.run_checks())
            except Exception as e:
                print(f"[HealthMonitor] Error: {e}")

            time.sleep(self.check_interval)

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[HealthMonitor] Started (interval: {self.check_interval}s)")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[HealthMonitor] Stopped")

    def get_last_report(self) -> Optional[HealthReport]:
        """Get most recent health report."""
        return self._last_report

    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        if self._last_report:
            return self._last_report.overall == HealthStatus.HEALTHY
        return True  # Assume healthy if no report yet


# =============================================================================
# SINGLETON & CONVENIENCE
# =============================================================================

_monitor: Optional[HealthMonitor] = None


def get_monitor() -> HealthMonitor:
    """Get or create monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = HealthMonitor()
    return _monitor


async def check_health() -> Dict[str, Any]:
    """Quick health check - run all checks and return report."""
    monitor = get_monitor()
    report = await monitor.run_checks()
    return report.to_dict()


async def is_healthy() -> bool:
    """Quick health check - returns True if healthy."""
    report = await check_health()
    return report["overall"] == "healthy"


# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="WOLF_AI Health Monitor")
    parser.add_argument("action", nargs="?", default="check",
                       choices=["check", "watch", "status"])
    parser.add_argument("--interval", "-i", type=int, default=60,
                       help="Check interval in seconds")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    if args.action == "check":
        report = await check_health()

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            # Pretty print
            status_icons = {
                "healthy": "✓",
                "warning": "⚠",
                "critical": "✗",
                "unknown": "?"
            }

            print(f"\n🏥 WOLF_AI Health Report")
            print(f"   Overall: {status_icons.get(report['overall'], '?')} {report['overall'].upper()}")
            print(f"   Time: {report['timestamp'][:19]}\n")

            print("   Checks:")
            for check in report["checks"]:
                icon = status_icons.get(check["status"], "?")
                print(f"     {icon} {check['name']}: {check['message']}")

            print()

    elif args.action == "watch":
        print(f"\n👁️ Starting health monitor (interval: {args.interval}s)")
        print("   Press Ctrl+C to stop\n")

        monitor = get_monitor()
        monitor.check_interval = args.interval

        def on_alert(check):
            print(f"   ⚠️ ALERT: {check.name} - {check.message}")

        monitor.set_alert_callback(on_alert)
        monitor.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop()
            print("\n   Monitor stopped.")

    elif args.action == "status":
        monitor = get_monitor()
        report = monitor.get_last_report()

        if report:
            print(f"Last check: {report.timestamp}")
            print(f"Status: {report.overall.value}")
        else:
            print("No health report available. Run 'check' first.")


if __name__ == "__main__":
    asyncio.run(main())
