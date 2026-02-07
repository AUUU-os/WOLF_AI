"""
Pack Scheduler - Automated hunt scheduling

Schedule recurring tasks for the pack to execute.

Usage:
    from modules.scheduler import get_scheduler

    scheduler = get_scheduler()
    scheduler.add_job(
        name="Daily backup",
        schedule="0 2 * * *",  # 2 AM daily
        command="wolf hunt 'backup database'"
    )
    scheduler.start()
"""

import json
import asyncio
import threading
import time
import uuid
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIDGE_PATH


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class ScheduledJob:
    id: str
    name: str
    schedule: str  # Cron-like: "*/5 * * * *" or simple: "5m", "1h", "1d"
    command: str
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    fail_count: int = 0
    created: str = None

    def __post_init__(self):
        if not self.created:
            self.created = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CronParser:
    """Simple cron-like schedule parser."""

    @staticmethod
    def parse_interval(schedule: str) -> Optional[int]:
        """
        Parse simple interval schedules.

        Examples: "5m" (5 minutes), "1h" (1 hour), "1d" (1 day)
        Returns seconds.
        """
        schedule = schedule.strip().lower()

        multipliers = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400,
            'w': 604800
        }

        for suffix, mult in multipliers.items():
            if schedule.endswith(suffix):
                try:
                    value = int(schedule[:-1])
                    return value * mult
                except ValueError:
                    return None

        return None

    @staticmethod
    def next_run_time(schedule: str, last_run: Optional[datetime] = None) -> datetime:
        """Calculate next run time based on schedule."""
        now = datetime.utcnow()

        # Try simple interval first
        interval = CronParser.parse_interval(schedule)
        if interval:
            if last_run:
                next_time = last_run + timedelta(seconds=interval)
                if next_time < now:
                    # Missed run, schedule for now + interval
                    return now + timedelta(seconds=interval)
                return next_time
            return now + timedelta(seconds=interval)

        # Parse cron-like format: "minute hour day month weekday"
        # Simplified: only supports basic patterns
        parts = schedule.split()
        if len(parts) == 5:
            minute, hour, day, month, weekday = parts

            next_time = now.replace(second=0, microsecond=0)

            # Handle minute
            if minute != '*':
                if minute.startswith('*/'):
                    interval_min = int(minute[2:])
                    next_min = ((now.minute // interval_min) + 1) * interval_min
                    if next_min >= 60:
                        next_time = next_time.replace(minute=0) + timedelta(hours=1)
                    else:
                        next_time = next_time.replace(minute=next_min)
                else:
                    next_time = next_time.replace(minute=int(minute))
                    if next_time <= now:
                        next_time += timedelta(hours=1)

            # Handle hour
            if hour != '*' and not hour.startswith('*/'):
                next_time = next_time.replace(hour=int(hour))
                if next_time <= now:
                    next_time += timedelta(days=1)

            return next_time

        # Default: run in 1 hour
        return now + timedelta(hours=1)


class Scheduler:
    """
    Pack scheduler for automated tasks.

    Runs in background thread, executes jobs on schedule.
    """

    def __init__(self, jobs_file: Optional[Path] = None):
        self.jobs_file = jobs_file or (BRIDGE_PATH / "scheduled_jobs.json")
        self.jobs: Dict[str, ScheduledJob] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._load_jobs()

    def _load_jobs(self) -> None:
        """Load jobs from file."""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, "r") as f:
                    data = json.load(f)
                    for job_data in data.get("jobs", []):
                        job = ScheduledJob(**job_data)
                        self.jobs[job.id] = job
            except Exception as e:
                print(f"[Scheduler] Failed to load jobs: {e}")

    def _save_jobs(self) -> None:
        """Save jobs to file."""
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.jobs_file, "w") as f:
            json.dump({
                "version": "1.0",
                "updated": datetime.utcnow().isoformat() + "Z",
                "jobs": [job.to_dict() for job in self.jobs.values()]
            }, f, indent=2)

    def add_job(
        self,
        name: str,
        schedule: str,
        command: str,
        enabled: bool = True
    ) -> str:
        """
        Add a scheduled job.

        Args:
            name: Job name
            schedule: Cron-like or interval (e.g., "5m", "0 * * * *")
            command: Wolf CLI command or shell command
            enabled: Start enabled

        Returns:
            Job ID
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"

        next_run = CronParser.next_run_time(schedule)

        job = ScheduledJob(
            id=job_id,
            name=name,
            schedule=schedule,
            command=command,
            enabled=enabled,
            next_run=next_run.isoformat() + "Z"
        )

        with self._lock:
            self.jobs[job_id] = job
            self._save_jobs()

        self._log_event("job_added", job_id, name)
        return job_id

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job."""
        with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                self._save_jobs()
                self._log_event("job_removed", job_id)
                return True
        return False

    def enable_job(self, job_id: str) -> bool:
        """Enable a job."""
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].enabled = True
                self._save_jobs()
                return True
        return False

    def disable_job(self, job_id: str) -> bool:
        """Disable a job."""
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].enabled = False
                self._save_jobs()
                return True
        return False

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs."""
        return [job.to_dict() for job in self.jobs.values()]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific job."""
        job = self.jobs.get(job_id)
        return job.to_dict() if job else None

    def _execute_job(self, job: ScheduledJob) -> bool:
        """Execute a job."""
        self._log_event("job_started", job.id, job.name)

        try:
            # Check if it's a wolf command
            if job.command.startswith("wolf "):
                cmd = f"python wolf.py {job.command[5:]}"
            else:
                cmd = job.command

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(Path(__file__).parent.parent)
            )

            success = result.returncode == 0

            with self._lock:
                job.last_run = datetime.utcnow().isoformat() + "Z"
                job.run_count += 1
                if not success:
                    job.fail_count += 1

                # Calculate next run
                job.next_run = CronParser.next_run_time(
                    job.schedule,
                    datetime.utcnow()
                ).isoformat() + "Z"

                self._save_jobs()

            self._log_event(
                "job_completed" if success else "job_failed",
                job.id,
                job.name,
                result.stdout[:200] if success else result.stderr[:200]
            )

            return success

        except subprocess.TimeoutExpired:
            self._log_event("job_timeout", job.id, job.name)
            job.fail_count += 1
            return False

        except Exception as e:
            self._log_event("job_error", job.id, job.name, str(e))
            job.fail_count += 1
            return False

    def _log_event(self, event_type: str, job_id: str, job_name: str = "", details: str = "") -> None:
        """Log scheduler event."""
        log_entry = {
            "type": event_type,
            "job_id": job_id,
            "job_name": job_name,
            "details": details[:200] if details else "",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        log_file = BRIDGE_PATH / "scheduler_log.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            now = datetime.utcnow()

            jobs_to_run = []

            with self._lock:
                for job in self.jobs.values():
                    if not job.enabled:
                        continue

                    if job.next_run:
                        next_run = datetime.fromisoformat(job.next_run.rstrip("Z"))
                        if next_run <= now:
                            jobs_to_run.append(job)

            # Execute due jobs
            for job in jobs_to_run:
                self._execute_job(job)

            # Sleep before next check
            time.sleep(10)  # Check every 10 seconds

    def start(self) -> None:
        """Start the scheduler in background."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._log_event("scheduler_started", "system")
        print("[Scheduler] Started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._log_event("scheduler_stopped", "system")
        print("[Scheduler] Stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# =============================================================================
# SINGLETON
# =============================================================================

_scheduler: Optional[Scheduler] = None


def get_scheduler() -> Scheduler:
    """Get or create scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler()
    return _scheduler


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="WOLF_AI Pack Scheduler")
    parser.add_argument("action", choices=["start", "list", "add", "remove", "run-once"])
    parser.add_argument("--name", help="Job name")
    parser.add_argument("--schedule", help="Schedule (e.g., '5m', '0 * * * *')")
    parser.add_argument("--command", help="Command to run")
    parser.add_argument("--job-id", help="Job ID")
    args = parser.parse_args()

    scheduler = get_scheduler()

    if args.action == "start":
        print("Starting scheduler... Press Ctrl+C to stop.")
        scheduler.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            scheduler.stop()

    elif args.action == "list":
        jobs = scheduler.list_jobs()
        print(f"\nScheduled Jobs ({len(jobs)}):\n")
        for job in jobs:
            status = "✓" if job["enabled"] else "✗"
            print(f"  [{status}] {job['id']}: {job['name']}")
            print(f"      Schedule: {job['schedule']}")
            print(f"      Command: {job['command']}")
            print(f"      Next run: {job['next_run']}")
            print()

    elif args.action == "add":
        if not all([args.name, args.schedule, args.command]):
            print("Error: --name, --schedule, and --command required")
            return
        job_id = scheduler.add_job(args.name, args.schedule, args.command)
        print(f"Job added: {job_id}")

    elif args.action == "remove":
        if not args.job_id:
            print("Error: --job-id required")
            return
        if scheduler.remove_job(args.job_id):
            print(f"Job removed: {args.job_id}")
        else:
            print(f"Job not found: {args.job_id}")

    elif args.action == "run-once":
        if not args.job_id:
            print("Error: --job-id required")
            return
        job = scheduler.jobs.get(args.job_id)
        if job:
            scheduler._execute_job(job)
        else:
            print(f"Job not found: {args.job_id}")


if __name__ == "__main__":
    main()
