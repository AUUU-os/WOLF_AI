#!/usr/bin/env python3
"""
Wolf CLI - Unified command interface for WOLF_AI

The one command to rule the pack.

Usage:
    wolf status              - Pack status
    wolf awaken              - Wake the pack
    wolf howl "message"      - Send a howl
    wolf hunt "task"         - Start a hunt
    wolf think "question"    - Ask Alpha to think
    wolf plan "objective"    - Create a plan
    wolf exec "code"         - Execute in sandbox
    wolf voice               - Start voice control
    wolf server              - Start API server
    wolf sync                - Git sync
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure WOLF_AI is in path
WOLF_ROOT = Path(__file__).parent
sys.path.insert(0, str(WOLF_ROOT))

from config import BRIDGE_PATH

# Colors for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def c(text: str, color: str) -> str:
    """Colorize text."""
    return f"{color}{text}{Colors.END}"

def banner():
    """Print wolf banner."""
    print(c("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🐺 WOLF_AI CLI v0.4.0                                   ║
    ║   ═══════════════════════════════════════                 ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """, Colors.CYAN))


# =============================================================================
# COMMANDS
# =============================================================================

def cmd_status(args):
    """Show pack status."""
    from core.pack import get_pack

    pack = get_pack()
    report = pack.status_report()

    print(c("\n🐺 Pack Status\n", Colors.BOLD))
    print(f"  Status: {c(report['pack_status'], Colors.GREEN if report['pack_status'] == 'active' else Colors.YELLOW)}")
    print(f"  Resonance: {'🔥 Active' if report.get('resonance_active') else '💤 Inactive'}")
    print(f"  Active Hunts: {report.get('active_hunts', 0)}")

    if report.get('wolves'):
        print(c("\n  Wolves:", Colors.BOLD))
        for name, info in report['wolves'].items():
            status_color = Colors.GREEN if info['status'] == 'hunting' else Colors.YELLOW
            print(f"    {name}: {c(info['status'], status_color)} ({info['role']})")

    print()


def cmd_awaken(args):
    """Awaken the pack."""
    from core.pack import awaken_pack

    print(c("\n🐺 Awakening the pack...\n", Colors.YELLOW))
    pack = awaken_pack()
    print(c("✓ Pack is awake and ready!", Colors.GREEN))
    print(c("\nAUUUUUUUUUUUUUUUUUU!\n", Colors.CYAN))


def cmd_howl(args):
    """Send a howl to the pack."""
    from modules.howl import get_bridge

    bridge = get_bridge()
    howl = bridge.howl(
        message=args.message,
        to=args.to,
        frequency=args.frequency
    )

    print(c(f"\n🐺 Howl sent!", Colors.GREEN))
    print(f"  To: {howl.to}")
    print(f"  Frequency: {howl.frequency}")
    print(f"  Message: {args.message[:50]}...")
    print()


def cmd_hunt(args):
    """Start a hunt (task)."""
    from core.pack import get_pack

    pack = get_pack()
    success = pack.hunt(args.target, args.wolf)

    if success:
        print(c(f"\n🎯 Hunt started!", Colors.GREEN))
        print(f"  Target: {args.target}")
        print(f"  Assigned to: {args.wolf}")
    else:
        print(c(f"\n❌ Failed to start hunt", Colors.RED))
    print()


def cmd_think(args):
    """Ask Alpha to think."""
    async def _think():
        from core.alpha import get_alpha_brain

        brain = get_alpha_brain()

        if not brain.is_available:
            print(c("\n❌ Alpha brain offline. Set ANTHROPIC_API_KEY.", Colors.RED))
            return

        print(c(f"\n🧠 Alpha is thinking...\n", Colors.YELLOW))

        response = await brain.think(args.question)
        print(c("Alpha:", Colors.BOLD))
        print(response)
        print()

    asyncio.run(_think())


def cmd_plan(args):
    """Create a strategic plan."""
    async def _plan():
        from core.alpha import get_alpha_brain

        brain = get_alpha_brain()

        if not brain.is_available:
            print(c("\n❌ Alpha brain offline. Set ANTHROPIC_API_KEY.", Colors.RED))
            return

        print(c(f"\n📋 Alpha is planning...\n", Colors.YELLOW))

        plan = await brain.plan(args.objective)

        print(c("Strategic Plan:", Colors.BOLD))
        print(json.dumps(plan, indent=2))
        print()

    asyncio.run(_plan())


def cmd_exec(args):
    """Execute code in sandbox."""
    async def _exec():
        from modules.sandbox import execute_code

        print(c(f"\n⚡ Executing {args.lang} code...\n", Colors.YELLOW))

        result = await execute_code(args.code, args.lang)

        if result.success:
            print(c("✓ Success", Colors.GREEN))
        else:
            print(c("✗ Failed", Colors.RED))

        print(f"\nExit code: {result.exit_code}")
        print(f"Time: {result.execution_time:.2f}s")

        if result.stdout:
            print(c("\n--- STDOUT ---", Colors.CYAN))
            print(result.stdout)

        if result.stderr:
            print(c("\n--- STDERR ---", Colors.RED))
            print(result.stderr)
        print()

    asyncio.run(_exec())


def cmd_voice(args):
    """Start voice control."""
    print(c("\n🎤 Starting voice control...\n", Colors.CYAN))
    print("Say 'Hey Wolf' to activate")
    print("Commands: status, hunt <task>, howl <message>, think <question>")
    print("Press Ctrl+C to exit\n")

    os.system(f"python {WOLF_ROOT}/voice/voice_control.py")


def cmd_server(args):
    """Start API server."""
    print(c("\n🚀 Starting WOLF_AI Command Center...\n", Colors.CYAN))

    host = args.host or "0.0.0.0"
    port = args.port or 8000

    os.system(f"python -m uvicorn api.server:app --host {host} --port {port} --reload")


def cmd_sync(args):
    """Sync with GitHub."""
    import subprocess

    print(c("\n📡 Syncing with GitHub...\n", Colors.YELLOW))

    result = subprocess.run(
        ["git", "pull", "origin", "main"],
        cwd=str(WOLF_ROOT),
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print(c("✓ Sync complete", Colors.GREEN))
        print(result.stdout)
    else:
        print(c("✗ Sync failed", Colors.RED))
        print(result.stderr)
    print()


def cmd_logs(args):
    """Show recent howls/logs."""
    howls_file = BRIDGE_PATH / "howls.jsonl"

    if not howls_file.exists():
        print(c("\nNo logs found.\n", Colors.YELLOW))
        return

    with open(howls_file, "r") as f:
        lines = f.readlines()

    print(c(f"\n📜 Last {args.limit} howls:\n", Colors.BOLD))

    for line in lines[-args.limit:]:
        try:
            howl = json.loads(line.strip())
            ts = howl.get('timestamp', '')[:19]
            frm = howl.get('from', '?')
            to = howl.get('to', '?')
            msg = howl.get('howl', '')[:60]
            freq = howl.get('frequency', 'medium')

            freq_color = {
                'low': Colors.BLUE,
                'medium': Colors.YELLOW,
                'high': Colors.RED,
                'AUUUU': Colors.CYAN
            }.get(freq, Colors.END)

            print(f"  [{ts}] {c(frm, Colors.GREEN)} → {to}: {c(msg, freq_color)}")
        except:
            continue
    print()


def cmd_metrics(args):
    """Show pack metrics."""
    from modules.analytics import get_analytics

    analytics = get_analytics()
    metrics = analytics.get_summary()

    print(c("\n📊 Pack Metrics\n", Colors.BOLD))

    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print()


def cmd_schedule(args):
    """Manage scheduled tasks."""
    from modules.scheduler import get_scheduler

    scheduler = get_scheduler()

    if args.action == "list":
        jobs = scheduler.list_jobs()
        print(c("\n📅 Scheduled Jobs\n", Colors.BOLD))
        for job in jobs:
            print(f"  [{job['id']}] {job['name']} - {job['schedule']}")
        print()

    elif args.action == "add":
        job_id = scheduler.add_job(
            name=args.name,
            schedule=args.cron,
            command=args.command
        )
        print(c(f"\n✓ Job added: {job_id}\n", Colors.GREEN))

    elif args.action == "remove":
        scheduler.remove_job(args.job_id)
        print(c(f"\n✓ Job removed: {args.job_id}\n", Colors.GREEN))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="🐺 WOLF_AI CLI - Control your pack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  wolf status              Show pack status
  wolf awaken              Wake the pack
  wolf howl "Hello pack!"  Send a howl
  wolf hunt "Fix bug"      Start a hunt
  wolf think "What next?"  Ask Alpha
  wolf exec "print(1+1)"   Run code
  wolf server              Start API
  wolf logs                Show recent logs
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status
    subparsers.add_parser("status", help="Show pack status")

    # Awaken
    subparsers.add_parser("awaken", help="Awaken the pack")

    # Howl
    howl_parser = subparsers.add_parser("howl", help="Send a howl")
    howl_parser.add_argument("message", help="Message to howl")
    howl_parser.add_argument("--to", default="pack", help="Recipient")
    howl_parser.add_argument("--frequency", "-f", default="medium",
                            choices=["low", "medium", "high", "AUUUU"])

    # Hunt
    hunt_parser = subparsers.add_parser("hunt", help="Start a hunt")
    hunt_parser.add_argument("target", help="Hunt target/task")
    hunt_parser.add_argument("--wolf", "-w", default="hunter", help="Assign to wolf")

    # Think
    think_parser = subparsers.add_parser("think", help="Ask Alpha to think")
    think_parser.add_argument("question", help="Question for Alpha")

    # Plan
    plan_parser = subparsers.add_parser("plan", help="Create strategic plan")
    plan_parser.add_argument("objective", help="What to plan")

    # Exec
    exec_parser = subparsers.add_parser("exec", help="Execute code in sandbox")
    exec_parser.add_argument("code", help="Code to execute")
    exec_parser.add_argument("--lang", "-l", default="python", help="Language")

    # Voice
    subparsers.add_parser("voice", help="Start voice control")

    # Server
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", "-p", type=int, default=8000)

    # Sync
    subparsers.add_parser("sync", help="Sync with GitHub")

    # Logs
    logs_parser = subparsers.add_parser("logs", help="Show recent logs")
    logs_parser.add_argument("--limit", "-n", type=int, default=10)

    # Metrics
    subparsers.add_parser("metrics", help="Show pack metrics")

    # Schedule
    schedule_parser = subparsers.add_parser("schedule", help="Manage scheduled tasks")
    schedule_parser.add_argument("action", choices=["list", "add", "remove"])
    schedule_parser.add_argument("--name", help="Job name")
    schedule_parser.add_argument("--cron", help="Cron schedule")
    schedule_parser.add_argument("--command", help="Command to run")
    schedule_parser.add_argument("--job-id", help="Job ID to remove")

    args = parser.parse_args()

    if not args.command:
        banner()
        parser.print_help()
        return

    # Dispatch commands
    commands = {
        "status": cmd_status,
        "awaken": cmd_awaken,
        "howl": cmd_howl,
        "hunt": cmd_hunt,
        "think": cmd_think,
        "plan": cmd_plan,
        "exec": cmd_exec,
        "voice": cmd_voice,
        "server": cmd_server,
        "sync": cmd_sync,
        "logs": cmd_logs,
        "metrics": cmd_metrics,
        "schedule": cmd_schedule,
    }

    if args.command in commands:
        try:
            commands[args.command](args)
        except KeyboardInterrupt:
            print(c("\n\n🐺 Wolf rests. AUUUUUUUU!\n", Colors.CYAN))
        except Exception as e:
            print(c(f"\n❌ Error: {e}\n", Colors.RED))
            if os.getenv("WOLF_DEBUG"):
                raise
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
