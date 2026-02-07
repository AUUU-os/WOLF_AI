"""
Pack Protocols - Predefined workflows and automation

Standard operating procedures for the wolf pack.

Usage:
    from modules.protocols import run_protocol, PROTOCOLS

    # Run a protocol
    result = await run_protocol("code_review", {"file": "main.py"})

    # List available
    for name, proto in PROTOCOLS.items():
        print(f"{name}: {proto.description}")
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIDGE_PATH, WOLF_ROOT


class ProtocolStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProtocolStep:
    """A single step in a protocol."""
    name: str
    action: str  # wolf action: think, hunt, howl, execute, etc.
    params: Dict[str, Any] = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)  # Previous step names
    on_failure: str = "stop"  # stop, continue, retry


@dataclass
class Protocol:
    """A complete protocol definition."""
    name: str
    description: str
    steps: List[ProtocolStep]
    required_params: List[str] = field(default_factory=list)
    timeout: int = 300  # 5 minutes default
    notify_on_complete: bool = True


# =============================================================================
# PROTOCOL DEFINITIONS
# =============================================================================

PROTOCOLS: Dict[str, Protocol] = {
    "morning_standup": Protocol(
        name="morning_standup",
        description="Daily standup - check status, review tasks, plan day",
        required_params=[],
        steps=[
            ProtocolStep(
                name="check_status",
                action="status",
                params={}
            ),
            ProtocolStep(
                name="review_tasks",
                action="think",
                params={
                    "question": "Review active hunts and summarize progress. What needs attention today?"
                },
                requires=["check_status"]
            ),
            ProtocolStep(
                name="plan_day",
                action="plan",
                params={
                    "objective": "Plan today's priorities based on active tasks"
                },
                requires=["review_tasks"]
            ),
            ProtocolStep(
                name="announce",
                action="howl",
                params={
                    "message": "Morning standup complete. Pack is ready!",
                    "frequency": "medium"
                },
                requires=["plan_day"]
            )
        ]
    ),

    "code_review": Protocol(
        name="code_review",
        description="Review code for quality, bugs, and improvements",
        required_params=["file"],
        steps=[
            ProtocolStep(
                name="read_code",
                action="read_file",
                params={"file": "{file}"}
            ),
            ProtocolStep(
                name="analyze",
                action="think",
                params={
                    "question": "Analyze this code for:\n1. Bugs and issues\n2. Code quality\n3. Security concerns\n4. Performance\n5. Suggested improvements\n\nCode:\n{code}"
                },
                requires=["read_code"]
            ),
            ProtocolStep(
                name="report",
                action="howl",
                params={
                    "message": "Code review complete for {file}",
                    "frequency": "medium"
                },
                requires=["analyze"]
            )
        ]
    ),

    "security_scan": Protocol(
        name="security_scan",
        description="Scan codebase for security vulnerabilities",
        required_params=[],
        steps=[
            ProtocolStep(
                name="find_sensitive",
                action="search",
                params={
                    "pattern": "(password|secret|api_key|token|credential)",
                    "type": "content"
                }
            ),
            ProtocolStep(
                name="find_env_files",
                action="search",
                params={
                    "pattern": ".env*",
                    "type": "files"
                }
            ),
            ProtocolStep(
                name="analyze",
                action="think",
                params={
                    "question": "Analyze these security findings and provide recommendations:\n\nSensitive patterns found: {find_sensitive}\n\nEnv files: {find_env_files}"
                },
                requires=["find_sensitive", "find_env_files"]
            ),
            ProtocolStep(
                name="alert",
                action="howl",
                params={
                    "message": "Security scan complete",
                    "frequency": "high"
                },
                requires=["analyze"]
            )
        ]
    ),

    "deploy_check": Protocol(
        name="deploy_check",
        description="Pre-deployment checklist and verification",
        required_params=[],
        steps=[
            ProtocolStep(
                name="git_status",
                action="execute",
                params={
                    "code": "git status --porcelain",
                    "language": "bash"
                }
            ),
            ProtocolStep(
                name="run_tests",
                action="execute",
                params={
                    "code": "python -m pytest --tb=short 2>&1 || echo 'No tests found'",
                    "language": "bash"
                },
                on_failure="continue"
            ),
            ProtocolStep(
                name="check_deps",
                action="execute",
                params={
                    "code": "pip check 2>&1 || echo 'Dependency check failed'",
                    "language": "bash"
                },
                on_failure="continue"
            ),
            ProtocolStep(
                name="analyze",
                action="think",
                params={
                    "question": "Based on these pre-deploy checks, is it safe to deploy?\n\nGit status: {git_status}\nTests: {run_tests}\nDeps: {check_deps}"
                },
                requires=["git_status", "run_tests", "check_deps"]
            ),
            ProtocolStep(
                name="report",
                action="howl",
                params={
                    "message": "Deploy check complete: {analyze_summary}",
                    "frequency": "high"
                },
                requires=["analyze"]
            )
        ]
    ),

    "research": Protocol(
        name="research",
        description="Research a topic and compile findings",
        required_params=["topic"],
        steps=[
            ProtocolStep(
                name="initial_analysis",
                action="think",
                params={
                    "question": "What are the key aspects to research about: {topic}? List 5 specific questions to investigate."
                }
            ),
            ProtocolStep(
                name="deep_dive",
                action="think",
                params={
                    "question": "Provide detailed analysis on: {topic}\n\nAddress these questions:\n{initial_analysis}"
                },
                requires=["initial_analysis"]
            ),
            ProtocolStep(
                name="summarize",
                action="think",
                params={
                    "question": "Create an executive summary of the research on {topic}. Include key findings, recommendations, and next steps."
                },
                requires=["deep_dive"]
            ),
            ProtocolStep(
                name="save_report",
                action="memory_store",
                params={
                    "key": "research_{topic}",
                    "value": "{summarize}",
                    "category": "research"
                },
                requires=["summarize"]
            )
        ]
    ),

    "bug_hunt": Protocol(
        name="bug_hunt",
        description="Investigate and fix a bug",
        required_params=["description"],
        steps=[
            ProtocolStep(
                name="analyze_bug",
                action="think",
                params={
                    "question": "Analyze this bug report and identify likely causes:\n\n{description}\n\nWhat files should we examine? What's the likely root cause?"
                }
            ),
            ProtocolStep(
                name="search_related",
                action="search",
                params={
                    "pattern": "{search_term}",
                    "type": "content"
                },
                requires=["analyze_bug"]
            ),
            ProtocolStep(
                name="propose_fix",
                action="think",
                params={
                    "question": "Based on the analysis and search results, propose a fix:\n\nBug: {description}\nAnalysis: {analyze_bug}\nRelated code: {search_related}"
                },
                requires=["search_related"]
            ),
            ProtocolStep(
                name="create_hunt",
                action="hunt",
                params={
                    "target": "Fix bug: {description}",
                    "assigned_to": "hunter"
                },
                requires=["propose_fix"]
            )
        ]
    ),

    "health_check": Protocol(
        name="health_check",
        description="Check system health and report issues",
        required_params=[],
        steps=[
            ProtocolStep(
                name="check_api",
                action="execute",
                params={
                    "code": "curl -s http://localhost:8000/api/status || echo 'API not running'",
                    "language": "bash"
                },
                on_failure="continue"
            ),
            ProtocolStep(
                name="check_disk",
                action="execute",
                params={
                    "code": "df -h . | tail -1",
                    "language": "bash"
                },
                on_failure="continue"
            ),
            ProtocolStep(
                name="check_memory",
                action="execute",
                params={
                    "code": "free -h 2>/dev/null || vm_stat 2>/dev/null || echo 'Memory check unavailable'",
                    "language": "bash"
                },
                on_failure="continue"
            ),
            ProtocolStep(
                name="analyze_health",
                action="think",
                params={
                    "question": "Analyze system health:\n\nAPI: {check_api}\nDisk: {check_disk}\nMemory: {check_memory}\n\nAre there any concerns?"
                },
                requires=["check_api", "check_disk", "check_memory"]
            ),
            ProtocolStep(
                name="report",
                action="howl",
                params={
                    "message": "Health check: {health_summary}",
                    "frequency": "low"
                },
                requires=["analyze_health"]
            )
        ]
    ),

    "night_watch": Protocol(
        name="night_watch",
        description="Shadow's nightly monitoring routine",
        required_params=[],
        steps=[
            ProtocolStep(
                name="check_logs",
                action="execute",
                params={
                    "code": "tail -50 bridge/howls.jsonl 2>/dev/null | grep -i error || echo 'No errors'",
                    "language": "bash"
                },
                on_failure="continue"
            ),
            ProtocolStep(
                name="backup_state",
                action="execute",
                params={
                    "code": "cp bridge/state.json bridge/state.backup.json 2>/dev/null || echo 'No state to backup'",
                    "language": "bash"
                },
                on_failure="continue"
            ),
            ProtocolStep(
                name="summarize",
                action="think",
                params={
                    "question": "Night watch summary - review any errors and notable events:\n\n{check_logs}"
                },
                requires=["check_logs"]
            ),
            ProtocolStep(
                name="whisper",
                action="howl",
                params={
                    "message": "Night watch complete. All quiet.",
                    "to": "oracle",
                    "frequency": "low"
                },
                requires=["summarize", "backup_state"]
            )
        ]
    )
}


# =============================================================================
# PROTOCOL EXECUTOR
# =============================================================================

class ProtocolExecutor:
    """Execute protocols with step tracking."""

    def __init__(self):
        self.current_run: Optional[Dict] = None
        self.results: Dict[str, Any] = {}

    async def _execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute a single action."""
        if action == "status":
            from core.pack import get_pack
            pack = get_pack()
            return pack.status_report()

        elif action == "think":
            from core.alpha import get_alpha_brain
            brain = get_alpha_brain()
            if brain.is_available:
                return await brain.think(params.get("question", ""))
            return "[Alpha brain offline]"

        elif action == "plan":
            from core.alpha import get_alpha_brain
            brain = get_alpha_brain()
            if brain.is_available:
                return await brain.plan(params.get("objective", ""))
            return {"error": "Alpha brain offline"}

        elif action == "howl":
            from modules.howl import get_bridge
            bridge = get_bridge()
            return bridge.howl(
                message=params.get("message", ""),
                to=params.get("to", "pack"),
                frequency=params.get("frequency", "medium")
            )

        elif action == "hunt":
            from core.pack import get_pack
            pack = get_pack()
            return pack.hunt(
                params.get("target", ""),
                params.get("assigned_to", "hunter")
            )

        elif action == "execute":
            from modules.sandbox import execute_code
            result = await execute_code(
                params.get("code", ""),
                params.get("language", "bash")
            )
            return result.stdout if result.success else result.stderr

        elif action == "search":
            from modules.track import get_tracker
            tracker = get_tracker()
            search_type = params.get("type", "files")
            pattern = params.get("pattern", "*")

            if search_type == "files":
                results = tracker.find(pattern)
            else:
                results = tracker.grep(pattern)

            return [r.to_dict() for r in results[:10]]

        elif action == "read_file":
            file_path = Path(params.get("file", ""))
            if not file_path.is_absolute():
                file_path = WOLF_ROOT / file_path

            if file_path.exists():
                return file_path.read_text()[:5000]
            return f"File not found: {file_path}"

        elif action == "memory_store":
            from memory.store import get_memory
            memory = get_memory()
            memory.set(
                params.get("key", ""),
                params.get("value", ""),
                namespace=params.get("category", "general")
            )
            return {"stored": params.get("key")}

        else:
            return {"error": f"Unknown action: {action}"}

    def _interpolate(self, template: str, context: Dict[str, Any]) -> str:
        """Interpolate variables in template."""
        result = template
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)[:1000]
            elif not isinstance(value, str):
                value = str(value)
            result = result.replace(f"{{{key}}}", value[:2000])
        return result

    async def run(
        self,
        protocol: Protocol,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a protocol.

        Args:
            protocol: Protocol to run
            params: Parameters for the protocol

        Returns:
            Execution results
        """
        params = params or {}
        self.results = dict(params)

        # Validate required params
        for req in protocol.required_params:
            if req not in params:
                return {
                    "status": ProtocolStatus.FAILED.value,
                    "error": f"Missing required param: {req}"
                }

        run_id = f"proto_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.current_run = {
            "id": run_id,
            "protocol": protocol.name,
            "status": ProtocolStatus.RUNNING.value,
            "started": datetime.utcnow().isoformat() + "Z",
            "steps": {}
        }

        self._log_run("started")

        # Execute steps
        completed_steps = set()

        for step in protocol.steps:
            # Check dependencies
            for req in step.requires:
                if req not in completed_steps:
                    # Wait for required step (should be done in order)
                    pass

            # Interpolate params
            step_params = {}
            for key, value in step.params.items():
                if isinstance(value, str):
                    step_params[key] = self._interpolate(value, self.results)
                else:
                    step_params[key] = value

            self.current_run["steps"][step.name] = {
                "status": "running",
                "started": datetime.utcnow().isoformat() + "Z"
            }

            try:
                result = await asyncio.wait_for(
                    self._execute_action(step.action, step_params),
                    timeout=60  # Per-step timeout
                )

                self.results[step.name] = result
                completed_steps.add(step.name)

                self.current_run["steps"][step.name].update({
                    "status": "completed",
                    "completed": datetime.utcnow().isoformat() + "Z"
                })

            except Exception as e:
                self.current_run["steps"][step.name].update({
                    "status": "failed",
                    "error": str(e)
                })

                if step.on_failure == "stop":
                    self.current_run["status"] = ProtocolStatus.FAILED.value
                    self._log_run("failed")
                    return {
                        "status": ProtocolStatus.FAILED.value,
                        "error": f"Step {step.name} failed: {e}",
                        "results": self.results
                    }
                # else continue

        self.current_run["status"] = ProtocolStatus.COMPLETED.value
        self.current_run["completed"] = datetime.utcnow().isoformat() + "Z"

        self._log_run("completed")

        return {
            "status": ProtocolStatus.COMPLETED.value,
            "run_id": run_id,
            "results": self.results
        }

    def _log_run(self, event: str) -> None:
        """Log protocol run."""
        log_file = BRIDGE_PATH / "protocol_runs.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "event": event,
            "run": self.current_run,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_protocol(
    name: str,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run a protocol by name.

    Args:
        name: Protocol name
        params: Protocol parameters

    Returns:
        Execution results
    """
    if name not in PROTOCOLS:
        return {"error": f"Unknown protocol: {name}"}

    executor = ProtocolExecutor()
    return await executor.run(PROTOCOLS[name], params)


def list_protocols() -> List[Dict[str, Any]]:
    """List all available protocols."""
    return [
        {
            "name": p.name,
            "description": p.description,
            "required_params": p.required_params,
            "steps": len(p.steps)
        }
        for p in PROTOCOLS.values()
    ]


# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="WOLF_AI Pack Protocols")
    parser.add_argument("action", choices=["list", "run", "info"])
    parser.add_argument("--name", "-n", help="Protocol name")
    parser.add_argument("--params", "-p", help="JSON params")
    args = parser.parse_args()

    if args.action == "list":
        print("\n📋 Available Protocols:\n")
        for proto in list_protocols():
            print(f"  {proto['name']}")
            print(f"    {proto['description']}")
            if proto['required_params']:
                print(f"    Requires: {', '.join(proto['required_params'])}")
            print()

    elif args.action == "info":
        if not args.name or args.name not in PROTOCOLS:
            print("Error: --name required and must be valid protocol")
            return

        proto = PROTOCOLS[args.name]
        print(f"\n📋 Protocol: {proto.name}")
        print(f"   {proto.description}\n")
        print("   Steps:")
        for i, step in enumerate(proto.steps, 1):
            print(f"     {i}. {step.name} ({step.action})")
        print()

    elif args.action == "run":
        if not args.name:
            print("Error: --name required")
            return

        params = json.loads(args.params) if args.params else {}

        print(f"\n🐺 Running protocol: {args.name}\n")
        result = await run_protocol(args.name, params)

        if result.get("status") == "completed":
            print("✓ Protocol completed successfully")
        else:
            print(f"✗ Protocol failed: {result.get('error')}")

        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
