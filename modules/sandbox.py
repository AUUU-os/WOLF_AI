"""
Hunter Sandbox - Docker-based safe code execution

Provides isolated execution environment for Hunter wolf.
Runs code in disposable containers with resource limits.

Usage:
    from modules.sandbox import Sandbox, execute_code

    # Quick execution
    result = await execute_code("print('Hello Wolf!')", language="python")

    # With sandbox instance
    sandbox = Sandbox()
    result = await sandbox.run("console.log('AUUUU!')", language="javascript")
"""

import os
import json
import asyncio
import tempfile
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIDGE_PATH

# Check Docker availability
DOCKER_AVAILABLE = False
try:
    import subprocess
    result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
    DOCKER_AVAILABLE = result.returncode == 0
except Exception:
    pass


class Language(Enum):
    """Supported languages for sandbox execution."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    NODE = "node"
    BASH = "bash"
    RUBY = "ruby"
    GO = "go"
    RUST = "rust"


# Docker images for each language
LANGUAGE_IMAGES = {
    Language.PYTHON: "python:3.11-slim",
    Language.JAVASCRIPT: "node:20-slim",
    Language.NODE: "node:20-slim",
    Language.BASH: "alpine:latest",
    Language.RUBY: "ruby:3.2-slim",
    Language.GO: "golang:1.21-alpine",
    Language.RUST: "rust:1.75-slim",
}

# File extensions for each language
LANGUAGE_EXTENSIONS = {
    Language.PYTHON: ".py",
    Language.JAVASCRIPT: ".js",
    Language.NODE: ".js",
    Language.BASH: ".sh",
    Language.RUBY: ".rb",
    Language.GO: ".go",
    Language.RUST: ".rs",
}

# Run commands for each language
LANGUAGE_COMMANDS = {
    Language.PYTHON: ["python", "/code/script.py"],
    Language.JAVASCRIPT: ["node", "/code/script.js"],
    Language.NODE: ["node", "/code/script.js"],
    Language.BASH: ["sh", "/code/script.sh"],
    Language.RUBY: ["ruby", "/code/script.rb"],
    Language.GO: ["go", "run", "/code/script.go"],
    Language.RUST: ["sh", "-c", "cd /code && rustc script.rs -o script && ./script"],
}


@dataclass
class ExecutionResult:
    """Result of sandbox execution."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    language: str
    container_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Sandbox:
    """
    Docker-based sandbox for safe code execution.

    Resources are limited:
    - Memory: 256MB (configurable)
    - CPU: 0.5 cores
    - Network: disabled by default
    - Time: 30s timeout
    """

    def __init__(
        self,
        memory_limit: str = "256m",
        cpu_limit: float = 0.5,
        timeout: int = 30,
        network_enabled: bool = False
    ):
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout = timeout
        self.network_enabled = network_enabled
        self._temp_dirs: List[str] = []

    @property
    def is_available(self) -> bool:
        """Check if Docker sandbox is available."""
        return DOCKER_AVAILABLE

    def _log_execution(self, result: ExecutionResult, code_preview: str) -> None:
        """Log execution to bridge."""
        log_entry = {
            "type": "sandbox_execution",
            "success": result.success,
            "language": result.language,
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
            "code_preview": code_preview[:100],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        log_file = BRIDGE_PATH / "sandbox_log.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _create_temp_dir(self) -> str:
        """Create temporary directory for code."""
        temp_dir = tempfile.mkdtemp(prefix="wolf_sandbox_")
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def _cleanup_temp_dirs(self) -> None:
        """Cleanup all temporary directories."""
        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
        self._temp_dirs = []

    def _get_language(self, language: str) -> Language:
        """Parse language string to enum."""
        language = language.lower().strip()
        for lang in Language:
            if lang.value == language:
                return lang
        # Default to Python
        return Language.PYTHON

    async def run(
        self,
        code: str,
        language: str = "python",
        files: Optional[Dict[str, str]] = None,
        env_vars: Optional[Dict[str, str]] = None
    ) -> ExecutionResult:
        """
        Execute code in Docker sandbox.

        Args:
            code: Source code to execute
            language: Programming language
            files: Additional files to include {filename: content}
            env_vars: Environment variables

        Returns:
            ExecutionResult with output
        """
        lang = self._get_language(language)
        start_time = datetime.utcnow()

        if not self.is_available:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Docker not available",
                execution_time=0,
                language=lang.value,
                error="Docker not installed or not running"
            )

        # Create temp directory with code
        temp_dir = self._create_temp_dir()
        script_file = f"script{LANGUAGE_EXTENSIONS[lang]}"
        script_path = Path(temp_dir) / script_file

        # Write main script
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Write additional files
        if files:
            for filename, content in files.items():
                file_path = Path(temp_dir) / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

        # Build Docker command
        container_name = f"wolf_sandbox_{uuid.uuid4().hex[:8]}"
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution
            "--name", container_name,
            "-m", self.memory_limit,
            f"--cpus={self.cpu_limit}",
            "-v", f"{temp_dir}:/code:ro",  # Read-only mount
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",  # Drop all capabilities
            "--read-only",  # Read-only container filesystem
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",  # Temp space
        ]

        # Network isolation
        if not self.network_enabled:
            docker_cmd.extend(["--network", "none"])

        # Environment variables
        if env_vars:
            for key, value in env_vars.items():
                docker_cmd.extend(["-e", f"{key}={value}"])

        # Image and command
        docker_cmd.append(LANGUAGE_IMAGES[lang])
        docker_cmd.extend(LANGUAGE_COMMANDS[lang])

        try:
            # Run with timeout
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                # Kill the container
                try:
                    await asyncio.create_subprocess_exec(
                        "docker", "kill", container_name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                except Exception:
                    pass

                execution_time = (datetime.utcnow() - start_time).total_seconds()
                result = ExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Execution timed out after {self.timeout}s",
                    execution_time=execution_time,
                    language=lang.value,
                    container_id=container_name,
                    error="timeout"
                )
                self._log_execution(result, code)
                return result

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            result = ExecutionResult(
                success=process.returncode == 0,
                exit_code=process.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                execution_time=execution_time,
                language=lang.value,
                container_id=container_name
            )

            self._log_execution(result, code)
            return result

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result = ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                language=lang.value,
                error=str(e)
            )
            self._log_execution(result, code)
            return result

        finally:
            self._cleanup_temp_dirs()

    async def run_script(
        self,
        script_path: str,
        language: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a script file in sandbox.

        Args:
            script_path: Path to script file
            language: Language (auto-detected if not provided)

        Returns:
            ExecutionResult
        """
        script_path = Path(script_path)

        if not script_path.exists():
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Script not found: {script_path}",
                execution_time=0,
                language=language or "unknown",
                error="file_not_found"
            )

        # Auto-detect language from extension
        if not language:
            ext = script_path.suffix.lower()
            ext_map = {v: k.value for k, v in LANGUAGE_EXTENSIONS.items()}
            language = ext_map.get(ext, "python")

        code = script_path.read_text(encoding="utf-8")
        return await self.run(code, language)

    def __del__(self):
        """Cleanup on destruction."""
        self._cleanup_temp_dirs()


# =============================================================================
# FALLBACK: Local execution (less secure, for development)
# =============================================================================

class LocalSandbox:
    """
    Local execution fallback when Docker is not available.
    WARNING: Less secure - only use for development/testing.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        return True  # Always available

    async def run(
        self,
        code: str,
        language: str = "python"
    ) -> ExecutionResult:
        """Execute code locally (Python only for safety)."""
        if language.lower() not in ["python", "py"]:
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Local sandbox only supports Python",
                execution_time=0,
                language=language,
                error="unsupported_language"
            )

        start_time = datetime.utcnow()

        # Create temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ExecutionResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Timeout after {self.timeout}s",
                    execution_time=self.timeout,
                    language="python",
                    error="timeout"
                )

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return ExecutionResult(
                success=process.returncode == 0,
                exit_code=process.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                execution_time=execution_time,
                language="python"
            )

        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_sandbox: Optional[Sandbox] = None


def get_sandbox() -> Sandbox:
    """Get or create sandbox singleton."""
    global _sandbox
    if _sandbox is None:
        _sandbox = Sandbox()
    return _sandbox


async def execute_code(
    code: str,
    language: str = "python",
    use_docker: bool = True
) -> ExecutionResult:
    """
    Execute code in sandbox.

    Args:
        code: Source code
        language: Programming language
        use_docker: Use Docker if available (fallback to local)

    Returns:
        ExecutionResult
    """
    if use_docker and DOCKER_AVAILABLE:
        sandbox = get_sandbox()
    else:
        sandbox = LocalSandbox()

    return await sandbox.run(code, language)


# =============================================================================
# HUNTER INTEGRATION
# =============================================================================

class HunterExecutor:
    """
    Execution engine for Hunter wolf.

    Provides safe code execution with approval workflow.
    """

    def __init__(self, require_approval: bool = True):
        self.require_approval = require_approval
        self.sandbox = Sandbox() if DOCKER_AVAILABLE else LocalSandbox()
        self.pending_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[ExecutionResult] = []

    async def execute(
        self,
        code: str,
        language: str = "python",
        approved: bool = False,
        task_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute code with optional approval check.

        Args:
            code: Source code
            language: Programming language
            approved: Whether execution is approved
            task_id: Optional task ID for tracking

        Returns:
            ExecutionResult
        """
        exec_id = task_id or f"exec_{uuid.uuid4().hex[:8]}"

        if self.require_approval and not approved:
            # Queue for approval
            self.pending_executions[exec_id] = {
                "code": code,
                "language": language,
                "created": datetime.utcnow().isoformat() + "Z"
            }
            return ExecutionResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Execution requires approval",
                execution_time=0,
                language=language,
                error="approval_required"
            )

        # Execute
        result = await self.sandbox.run(code, language)
        self.execution_history.append(result)

        # Remove from pending if it was there
        self.pending_executions.pop(exec_id, None)

        return result

    def approve(self, exec_id: str) -> bool:
        """Approve a pending execution."""
        if exec_id in self.pending_executions:
            self.pending_executions[exec_id]["approved"] = True
            return True
        return False

    def get_pending(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending executions."""
        return self.pending_executions.copy()

    def get_history(self, limit: int = 10) -> List[ExecutionResult]:
        """Get execution history."""
        return self.execution_history[-limit:]


# =============================================================================
# CLI
# =============================================================================

async def main():
    """Test sandbox execution."""
    import argparse

    parser = argparse.ArgumentParser(description="WOLF_AI Hunter Sandbox")
    parser.add_argument("--code", type=str, help="Code to execute")
    parser.add_argument("--file", type=str, help="Script file to execute")
    parser.add_argument("--lang", default="python", help="Language")
    parser.add_argument("--local", action="store_true", help="Force local execution")
    args = parser.parse_args()

    print("\n🐺 Hunter Sandbox")
    print(f"   Docker available: {DOCKER_AVAILABLE}")
    print()

    if args.file:
        sandbox = Sandbox() if DOCKER_AVAILABLE and not args.local else LocalSandbox()
        result = await sandbox.run_script(args.file, args.lang)
    elif args.code:
        result = await execute_code(args.code, args.lang, use_docker=not args.local)
    else:
        # Demo
        demo_code = '''
print("🐺 Hello from Hunter Sandbox!")
print("AUUUUUUUUUUUUUUUUUU!")

import sys
print(f"Python version: {sys.version}")
'''
        print(f"Running demo code...")
        result = await execute_code(demo_code)

    print(f"Success: {result.success}")
    print(f"Exit code: {result.exit_code}")
    print(f"Time: {result.execution_time:.2f}s")
    print(f"\n--- STDOUT ---\n{result.stdout}")
    if result.stderr:
        print(f"\n--- STDERR ---\n{result.stderr}")


if __name__ == "__main__":
    asyncio.run(main())
