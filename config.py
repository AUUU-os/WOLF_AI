"""
WOLF_AI Global Configuration

Central configuration for all WOLF_AI components.
Uses .env file or environment variables.
"""
import os
import secrets
from pathlib import Path
from typing import Optional

# Try to load dotenv, but don't fail if not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

def _detect_root() -> Path:
    """Auto-detect WOLF_AI root directory."""
    # Check environment variable first
    env_root = os.getenv("WOLF_ROOT")
    if env_root:
        return Path(env_root)

    # Try to find root by looking for marker files
    current = Path(__file__).parent

    # Go up looking for WOLF_AI markers
    for _ in range(5):
        if (current / "bridge").exists() or (current / "CLAUDE.md").exists():
            return current
        current = current.parent

    # Fallback based on OS
    if os.name == 'nt':  # Windows
        return Path("E:/WOLF_AI")
    else:  # Linux/Mac
        return Path.home() / "WOLF_AI"


# Core paths
WOLF_ROOT = _detect_root()
BRIDGE_PATH = WOLF_ROOT / "bridge"
LOGS_PATH = WOLF_ROOT / "logs"
MEMORY_PATH = WOLF_ROOT / "memory"
ARENA_PATH = WOLF_ROOT / "arena"

# Ensure critical directories exist
for path in [BRIDGE_PATH, LOGS_PATH, MEMORY_PATH, ARENA_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_HOST = os.getenv("WOLF_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("WOLF_API_PORT", "8000"))

# API Key - generate if not set
API_KEY = os.getenv("WOLF_API_KEY")
if API_KEY is None:
    API_KEY = secrets.token_urlsafe(32)
    print(f"\n{'='*60}")
    print(f"[!] GENERATED API KEY (save this to .env):")
    print(f"    WOLF_API_KEY={API_KEY}")
    print(f"{'='*60}\n")

# =============================================================================
# TELEGRAM CONFIGURATION
# =============================================================================

TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_ALLOWED_USERS = [
    u.strip() for u in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")
    if u.strip()
]

# =============================================================================
# OLLAMA / WILK CONFIGURATION
# =============================================================================

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3:latest")

# =============================================================================
# GITHUB SYNC CONFIGURATION
# =============================================================================

GITHUB_REPO = os.getenv("WOLF_GITHUB_REPO", "AUUU-os/WOLF_AI")
GITHUB_BRANCH = os.getenv("WOLF_GITHUB_BRANCH", "main")
AUTO_SYNC = os.getenv("WOLF_AUTO_SYNC", "true").lower() == "true"

# =============================================================================
# MEMORY CONFIGURATION
# =============================================================================

MEMORY_BACKEND = os.getenv("WOLF_MEMORY_BACKEND", "json")  # json, sqlite, chromadb
MEMORY_VECTOR_MODEL = os.getenv("WOLF_MEMORY_VECTOR_MODEL", "all-MiniLM-L6-v2")

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = os.getenv("WOLF_LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("WOLF_LOG_FORMAT", "%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# =============================================================================
# DEBUG
# =============================================================================

DEBUG = os.getenv("WOLF_DEBUG", "false").lower() == "true"

if DEBUG:
    print(f"[CONFIG] WOLF_ROOT: {WOLF_ROOT}")
    print(f"[CONFIG] BRIDGE_PATH: {BRIDGE_PATH}")
    print(f"[CONFIG] API: {API_HOST}:{API_PORT}")
