"""
API Configuration

Re-exports from central config for backward compatibility.
"""
import sys
from pathlib import Path

# Use central config
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    # Paths
    WOLF_ROOT,
    BRIDGE_PATH,
    LOGS_PATH,
    MEMORY_PATH,
    ARENA_PATH,

    # API
    API_HOST,
    API_PORT,
    API_KEY,

    # Telegram
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_ALLOWED_USERS,

    # Ollama
    OLLAMA_URL,
    OLLAMA_MODEL,

    # GitHub
    GITHUB_REPO,
    GITHUB_BRANCH,
    AUTO_SYNC,

    # Debug
    DEBUG
)

__all__ = [
    "WOLF_ROOT", "BRIDGE_PATH", "LOGS_PATH", "MEMORY_PATH", "ARENA_PATH",
    "API_HOST", "API_PORT", "API_KEY",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS",
    "OLLAMA_URL", "OLLAMA_MODEL",
    "GITHUB_REPO", "GITHUB_BRANCH", "AUTO_SYNC",
    "DEBUG"
]
