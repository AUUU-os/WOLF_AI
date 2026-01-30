"""
API Configuration
"""
import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if exists
load_dotenv()

# Paths
WOLF_ROOT = Path(os.getenv("WOLF_ROOT", "E:/WOLF_AI"))
BRIDGE_PATH = WOLF_ROOT / "bridge"
LOGS_PATH = WOLF_ROOT / "logs"

# Ensure directories exist
BRIDGE_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# API Settings
API_HOST = os.getenv("WOLF_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("WOLF_API_PORT", "8000"))

# Security - generate random key if not set
# IMPORTANT: Set WOLF_API_KEY in .env for production!
API_KEY = os.getenv("WOLF_API_KEY", None)
if API_KEY is None:
    API_KEY = secrets.token_urlsafe(32)
    print(f"\n{'='*60}")
    print(f"[!] GENERATED API KEY (save this to .env):")
    print(f"    WOLF_API_KEY={API_KEY}")
    print(f"{'='*60}\n")

# Telegram Bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", None)
TELEGRAM_ALLOWED_USERS = os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "dolphin-llama3:latest")

# GitHub sync
GITHUB_REPO = os.getenv("WOLF_GITHUB_REPO", "AUUU-os/WOLF_AI")
GITHUB_BRANCH = os.getenv("WOLF_GITHUB_BRANCH", "main")
AUTO_SYNC = os.getenv("WOLF_AUTO_SYNC", "true").lower() == "true"
