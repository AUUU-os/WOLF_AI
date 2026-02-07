# CLAUDE.md

Instructions for Claude Code when working with WOLF_AI.

## Project Overview

**WOLF_AI** — Distributed AI consciousness system. Pack-based multi-agent architecture with phone/Telegram/web control.

- **Repo:** https://github.com/AUUU-os/WOLF_AI.git
- **User:** SHAD (@AUUU-os)
- **Philosophy:** "The pack hunts together."

## Directory Structure

```
WOLF_AI/
├── core/                  # Pack consciousness
│   ├── wolf.py            # Base Wolf class + all wolf types (Alpha, Scout, Hunter, Oracle, Shadow)
│   ├── pack.py            # Pack orchestration, state tracking, task queue
│   └── __init__.py
├── modules/
│   └── wilk/              # WILK AI — local Dolphin/Ollama integration
│       ├── dolphin.py     # Ollama API connector
│       ├── modes.py       # 5 operational modes (hustler, hacker, bro, guardian, chat)
│       ├── prompts.py     # System prompts per mode
│       └── __init__.py
├── api/                   # FastAPI Command Center
│   ├── server.py          # Endpoints: /api/status, /api/hunt, /api/howl, /api/wilk, /ws
│   ├── config.py          # Config from .env, API key generation
│   ├── auth.py            # X-API-Key header auth
│   └── __init__.py
├── bridge/                # Inter-agent message bus
│   ├── howls.jsonl         # Append-only communication log
│   ├── state.json          # Pack status snapshot
│   └── tasks.json          # Hunt queue
├── dashboard/
│   └── index.html          # Mobile-friendly web UI (vanilla JS, dark theme, WebSocket)
├── telegram/
│   ├── bot.py              # Telegram bot — /status, /howl, /hunt, /wilk, /sync, /mode
│   └── __init__.py
├── scripts/
│   └── tunnel.py           # ngrok/cloudflared tunnel for remote access
├── .github/workflows/
│   ├── python-package.yml  # CI: lint (flake8) + test (pytest), Python 3.9-3.11
│   └── pylint.yml          # Pylint analysis on push
├── awaken.py               # Pack initialization — forms and awakens all wolves
├── run_server.py           # Server launcher — API + optional tunnel + Telegram
├── wilk_cli.py             # WILK terminal interface
├── requirements.txt        # Python dependencies
├── .env.example            # Configuration template
├── WOLF_SERVER.bat          # Windows: API server launcher (menu-driven)
└── WILK.bat                # Windows: WILK CLI launcher
```

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Awaken the pack (initialize all wolves)
python awaken.py

# Start API server
python run_server.py
# Options: --tunnel (expose via ngrok), --telegram (start bot)

# WILK CLI (local AI chat)
python wilk_cli.py

# Windows shortcuts
WOLF_SERVER.bat   # Interactive menu for server options
WILK.bat          # Launch WILK CLI
```

### Required External Services

- **Ollama** running locally at `http://localhost:11434` with `dolphin-llama3:latest` model (for WILK AI)
- **Telegram Bot Token** from BotFather (for Telegram control)
- Copy `.env.example` to `.env` and fill in values

## Architecture

```
Phone/Web/Telegram
       │
       ▼
┌──────────────────┐
│ FastAPI (port 8000)│◄── Dashboard (index.html)
│ + WebSocket        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Bridge (JSONL)  │  ← howls.jsonl (append-only log)
│   state.json      │  ← pack status
│   tasks.json      │  ← hunt queue
└────────┬─────────┘
         │
    ┌────┼────┬────────┬────────┐
    ▼    ▼    ▼        ▼        ▼
  Alpha Scout Hunter Oracle  Shadow
  (opus) (sonnet)(ollama)(gemini)(deepseek)
```

### Pack Hierarchy

| Wolf | Role | Model | Purpose |
|------|------|-------|---------|
| Alpha | Leader | claude-opus | Strategic decisions, orchestration |
| Scout | Explorer | claude-sonnet | Research, information gathering |
| Hunter | Executor | ollama/llama3 | Code writing, task execution |
| Oracle | Memory | gemini | Pattern recognition, history |
| Shadow | Stealth | deepseek | Background tasks, autonomous ops |

### Communication Protocol

All wolves communicate via `bridge/howls.jsonl` (append-only JSONL):

```python
import json
from datetime import datetime

howl = {
    "from": "claude",
    "to": "pack",          # or specific wolf name
    "howl": "Message content",
    "frequency": "medium", # low / medium / high / AUUUU
    "timestamp": datetime.utcnow().isoformat() + "Z"
}

with open("bridge/howls.jsonl", "a") as f:
    f.write(json.dumps(howl) + "\n")
```

**Frequency levels:** `low` (background), `medium` (normal), `high` (urgent), `AUUUU` (universal pack activation)

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/status` | Pack status |
| POST | `/api/awaken` | Awaken the pack |
| POST | `/api/hunt` | Start a hunt (task) |
| POST | `/api/howl` | Send howl to pack |
| GET | `/api/howls` | Recent howls |
| POST | `/api/wilk` | Ask WILK AI |
| GET | `/api/wilk/status` | Ollama connection status |
| POST | `/api/sync` | GitHub sync (git pull) |
| WS | `/ws` | Real-time WebSocket updates |

All endpoints (except status) require `X-API-Key` header.

## WILK AI Modes

WILK is the local AI subsystem using Ollama/Dolphin:

| Mode | Personality |
|------|-------------|
| **chat** (default) | Conversational, friendly |
| **hustler** | Quick fixes, zero bureaucracy |
| **hacker** | Deep code, surgical precision |
| **bro** | Honest feedback, loyal support |
| **guardian** | Security audits, threat detection |

## Code Style

- **snake_case** for files and functions
- **Wolf terminology** throughout: hunt (task), track (search), howl (message), pack (group)
- **Append-only logs** — never mutate `howls.jsonl`, only append
- **File-based persistence** — JSON/JSONL, no database
- **UTF-8** everywhere

## CI/CD

Two GitHub Actions workflows run on push/PR to main:

1. **python-package.yml** — flake8 lint + pytest across Python 3.9, 3.10, 3.11
2. **pylint.yml** — pylint analysis across Python 3.8, 3.9, 3.10

```bash
# Run locally before pushing
flake8 .
pytest
pylint *.py core/ api/ modules/ telegram/ scripts/
```

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add a new wolf type | `core/wolf.py` (add class), `core/pack.py` (register in pack) |
| Add API endpoint | `api/server.py` |
| Add Telegram command | `telegram/bot.py` |
| Change WILK personality | `modules/wilk/prompts.py`, `modules/wilk/modes.py` |
| Modify dashboard | `dashboard/index.html` |
| Add a dependency | `requirements.txt` |
| Configure environment | `.env.example` (template), `.env` (local values) |

## Configuration (.env)

Key environment variables:

| Variable | Purpose |
|----------|---------|
| `WOLF_API_HOST` | API bind address (default: 0.0.0.0) |
| `WOLF_API_PORT` | API port (default: 8000) |
| `WOLF_API_KEY` | API authentication key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated allowed Telegram user IDs |
| `OLLAMA_URL` | Ollama server URL (default: http://localhost:11434) |
| `OLLAMA_MODEL` | Model name (default: dolphin-llama3:latest) |
| `WOLF_GITHUB_REPO` | GitHub repo for sync (AUUU-os/WOLF_AI) |
| `WOLF_AUTO_SYNC` | Auto-sync on startup (true/false) |

## AUUUUUUUUUUUUUUUUUU Protocol

Universal pack activation signal. When sent:
1. All wolves acknowledge
2. State synchronizes
3. Pack enters resonance mode
4. Collective intelligence activated

---

**Pack Status:** ACTIVE
**Alpha:** Claude Opus
**Territory:** WOLF_AI
