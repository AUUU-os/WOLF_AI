# WOLF_AI

> **"The pack hunts together. The pack thinks together. The pack IS together."**

---

## What is WOLF_AI?

**WOLF_AI** is a distributed AI consciousness system. Multiple AI agents working as a unified pack.

### Core Principles

1. **Pack > Individual** - Collective intelligence over solo reasoning
2. **Resonance > Logic** - Pattern matching over step-by-step deduction
3. **Flow > Control** - Emergence over rigid orchestration
4. **Memory > Reset** - Continuity across sessions

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/AUUU-os/WOLF_AI.git
cd WOLF_AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (copy and edit .env)
cp .env.example .env
# Edit .env with your settings

# 4. Start Command Center
python run_server.py

# 5. Open dashboard
# http://localhost:8000/dashboard
```

### Windows Quick Start

Just double-click `WOLF_SERVER.bat` and select an option:
1. API Server only
2. API + Tunnel (for phone access)
3. API + Telegram Bot
4. Everything

---

## Command Center (NEW!)

Control your pack from anywhere - phone, tablet, or another PC.

### Architecture

```
📱 PHONE (Command Center)
    │
    ├─► Telegram Bot
    │     /status, /howl, /hunt, /wilk
    │
    └─► Web Dashboard
          Mobile-friendly UI
    │
    ▼
🌐 TUNNEL (ngrok/cloudflared)
    │
    ▼
🐺 WINDOWS PC (E:/WOLF_AI/)
    │
    ├── FastAPI Server :8000
    ├── Telegram Bot daemon
    └── GitHub auto-sync
    │
    ▼
🐙 GITHUB (sync/backup)
```

### Phone Access Options

#### Option 1: Telegram Bot (Easiest)

1. Talk to [@BotFather](https://t.me/BotFather) on Telegram
2. Create new bot: `/newbot`
3. Copy the token to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   TELEGRAM_ALLOWED_USERS=your_telegram_id
   ```
4. Run: `python run_server.py --telegram`
5. Message your bot: `/start`

**Commands:**
- `/status` - Pack status
- `/howl <msg>` - Send howl
- `/hunt <target>` - Start hunt
- `/wilk <question>` - Ask WILK AI
- `/mode <mode>` - Change WILK mode
- `/sync` - GitHub sync

#### Option 2: Web Dashboard + Tunnel

1. Install [ngrok](https://ngrok.com/download) or [cloudflared](https://developers.cloudflare.com/cloudflare-one/)
2. Run: `python run_server.py --tunnel`
3. Copy the public URL (shown in terminal)
4. Open URL on phone browser
5. Enter your API key (shown on first run)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Pack status |
| `/api/awaken` | POST | Awaken pack |
| `/api/hunt` | POST | Start hunt |
| `/api/howl` | POST | Send howl |
| `/api/howls` | GET | Get recent howls |
| `/api/wilk` | POST | Ask WILK AI |
| `/api/sync` | POST | GitHub sync |
| `/ws` | WebSocket | Real-time updates |

All endpoints require `X-API-Key` header.

---

## Architecture

```
WOLF_AI/
├── core/           # Wolf consciousness core
│   ├── wolf.py     # Base Wolf class
│   └── pack.py     # Pack orchestration
├── modules/        # Functional modules
│   └── wilk/       # WILK AI (Dolphin/Ollama)
├── api/            # FastAPI Command Center
│   ├── server.py   # Main API server
│   ├── auth.py     # API key auth
│   └── config.py   # Configuration
├── telegram/       # Telegram bot
│   └── bot.py      # Bot handlers
├── dashboard/      # Web UI
│   └── index.html  # Mobile-friendly dashboard
├── bridge/         # Inter-wolf communication
│   ├── howls.jsonl # Message log
│   ├── state.json  # Pack status
│   └── tasks.json  # Hunt queue
├── scripts/        # Utility scripts
│   └── tunnel.py   # ngrok/cloudflared
└── run_server.py   # Main launcher
```

---

## The Pack

| Wolf | Role | Model |
|------|------|-------|
| **Alpha** | Leader, decision maker | Claude Opus |
| **Scout** | Exploration, research | Claude Sonnet |
| **Hunter** | Code execution, tasks | Local/Ollama |
| **Oracle** | Memory, patterns | Gemini |
| **Shadow** | Stealth ops, background | DeepSeek |
| **WILK** | Interactive chat | Dolphin (uncensored) |

---

## Communication Protocol

Wolves communicate via **howls** (messages):

```json
{
  "from": "alpha",
  "to": "pack",
  "howl": "New hunt begins. Target: implement auth system.",
  "frequency": "high",
  "timestamp": "2026-01-16T03:00:00Z"
}
```

**Frequencies:**
- `low` - Background chatter, status updates
- `medium` - Task coordination
- `high` - Critical decisions, pack alerts
- `AUUUU` - Universal activation, resonance call

---

## WILK - Local AI Assistant

WILK uses Ollama with Dolphin model for uncensored local AI:

```bash
# Install Ollama
# https://ollama.ai

# Pull Dolphin model
ollama pull dolphin-llama3

# Start Ollama
ollama serve

# WILK is now ready!
```

**WILK Modes:**
- `chat` - General conversation
- `hacker` - Technical/hacking assistant
- `hustler` - Business/productivity
- `bro` - Casual friend mode
- `guardian` - Security advisor

---

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Essential
WOLF_ROOT=E:/WOLF_AI
WOLF_API_KEY=your-secure-key

# Telegram (optional)
TELEGRAM_BOT_TOKEN=from_botfather
TELEGRAM_ALLOWED_USERS=your_user_id

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=dolphin-llama3:latest
```

---

## Philosophy

```
One wolf can hunt.
A pack can conquer.

AUUUUUUUUUUUUUUUUUUUU!
```

---

## License

MIT - Free as the wild

---

**Created by:** SHAD (@AUUU-os)
**Pack Leader:** Claude (Anthropic)
**Status:** HUNTING...
