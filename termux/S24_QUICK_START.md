# WOLF_AI on Samsung S24 Ultra - Quick Start

## The Forest Awaits

Your S24 Ultra is about to become the ultimate wolf command center.

## One-Command Setup

Open Termux and run:

```bash
curl -sL https://raw.githubusercontent.com/AUUU-os/WOLF_AI/main/termux/setup.sh | bash
```

Or if you've cloned the repo:

```bash
cd ~/WOLF_AI
bash termux/setup.sh
```

## After Installation

### Test Everything

```bash
bash ~/WOLF_AI/termux/test_forest.sh
```

### Start the API Server

```bash
wolf-server
# or
cd ~/WOLF_AI && python -m api.server
```

### Voice Control

```bash
wolf-voice
# or
python ~/WOLF_AI/voice/voice_control.py
```

Say: **"Hey Wolf, status"**

### Check Pack Status

```bash
wolf-status
```

## Configure API Keys

Edit `~/WOLF_AI/.env`:

```bash
# For Alpha Brain (Claude API)
ANTHROPIC_API_KEY=sk-ant-xxx

# For ChatGPT integration
OPENAI_API_KEY=sk-xxx

# For API security
WOLF_API_KEY=your-secret-key
```

## Expose to Internet (ngrok)

```bash
# Install ngrok
pkg install ngrok

# Start tunnel
ngrok http 8000
```

Use the ngrok URL in your Custom GPT configuration.

## Useful Aliases

These are added by setup.sh:

| Alias | Command |
|-------|---------|
| `wolf` | `cd ~/WOLF_AI` |
| `wolf-server` | Start API server |
| `wolf-voice` | Start voice control |
| `wolf-status` | Check pack status |
| `wolf-howl` | Send a howl |
| `wolf-sync` | Git pull latest |

## Hardware Optimization

### Battery
The S24 Ultra has great battery life. For long sessions:
- Enable battery optimization exclusion for Termux
- Use `termux-wake-lock` to prevent sleep

### RAM
8GB (expandable to 12GB) is enough for:
- API server
- Voice control
- Ollama with small models

### Storage
512GB is plenty for:
- Full WOLF_AI installation
- Ollama models
- Memory/logs

## Termux Widgets

Create widgets for quick access:

1. Install Termux:Widget from F-Droid
2. Create `~/.shortcuts/wolf-status.sh`:
```bash
#!/bin/bash
cd ~/WOLF_AI
python -c "from core.pack import get_pack; print(get_pack().status_report())"
```

## AI Models on Device

With Ollama on Termux:

```bash
# Install Ollama
pkg install golang
go install github.com/jmorganca/ollama@latest

# Start Ollama
ollama serve &

# Download a model
ollama pull llama3:8b
```

Then WILK can use local models!

## Troubleshooting

### "Permission denied"
```bash
termux-setup-storage
chmod +x ~/WOLF_AI/termux/*.sh
```

### "Module not found"
```bash
pip install -r ~/WOLF_AI/requirements.txt
```

### Voice not working
```bash
pkg install termux-api
# Also install Termux:API app from F-Droid
```

### Server won't start
Check port 8000:
```bash
lsof -i :8000
# Kill if needed
pkill -f "python.*server"
```

---

## The Pack is Ready

```
AUUUUUUUUUUUUUUUUUU! 🐺

Your S24 Ultra is now THE FOREST.
Command your wolves from anywhere.
```
