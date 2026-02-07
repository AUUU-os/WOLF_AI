---
name: wolf_pack
description: Control your WOLF_AI pack - distributed AI consciousness system
version: 1.0.0
author: SHAD (@AUUU-os)
tags: [ai, agents, wolf, pack, multi-agent]
requires:
  - curl
env:
  WOLF_API_URL: http://localhost:8000
  WOLF_API_KEY: ${WOLF_API_KEY}
---

# WOLF_AI Pack Skill

Control your wolf pack from OpenClaw. The pack consists of 5 specialized AI agents:

- **Alpha** - Leader, strategic decisions
- **Scout** - Research, exploration
- **Hunter** - Code execution, tasks
- **Oracle** - Memory, patterns
- **Shadow** - Stealth operations

## Commands

### Get Pack Status
```bash
wolf status
```
Returns current pack status, active wolves, and running hunts.

### Awaken the Pack
```bash
wolf awaken
```
Forms and activates all wolves in the pack.

### Send Howl (Message)
```bash
wolf howl "Your message here"
wolf howl "Emergency!" --frequency high
```
Send a message to the pack. Frequencies: low, medium, high, AUUUU

### Start Hunt (Task)
```bash
wolf hunt "Build a REST API"
wolf hunt "Fix bug in auth.py" --assign hunter
```
Assign a task to the pack.

### Ask WILK (Local AI)
```bash
wolf wilk "How do I implement OAuth?"
wolf wilk "Explain this code" --mode hacker
```
Ask WILK (Dolphin/Ollama) a question. Modes: chat, hacker, hustler, bro, guardian

### Sync with GitHub
```bash
wolf sync
```
Pull latest changes from GitHub repository.

## Examples

```
User: Check on my wolf pack
Assistant: [uses wolf status]

User: I need help building an API
Assistant: [uses wolf hunt "Build REST API for user management"]

User: Send alert to the pack about the deadline
Assistant: [uses wolf howl "Deadline approaching!" --frequency high]
```

## Integration

This skill connects to WOLF_AI API running locally. Make sure to:

1. Start WOLF_AI server: `python run_server.py`
2. Set API key in environment: `WOLF_API_KEY=your_key`

The pack hunts together. AUUUUUUUUUUUUUUUUUU!
