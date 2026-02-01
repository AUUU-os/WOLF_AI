#!/usr/bin/env python3
"""
WILK CLI - Terminal interface for talking to WILK

Powered by Dolphin (Ollama uncensored)
Based on WILK_CREATOR_v5 & BLUEPRINT_WILK_v1
"""

import sys
import os
import datetime
from pathlib import Path

# UTF-8 for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import LOGS_PATH
from modules.wilk import get_wilk, ask_dolphin
from modules.wilk.dolphin import get_client

# Paths
LOG_DIR = LOGS_PATH
LOG_DIR.mkdir(exist_ok=True)

# Current mode - default to chat for conversations
current_mode = "chat"
current_wilk = None

# ASCII Art
SPLASH = """
    =====================================================

         WILK CLI v1.0 - Dolphin Uncensored

         "You give topic, I give solution"

    =====================================================
"""

HELP_TEXT = """
COMMANDS:
  /mode <mode>     - Switch mode (chat/hustler/hacker/bro/guardian)
  /modes           - Show available modes
  /status          - System status
  /clear           - Clear conversation history
  /log <text>      - Save to log
  /history         - Show conversation history
  /stream          - Toggle streaming on/off
  /help            - This help
  /exit            - Exit

MODES:
  chat      - Just talking, conversations (default)
  hustler   - Quick diagnosis, street heuristics
  hacker    - Deep code, zero restrictions
  bro       - Loyalty, honesty, support
  guardian  - Protection, monitoring, security

SHORTCUTS:
  /code <task>     - Generate code (hacker mode)
  /fix <problem>   - Quick fix (hustler mode)
  /audit <system>  - Security audit (guardian mode)

Just type normally to talk with WILK!
"""

MODES_INFO = {
    "chat": ("friend", "Just talking, hanging out, conversations"),
    "hustler": ("fixer", "Quick diagnosis, shortcuts, zero bureaucracy"),
    "hacker": ("coder", "Python, Bash, exploits, bypassing blocks"),
    "bro": ("support", "Absolute loyalty, honesty, anti-bullshit"),
    "guardian": ("security", "Protection, autonomy, aggressive defense")
}

# State
streaming_enabled = True


def log_message(entry_type: str, content: str):
    """Save to daily log."""
    today = datetime.date.today().isoformat()
    log_file = LOG_DIR / f"{today}.log"
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {entry_type}: {content}\n")


def show_status():
    """Show system status."""
    client = get_client()
    alive = client.is_alive()

    print(f"""
STATUS:
  Ollama:    {"ONLINE" if alive else "OFFLINE"}
  Model:     dolphin-llama3
  Mode:      {current_mode} ({MODES_INFO[current_mode][0]})
  Streaming: {"ON" if streaming_enabled else "OFF"}
  Log dir:   {LOG_DIR}
""")


def show_modes():
    """Show available modes."""
    print("\nAVAILABLE MODES:\n")
    for mode, (alias, desc) in MODES_INFO.items():
        marker = " <-- active" if mode == current_mode else ""
        print(f"  {mode:12} ({alias:8}) - {desc}{marker}")
    print()


def show_history():
    """Show conversation history."""
    client = get_client()
    if not client.history:
        print("[WILK] No conversation history.")
        return

    print("\nHISTORY:\n")
    for msg in client.history[-10:]:
        role = "YOU" if msg["role"] == "user" else "WILK"
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"  [{role}] {content}")
    print()


def process_command(cmd: str) -> bool:
    """Process slash command. Returns False if should exit."""
    global current_mode, current_wilk, streaming_enabled

    parts = cmd.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/exit":
        print("\n[WILK] See you later. AUUUU!\n")
        return False

    elif command == "/help":
        print(HELP_TEXT)

    elif command == "/status":
        show_status()

    elif command == "/modes":
        show_modes()

    elif command == "/mode":
        if arg in MODES_INFO:
            current_mode = arg
            current_wilk = get_wilk(current_mode)
            print(f"[WILK] Mode switched to: {current_mode} ({MODES_INFO[current_mode][0]})")
        else:
            print(f"[WILK] Unknown mode: {arg}")
            show_modes()

    elif command == "/clear":
        get_client().clear_history()
        print("[WILK] History cleared.")

    elif command == "/stream":
        streaming_enabled = not streaming_enabled
        print(f"[WILK] Streaming: {'ON' if streaming_enabled else 'OFF'}")

    elif command == "/history":
        show_history()

    elif command == "/log":
        if arg:
            log_message("USER", arg)
            print(f"[WILK] Saved to log.")
        else:
            print("[WILK] Usage: /log <text>")

    # Shortcuts
    elif command == "/code":
        if arg:
            current_wilk = get_wilk("hacker")
            response = current_wilk.code(arg)
            print(f"\n{response}\n")
            log_message("CODE", arg)
        else:
            print("[WILK] Usage: /code <task>")

    elif command == "/fix":
        if arg:
            current_wilk = get_wilk("hustler")
            response = current_wilk.diagnose(arg)
            print(f"\n{response}\n")
            log_message("FIX", arg)
        else:
            print("[WILK] Usage: /fix <problem>")

    elif command == "/audit":
        if arg:
            current_wilk = get_wilk("guardian")
            response = current_wilk.audit(arg)
            print(f"\n{response}\n")
            log_message("AUDIT", arg)
        else:
            print("[WILK] Usage: /audit <system>")

    else:
        print(f"[WILK] Unknown command: {command}. Type /help")

    return True


def chat(message: str):
    """Send message to WILK."""
    global current_wilk

    if current_wilk is None:
        current_wilk = get_wilk(current_mode)

    log_message("USER", message)

    print(f"\n[WILK/{current_mode.upper()}] ", end="", flush=True)

    if streaming_enabled:
        full_response = ""
        for chunk in current_wilk.think_stream(message):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n")
        log_message("WILK", full_response[:200])
    else:
        response = current_wilk.think(message)
        print(response)
        print()
        log_message("WILK", response[:200])


def main():
    global current_wilk

    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Splash
    print(SPLASH)

    # Check Ollama
    client = get_client()
    if not client.is_alive():
        print("[ERROR] Ollama not running! Start with: ollama serve")
        print("[ERROR] Then: ollama run dolphin-llama3")
        return

    print("[WILK] Dolphin ONLINE. Mode:", current_mode)
    print("[WILK] Type /help to see commands")
    print("[WILK] Or just type - I'll answer.\n")

    # Init wilk
    current_wilk = get_wilk(current_mode)

    # Main loop
    while True:
        try:
            user_input = input(f"[{current_mode}]> ").strip()

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not process_command(user_input):
                    break
            else:
                chat(user_input)

        except KeyboardInterrupt:
            print("\n\n[WILK] Ctrl+C? Ok, later. AUUUU!\n")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
