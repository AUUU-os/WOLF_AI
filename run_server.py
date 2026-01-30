#!/usr/bin/env python3
"""
WOLF_AI Server Launcher

Starts the FastAPI server with optional auto-tunnel.
"""
import os
import sys
import argparse
import subprocess
import threading
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_api_server():
    """Run the API server."""
    from api.server import run
    run()


def run_tunnel(port: int):
    """Run tunnel in background."""
    from scripts.tunnel import check_ngrok, check_cloudflared, start_ngrok, start_cloudflared

    if check_ngrok():
        start_ngrok(port)
    elif check_cloudflared():
        start_cloudflared(port)
    else:
        print("[!] No tunnel tool available")


def run_telegram_bot():
    """Run Telegram bot."""
    from telegram.bot import run_bot
    run_bot()


def main():
    parser = argparse.ArgumentParser(description="WOLF_AI Server")
    parser.add_argument("--tunnel", "-t", action="store_true", help="Start with public tunnel")
    parser.add_argument("--telegram", action="store_true", help="Start Telegram bot")
    parser.add_argument("--all", "-a", action="store_true", help="Start everything")
    parser.add_argument("--port", "-p", type=int, default=8000, help="API port (default: 8000)")

    args = parser.parse_args()

    os.environ["WOLF_API_PORT"] = str(args.port)

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🐺 WOLF_AI Command Center                               ║
    ║   ═══════════════════════════════════════                 ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    threads = []

    # Start tunnel if requested
    if args.tunnel or args.all:
        print("[*] Starting tunnel...")
        t = threading.Thread(target=run_tunnel, args=(args.port,), daemon=True)
        t.start()
        threads.append(t)
        import time
        time.sleep(5)  # Wait for tunnel to start

    # Start Telegram bot if requested
    if args.telegram or args.all:
        print("[*] Starting Telegram bot...")
        t = threading.Thread(target=run_telegram_bot, daemon=True)
        t.start()
        threads.append(t)

    # Start API server (blocking)
    print(f"[*] Starting API server on port {args.port}...")
    run_api_server()


if __name__ == "__main__":
    main()
