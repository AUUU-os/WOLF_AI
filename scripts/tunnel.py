"""
WOLF_AI Tunnel - Expose your pack to the internet

Uses ngrok or cloudflared to create a secure tunnel.
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_ngrok():
    """Check if ngrok is installed."""
    try:
        result = subprocess.run(["ngrok", "version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_cloudflared():
    """Check if cloudflared is installed."""
    try:
        result = subprocess.run(["cloudflared", "version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def start_ngrok(port: int = 8000):
    """Start ngrok tunnel."""
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║   Starting ngrok tunnel on port {port}                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Start ngrok in background
    process = subprocess.Popen(
        ["ngrok", "http", str(port), "--log=stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for tunnel to establish
    time.sleep(3)

    # Get public URL from ngrok API
    try:
        import requests
        response = requests.get("http://localhost:4040/api/tunnels")
        tunnels = response.json()
        public_url = tunnels["tunnels"][0]["public_url"]

        print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   TUNNEL ACTIVE                                           ║
    ║   ══════════════════════════════════════                  ║
    ║                                                           ║
    ║   Public URL: {public_url:<41} ║
    ║                                                           ║
    ║   Dashboard:  {public_url}/dashboard                      ║
    ║   API Docs:   {public_url}/docs                           ║
    ║                                                           ║
    ║   Use this URL on your phone!                             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
        """)

        # Save URL to file for reference
        url_file = PROJECT_ROOT / "bridge" / "tunnel_url.txt"
        with open(url_file, "w") as f:
            f.write(public_url)

        return process, public_url

    except Exception as e:
        print(f"[!] Could not get tunnel URL: {e}")
        print("[!] Check ngrok dashboard at http://localhost:4040")
        return process, None


def start_cloudflared(port: int = 8000):
    """Start cloudflared tunnel."""
    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║   Starting cloudflared tunnel on port {port}                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # cloudflared prints URL to stderr
    print("[*] Waiting for tunnel URL...")
    for line in process.stderr:
        if "trycloudflare.com" in line:
            # Extract URL
            import re
            match = re.search(r'https://[^\s]+\.trycloudflare\.com', line)
            if match:
                public_url = match.group(0)
                print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   TUNNEL ACTIVE                                           ║
    ║                                                           ║
    ║   Public URL: {public_url}
    ║                                                           ║
    ║   Use this URL on your phone!                             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
                """)

                # Save URL
                url_file = PROJECT_ROOT / "bridge" / "tunnel_url.txt"
                with open(url_file, "w") as f:
                    f.write(public_url)

                return process, public_url

    return process, None


def main():
    port = int(os.getenv("WOLF_API_PORT", "8000"))

    print("""
    🐺 WOLF_AI Tunnel Setup
    ═══════════════════════════════════════

    This will expose your WOLF_AI server to the internet,
    so you can control it from your phone anywhere.

    """)

    # Check available tools
    has_ngrok = check_ngrok()
    has_cloudflared = check_cloudflared()

    if not has_ngrok and not has_cloudflared:
        print("""
    [!] No tunnel tool found!

    Install one of these:

    NGROK (recommended):
    1. Go to https://ngrok.com/download
    2. Download and install
    3. Run: ngrok config add-authtoken <your-token>

    CLOUDFLARED (free, no account needed):
    1. Go to https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation
    2. Download and add to PATH
        """)
        return

    # Start tunnel
    if has_ngrok:
        print("[*] Using ngrok...")
        process, url = start_ngrok(port)
    else:
        print("[*] Using cloudflared...")
        process, url = start_cloudflared(port)

    if process:
        print("\n[*] Press Ctrl+C to stop the tunnel\n")
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n[*] Stopping tunnel...")
            process.terminate()


if __name__ == "__main__":
    main()
