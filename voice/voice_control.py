"""
WOLF_AI Voice Control - "Hey WOLF!" Voice Assistant

Control your wolf pack with voice commands.
Works on:
- Desktop (Windows/Mac/Linux) with microphone
- Android (Termux) via termux-speech-to-text
- Any device with browser (Web Speech API)

Commands:
- "Hey Wolf, status" - Get pack status
- "Hey Wolf, awaken" - Wake the pack
- "Hey Wolf, hunt [task]" - Assign a task
- "Hey Wolf, howl [message]" - Send message
- "Hey Wolf, ask [question]" - Ask WILK
"""

import os
import sys
import json
import asyncio
import threading
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import WOLF_ROOT, BRIDGE_PATH
except ImportError:
    WOLF_ROOT = Path.home() / "WOLF_AI"
    BRIDGE_PATH = WOLF_ROOT / "bridge"

# Try imports for different platforms
SPEECH_RECOGNITION_AVAILABLE = False
PYTTSX3_AVAILABLE = False
TERMUX_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    pass

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pass

# Check for Termux
if os.path.exists("/data/data/com.termux"):
    TERMUX_AVAILABLE = True


class VoicePlatform(Enum):
    DESKTOP = "desktop"
    TERMUX = "termux"
    WEB = "web"


@dataclass
class VoiceCommand:
    """Parsed voice command."""
    raw_text: str
    command: str
    arguments: str
    confidence: float = 1.0

    @classmethod
    def parse(cls, text: str) -> Optional["VoiceCommand"]:
        """Parse voice input into command."""
        text = text.lower().strip()

        # Check for wake word
        wake_words = ["hey wolf", "ok wolf", "wolf", "wilk", "hej wilk"]
        found_wake = False

        for wake in wake_words:
            if text.startswith(wake):
                text = text[len(wake):].strip()
                text = text.lstrip(",").strip()
                found_wake = True
                break

        if not found_wake and len(text.split()) > 3:
            # No wake word and long text - probably not a command
            return None

        # Parse command
        words = text.split()
        if not words:
            return None

        command = words[0]
        arguments = " ".join(words[1:]) if len(words) > 1 else ""

        # Normalize commands
        command_map = {
            "status": "status",
            "stan": "status",
            "awaken": "awaken",
            "wake": "awaken",
            "obudź": "awaken",
            "wstań": "awaken",
            "hunt": "hunt",
            "poluj": "hunt",
            "zadanie": "hunt",
            "howl": "howl",
            "wyj": "howl",
            "powiedz": "howl",
            "ask": "ask",
            "zapytaj": "ask",
            "pytanie": "ask",
            "help": "help",
            "pomoc": "help",
            "stop": "stop",
            "quit": "stop",
            "koniec": "stop",
        }

        command = command_map.get(command, command)

        return cls(
            raw_text=text,
            command=command,
            arguments=arguments
        )


class TextToSpeech:
    """Cross-platform text-to-speech."""

    def __init__(self, platform: VoicePlatform = None):
        self.platform = platform or self._detect_platform()
        self.engine = None

        if self.platform == VoicePlatform.DESKTOP and PYTTSX3_AVAILABLE:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 175)

    def _detect_platform(self) -> VoicePlatform:
        if TERMUX_AVAILABLE:
            return VoicePlatform.TERMUX
        return VoicePlatform.DESKTOP

    def speak(self, text: str) -> bool:
        """Speak text aloud."""
        print(f"🐺 WOLF: {text}")

        if self.platform == VoicePlatform.TERMUX:
            return self._speak_termux(text)
        elif self.platform == VoicePlatform.DESKTOP and self.engine:
            return self._speak_desktop(text)
        else:
            # Fallback - just print
            return True

    def _speak_desktop(self, text: str) -> bool:
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"[!] TTS error: {e}")
            return False

    def _speak_termux(self, text: str) -> bool:
        import subprocess
        try:
            subprocess.run(
                ["termux-tts-speak", text],
                timeout=30
            )
            return True
        except Exception as e:
            print(f"[!] Termux TTS error: {e}")
            return False


class SpeechToText:
    """Cross-platform speech recognition."""

    def __init__(self, platform: VoicePlatform = None):
        self.platform = platform or self._detect_platform()
        self.recognizer = None

        if self.platform == VoicePlatform.DESKTOP and SPEECH_RECOGNITION_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()

            # Adjust for ambient noise
            with self.microphone as source:
                print("[*] Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def _detect_platform(self) -> VoicePlatform:
        if TERMUX_AVAILABLE:
            return VoicePlatform.TERMUX
        return VoicePlatform.DESKTOP

    def listen(self, timeout: int = 5) -> Optional[str]:
        """Listen for speech and return text."""
        if self.platform == VoicePlatform.TERMUX:
            return self._listen_termux(timeout)
        elif self.platform == VoicePlatform.DESKTOP and self.recognizer:
            return self._listen_desktop(timeout)
        else:
            # Fallback - text input
            return input("🎤 You: ").strip()

    def _listen_desktop(self, timeout: int) -> Optional[str]:
        try:
            with self.microphone as source:
                print("🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=timeout)

            print("🔄 Processing...")
            text = self.recognizer.recognize_google(audio)
            print(f"🎤 You said: {text}")
            return text

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("[!] Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"[!] Recognition error: {e}")
            return None

    def _listen_termux(self, timeout: int) -> Optional[str]:
        import subprocess
        try:
            result = subprocess.run(
                ["termux-speech-to-text"],
                capture_output=True,
                text=True,
                timeout=timeout + 5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    return data.get("text")
                return result.stdout.strip()
            return None
        except Exception as e:
            print(f"[!] Termux STT error: {e}")
            return None


class WolfVoiceAssistant:
    """
    Voice-controlled wolf pack assistant.

    "Hey Wolf, hunt build an API!"
    """

    def __init__(self,
                 wolf_api_url: str = "http://localhost:8000",
                 wolf_api_key: str = ""):
        self.api_url = wolf_api_url
        self.api_key = wolf_api_key
        self.tts = TextToSpeech()
        self.stt = SpeechToText()
        self.running = False
        self.commands: Dict[str, Callable] = {}
        self._register_commands()

    def _register_commands(self):
        """Register voice commands."""
        self.commands = {
            "status": self._cmd_status,
            "awaken": self._cmd_awaken,
            "hunt": self._cmd_hunt,
            "howl": self._cmd_howl,
            "ask": self._cmd_ask,
            "help": self._cmd_help,
            "stop": self._cmd_stop,
        }

    async def _call_api(self, method: str, endpoint: str,
                        data: Optional[Dict] = None) -> Optional[Dict]:
        """Call WOLF_AI API."""
        try:
            import httpx
            url = f"{self.api_url}{endpoint}"
            headers = {"X-API-Key": self.api_key}

            async with httpx.AsyncClient(timeout=30) as client:
                if method == "GET":
                    response = await client.get(url, headers=headers)
                else:
                    response = await client.post(url, headers=headers, json=data)
                return response.json()
        except Exception as e:
            print(f"[!] API error: {e}")
            return None

    # =========================================================================
    # COMMANDS
    # =========================================================================

    async def _cmd_status(self, args: str) -> str:
        result = await self._call_api("GET", "/api/status")
        if result and "pack" in result:
            pack = result["pack"]
            status = pack.get("pack_status", "unknown")
            wolves = pack.get("wolves", {})
            active = sum(1 for w in wolves.values() if w.get("status") == "active")
            return f"Pack status: {status}. {active} of {len(wolves)} wolves active."
        return "Could not get pack status."

    async def _cmd_awaken(self, args: str) -> str:
        result = await self._call_api("POST", "/api/awaken")
        if result:
            return "The pack has awakened! AUUUUUUU!"
        return "Could not awaken the pack."

    async def _cmd_hunt(self, args: str) -> str:
        if not args:
            return "What should I hunt? Tell me the task."

        result = await self._call_api("POST", "/api/hunt", {
            "target": args,
            "assigned_to": "hunter"
        })
        if result:
            return f"Hunt started! Target: {args}. The pack is on it!"
        return "Could not start the hunt."

    async def _cmd_howl(self, args: str) -> str:
        if not args:
            return "What message should I howl?"

        result = await self._call_api("POST", "/api/howl", {
            "message": args,
            "frequency": "high"
        })
        if result:
            return f"Howl sent: {args}"
        return "Could not send howl."

    async def _cmd_ask(self, args: str) -> str:
        if not args:
            return "What would you like to ask WILK?"

        result = await self._call_api("POST", "/api/wilk", {
            "message": args,
            "mode": "chat"
        })
        if result and "response" in result:
            # Truncate long responses for speech
            response = result["response"]
            if len(response) > 200:
                response = response[:200] + "... I have more details if you want."
            return response
        return "WILK did not respond. Is Ollama running?"

    async def _cmd_help(self, args: str) -> str:
        return """Available commands:
        Status - check pack status.
        Awaken - wake the pack.
        Hunt followed by task - assign a task.
        Howl followed by message - send a message.
        Ask followed by question - ask WILK AI.
        Stop - exit voice control."""

    async def _cmd_stop(self, args: str) -> str:
        self.running = False
        return "Goodbye! The pack rests. AUUUUUU!"

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def process_command(self, voice_cmd: VoiceCommand) -> str:
        """Process a voice command."""
        handler = self.commands.get(voice_cmd.command)

        if handler:
            return await handler(voice_cmd.arguments)
        else:
            return f"I don't understand the command: {voice_cmd.command}. Say help for available commands."

    async def run(self):
        """Run the voice assistant loop."""
        self.running = True

        self.tts.speak("Wolf voice control activated. Say Hey Wolf followed by a command.")

        while self.running:
            try:
                # Listen for command
                text = self.stt.listen(timeout=10)

                if not text:
                    continue

                # Parse command
                cmd = VoiceCommand.parse(text)

                if not cmd:
                    # Not a recognized command pattern
                    continue

                # Process command
                response = await self.process_command(cmd)

                # Speak response
                self.tts.speak(response)

            except KeyboardInterrupt:
                self.tts.speak("Voice control stopped.")
                break
            except Exception as e:
                print(f"[!] Error: {e}")
                self.tts.speak("An error occurred.")

    def run_sync(self):
        """Run synchronously (for non-async contexts)."""
        asyncio.run(self.run())


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run voice assistant."""
    import argparse

    parser = argparse.ArgumentParser(description="WOLF_AI Voice Control")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="WOLF_AI API URL")
    parser.add_argument("--api-key", default=os.getenv("WOLF_API_KEY", ""),
                        help="WOLF_AI API Key")
    parser.add_argument("--text-mode", action="store_true",
                        help="Use text input instead of voice")
    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   🐺 WOLF_AI Voice Control                                                ║
║   ═══════════════════════════════════════                                 ║
║                                                                           ║
║   Say "Hey Wolf" followed by a command:                                   ║
║   - "Hey Wolf, status"                                                    ║
║   - "Hey Wolf, awaken"                                                    ║
║   - "Hey Wolf, hunt [task]"                                               ║
║   - "Hey Wolf, ask [question]"                                            ║
║   - "Hey Wolf, stop"                                                      ║
║                                                                           ║
║   Press Ctrl+C to exit                                                    ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Check dependencies
    if not SPEECH_RECOGNITION_AVAILABLE and not TERMUX_AVAILABLE:
        print("[!] Speech recognition not available.")
        print("    Install: pip install SpeechRecognition pyaudio")
        print("    Or use --text-mode for text input")

        if not args.text_mode:
            args.text_mode = True

    assistant = WolfVoiceAssistant(
        wolf_api_url=args.api_url,
        wolf_api_key=args.api_key
    )

    assistant.run_sync()


if __name__ == "__main__":
    main()
