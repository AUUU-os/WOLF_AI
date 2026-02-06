"""
Termux Bridge - Android integration for WOLF_AI

Connects WOLF_AI to Android device capabilities via Termux:API
- Notifications
- SMS
- Battery status
- Location
- Sensors
- Clipboard
- Speech (TTS/STT)

Requirements:
- Termux:API app installed
- termux-api package: pkg install termux-api
"""

import subprocess
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import BRIDGE_PATH
except ImportError:
    BRIDGE_PATH = Path.home() / "WOLF_AI" / "bridge"


class TermuxAPI:
    """Interface to Termux:API commands."""

    @staticmethod
    def _run(cmd: List[str], input_data: Optional[str] = None) -> Optional[str]:
        """Run termux-api command."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                input=input_data,
                timeout=30
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception as e:
            print(f"[!] Termux API error: {e}")
            return None

    @staticmethod
    def _run_json(cmd: List[str]) -> Optional[Dict]:
        """Run command and parse JSON output."""
        result = TermuxAPI._run(cmd)
        if result:
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return None
        return None


class TermuxNotification(TermuxAPI):
    """Android notifications."""

    @staticmethod
    def send(title: str, content: str,
             notification_id: str = "wolf_ai",
             priority: str = "high",
             vibrate: bool = True,
             led_color: str = "blue") -> bool:
        """Send Android notification."""
        cmd = [
            "termux-notification",
            "--id", notification_id,
            "--title", title,
            "--content", content,
            "--priority", priority,
            "--led-color", led_color
        ]
        if vibrate:
            cmd.extend(["--vibrate", "200,100,200"])

        return TermuxAPI._run(cmd) is not None

    @staticmethod
    def remove(notification_id: str = "wolf_ai") -> bool:
        """Remove notification."""
        return TermuxAPI._run(["termux-notification-remove", notification_id]) is not None


class TermuxSMS(TermuxAPI):
    """SMS functionality."""

    @staticmethod
    def list_inbox(limit: int = 10) -> Optional[List[Dict]]:
        """List SMS inbox."""
        return TermuxAPI._run_json(["termux-sms-list", "-l", str(limit)])

    @staticmethod
    def send(number: str, message: str) -> bool:
        """Send SMS."""
        return TermuxAPI._run(["termux-sms-send", "-n", number, message]) is not None


class TermuxDevice(TermuxAPI):
    """Device information and control."""

    @staticmethod
    def battery() -> Optional[Dict]:
        """Get battery status."""
        return TermuxAPI._run_json(["termux-battery-status"])

    @staticmethod
    def wifi() -> Optional[Dict]:
        """Get WiFi info."""
        return TermuxAPI._run_json(["termux-wifi-connectioninfo"])

    @staticmethod
    def location(provider: str = "gps") -> Optional[Dict]:
        """Get device location."""
        return TermuxAPI._run_json(["termux-location", "-p", provider])

    @staticmethod
    def vibrate(duration_ms: int = 500) -> bool:
        """Vibrate device."""
        return TermuxAPI._run(["termux-vibrate", "-d", str(duration_ms)]) is not None

    @staticmethod
    def torch(enable: bool = True) -> bool:
        """Control flashlight."""
        return TermuxAPI._run(["termux-torch", "on" if enable else "off"]) is not None

    @staticmethod
    def brightness(level: int = 255) -> bool:
        """Set screen brightness (0-255)."""
        return TermuxAPI._run(["termux-brightness", str(level)]) is not None

    @staticmethod
    def volume(stream: str = "music", volume: int = 10) -> bool:
        """Set volume."""
        return TermuxAPI._run(["termux-volume", stream, str(volume)]) is not None


class TermuxClipboard(TermuxAPI):
    """Clipboard operations."""

    @staticmethod
    def get() -> Optional[str]:
        """Get clipboard content."""
        return TermuxAPI._run(["termux-clipboard-get"])

    @staticmethod
    def set(text: str) -> bool:
        """Set clipboard content."""
        return TermuxAPI._run(["termux-clipboard-set", text]) is not None


class TermuxSpeech(TermuxAPI):
    """Text-to-speech and speech recognition."""

    @staticmethod
    def speak(text: str, rate: float = 1.0) -> bool:
        """Text to speech."""
        cmd = ["termux-tts-speak", "-r", str(rate)]
        return TermuxAPI._run(cmd, input_data=text) is not None

    @staticmethod
    def listen(timeout: int = 10) -> Optional[str]:
        """Speech to text."""
        result = TermuxAPI._run_json(["termux-speech-to-text"])
        if result and isinstance(result, dict):
            return result.get("text")
        return result


class TermuxSensors(TermuxAPI):
    """Device sensors."""

    @staticmethod
    def list_sensors() -> Optional[List[Dict]]:
        """List available sensors."""
        return TermuxAPI._run_json(["termux-sensor", "-l"])

    @staticmethod
    def read(sensor: str, num_readings: int = 1) -> Optional[Dict]:
        """Read sensor data."""
        return TermuxAPI._run_json([
            "termux-sensor", "-s", sensor, "-n", str(num_readings)
        ])


class TermuxMedia(TermuxAPI):
    """Media controls."""

    @staticmethod
    def play_audio(file_path: str) -> bool:
        """Play audio file."""
        return TermuxAPI._run(["termux-media-player", "play", file_path]) is not None

    @staticmethod
    def take_photo(camera: str = "0", output: str = "photo.jpg") -> bool:
        """Take photo with camera."""
        return TermuxAPI._run([
            "termux-camera-photo", "-c", camera, output
        ]) is not None

    @staticmethod
    def record_mic(output: str = "recording.m4a",
                   limit_secs: int = 10) -> bool:
        """Record from microphone."""
        return TermuxAPI._run([
            "termux-microphone-record", "-f", output, "-l", str(limit_secs)
        ]) is not None


# =============================================================================
# WOLF_AI Integration
# =============================================================================

class TermuxWolf:
    """
    Wolf with Termux powers - Android integration for the pack.

    This wolf can:
    - Send notifications to the device
    - Read/send SMS
    - Get device status (battery, location, wifi)
    - Speak and listen
    - Control device (vibrate, torch, volume)
    """

    def __init__(self):
        self.name = "termux"
        self.role = "android"
        self.status = "dormant"
        self.notification = TermuxNotification()
        self.sms = TermuxSMS()
        self.device = TermuxDevice()
        self.clipboard = TermuxClipboard()
        self.speech = TermuxSpeech()
        self.sensors = TermuxSensors()
        self.media = TermuxMedia()

    def awaken(self) -> "TermuxWolf":
        """Awaken the Termux wolf."""
        self.status = "active"
        self.notification.send(
            "🐺 WOLF_AI",
            "Pack awakened! The forest is alive.",
            priority="high"
        )
        self._howl("Termux wolf awakened on Android device")
        return self

    def alert(self, message: str, vibrate: bool = True) -> bool:
        """Send alert notification."""
        if vibrate:
            self.device.vibrate(300)
        return self.notification.send("🐺 WOLF Alert", message)

    def speak(self, message: str) -> bool:
        """Speak message via TTS."""
        return self.speech.speak(message)

    def listen(self) -> Optional[str]:
        """Listen for voice command."""
        return self.speech.listen()

    def get_status(self) -> Dict[str, Any]:
        """Get full device status."""
        return {
            "battery": self.device.battery(),
            "wifi": self.device.wifi(),
            "location": self.device.location()
        }

    def howl_notification(self, message: str, frequency: str = "medium") -> bool:
        """Send howl as notification."""
        priority_map = {
            "low": "low",
            "medium": "default",
            "high": "high",
            "AUUUU": "max"
        }
        return self.notification.send(
            f"🐺 Howl [{frequency}]",
            message,
            priority=priority_map.get(frequency, "default"),
            vibrate=(frequency in ["high", "AUUUU"])
        )

    def _howl(self, message: str):
        """Log to bridge."""
        howl_data = {
            "from": "termux",
            "to": "pack",
            "howl": message,
            "frequency": "low",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        howls_file = BRIDGE_PATH / "howls.jsonl"
        try:
            howls_file.parent.mkdir(parents=True, exist_ok=True)
            with open(howls_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(howl_data, ensure_ascii=False) + "\n")
        except:
            pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "device": self.get_status()
        }


# =============================================================================
# Convenience functions
# =============================================================================

_termux_wolf: Optional[TermuxWolf] = None


def get_termux() -> TermuxWolf:
    """Get Termux wolf instance."""
    global _termux_wolf
    if _termux_wolf is None:
        _termux_wolf = TermuxWolf()
    return _termux_wolf


def notify(title: str, message: str) -> bool:
    """Quick notification."""
    return TermuxNotification.send(title, message)


def speak(text: str) -> bool:
    """Quick TTS."""
    return TermuxSpeech.speak(text)


def battery() -> Optional[Dict]:
    """Quick battery check."""
    return TermuxDevice.battery()


def vibrate(ms: int = 500) -> bool:
    """Quick vibrate."""
    return TermuxDevice.vibrate(ms)


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    print("🐺 Testing Termux Bridge...")

    # Test notification
    print("\n[1] Sending notification...")
    if notify("🐺 WOLF_AI Test", "Termux bridge is working!"):
        print("    ✓ Notification sent")
    else:
        print("    ✗ Notification failed (is Termux:API installed?)")

    # Test battery
    print("\n[2] Checking battery...")
    bat = battery()
    if bat:
        print(f"    ✓ Battery: {bat.get('percentage', '?')}% ({bat.get('status', '?')})")
    else:
        print("    ✗ Battery check failed")

    # Test vibrate
    print("\n[3] Vibrating...")
    if vibrate(200):
        print("    ✓ Vibration triggered")
    else:
        print("    ✗ Vibration failed")

    print("\n🐺 Termux Bridge test complete!")
