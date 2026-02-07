"""
Notification Hub - Multi-channel alerts

Send notifications through multiple channels:
- Telegram
- Discord
- SMS (via Termux)
- Desktop (native notifications)
- Email

Usage:
    from modules.notify import notify, get_notifier

    # Quick notify
    await notify("Hunt completed!", channels=["telegram", "desktop"])

    # With notifier instance
    notifier = get_notifier()
    await notifier.send("Alert!", channel="discord", priority="high")
"""

import os
import json
import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIDGE_PATH


class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Channel(Enum):
    TELEGRAM = "telegram"
    DISCORD = "discord"
    SMS = "sms"
    DESKTOP = "desktop"
    EMAIL = "email"
    TERMUX = "termux"
    HOWL = "howl"  # Internal pack notification


@dataclass
class Notification:
    """A notification to send."""
    message: str
    title: str = "🐺 WOLF_AI"
    priority: Priority = Priority.MEDIUM
    channel: Channel = Channel.HOWL
    data: Dict[str, Any] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"
        if self.data is None:
            self.data = {}


# =============================================================================
# CHANNEL HANDLERS
# =============================================================================

class ChannelHandler(ABC):
    """Abstract base for notification channels."""

    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        pass


class TelegramHandler(ChannelHandler):
    """Telegram bot notifications."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id)

    async def send(self, notification: Notification) -> bool:
        if not self.is_configured():
            return False

        try:
            import httpx

            # Priority emoji
            priority_emoji = {
                Priority.LOW: "📢",
                Priority.MEDIUM: "📣",
                Priority.HIGH: "🚨",
                Priority.CRITICAL: "🔥"
            }

            emoji = priority_emoji.get(notification.priority, "📢")
            text = f"{emoji} *{notification.title}*\n\n{notification.message}"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "Markdown"
                    }
                )
                return response.status_code == 200

        except Exception as e:
            print(f"[Telegram] Error: {e}")
            return False


class DiscordHandler(ChannelHandler):
    """Discord webhook notifications."""

    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    async def send(self, notification: Notification) -> bool:
        if not self.is_configured():
            return False

        try:
            import httpx

            # Color based on priority
            colors = {
                Priority.LOW: 0x3498db,     # Blue
                Priority.MEDIUM: 0xf1c40f,  # Yellow
                Priority.HIGH: 0xe74c3c,    # Red
                Priority.CRITICAL: 0x9b59b6  # Purple
            }

            embed = {
                "title": notification.title,
                "description": notification.message,
                "color": colors.get(notification.priority, 0x3498db),
                "timestamp": notification.timestamp,
                "footer": {"text": "WOLF_AI Pack"}
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json={"embeds": [embed]}
                )
                return response.status_code in [200, 204]

        except Exception as e:
            print(f"[Discord] Error: {e}")
            return False


class TermuxHandler(ChannelHandler):
    """Termux notifications (Android)."""

    def is_configured(self) -> bool:
        # Check if termux-notification is available
        try:
            result = subprocess.run(
                ["which", "termux-notification"],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False

    async def send(self, notification: Notification) -> bool:
        if not self.is_configured():
            return False

        try:
            # Priority to importance
            importance = {
                Priority.LOW: "low",
                Priority.MEDIUM: "default",
                Priority.HIGH: "high",
                Priority.CRITICAL: "max"
            }

            cmd = [
                "termux-notification",
                "--title", notification.title,
                "--content", notification.message[:256],
                "--priority", importance.get(notification.priority, "default")
            ]

            if notification.priority == Priority.CRITICAL:
                cmd.extend(["--vibrate", "500,200,500"])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            return process.returncode == 0

        except Exception as e:
            print(f"[Termux] Error: {e}")
            return False


class SMSHandler(ChannelHandler):
    """SMS via Termux."""

    def __init__(self):
        self.phone = os.getenv("WOLF_SMS_PHONE")

    def is_configured(self) -> bool:
        if not self.phone:
            return False
        try:
            result = subprocess.run(
                ["which", "termux-sms-send"],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False

    async def send(self, notification: Notification) -> bool:
        if not self.is_configured():
            return False

        try:
            text = f"{notification.title}: {notification.message}"[:160]

            process = await asyncio.create_subprocess_exec(
                "termux-sms-send",
                "-n", self.phone,
                text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            return process.returncode == 0

        except Exception as e:
            print(f"[SMS] Error: {e}")
            return False


class DesktopHandler(ChannelHandler):
    """Desktop notifications (cross-platform)."""

    def is_configured(self) -> bool:
        return True  # Always try

    async def send(self, notification: Notification) -> bool:
        try:
            import platform
            system = platform.system()

            if system == "Darwin":  # macOS
                script = f'''
                display notification "{notification.message}" with title "{notification.title}"
                '''
                process = await asyncio.create_subprocess_exec(
                    "osascript", "-e", script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()
                return process.returncode == 0

            elif system == "Linux":
                process = await asyncio.create_subprocess_exec(
                    "notify-send",
                    notification.title,
                    notification.message,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()
                return process.returncode == 0

            elif system == "Windows":
                # PowerShell toast
                ps_script = f'''
                [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                $template = [Windows.UI.Notifications.ToastTemplateType]::ToastText02
                $xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($template)
                $text = $xml.GetElementsByTagName("text")
                $text[0].AppendChild($xml.CreateTextNode("{notification.title}")) | Out-Null
                $text[1].AppendChild($xml.CreateTextNode("{notification.message}")) | Out-Null
                $toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
                [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("WOLF_AI").Show($toast)
                '''
                process = await asyncio.create_subprocess_exec(
                    "powershell", "-Command", ps_script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.wait()
                return process.returncode == 0

        except Exception as e:
            print(f"[Desktop] Error: {e}")
            return False

        return False


class EmailHandler(ChannelHandler):
    """Email notifications via SMTP."""

    def __init__(self):
        self.smtp_host = os.getenv("WOLF_SMTP_HOST")
        self.smtp_port = int(os.getenv("WOLF_SMTP_PORT", "587"))
        self.smtp_user = os.getenv("WOLF_SMTP_USER")
        self.smtp_pass = os.getenv("WOLF_SMTP_PASS")
        self.to_email = os.getenv("WOLF_EMAIL_TO")

    def is_configured(self) -> bool:
        return all([
            self.smtp_host,
            self.smtp_user,
            self.smtp_pass,
            self.to_email
        ])

    async def send(self, notification: Notification) -> bool:
        if not self.is_configured():
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = self.to_email
            msg["Subject"] = f"[WOLF_AI] {notification.title}"

            body = f"""
{notification.message}

---
Priority: {notification.priority.value}
Time: {notification.timestamp}
Sent by WOLF_AI Pack
            """

            msg.attach(MIMEText(body, "plain"))

            def send_mail():
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_pass)
                    server.send_message(msg)

            await asyncio.to_thread(send_mail)
            return True

        except Exception as e:
            print(f"[Email] Error: {e}")
            return False


class HowlHandler(ChannelHandler):
    """Internal pack notification (howl)."""

    def is_configured(self) -> bool:
        return True

    async def send(self, notification: Notification) -> bool:
        try:
            from modules.howl import get_bridge

            # Map priority to frequency
            frequency_map = {
                Priority.LOW: "low",
                Priority.MEDIUM: "medium",
                Priority.HIGH: "high",
                Priority.CRITICAL: "AUUUU"
            }

            bridge = get_bridge()
            bridge.howl(
                message=f"[{notification.title}] {notification.message}",
                to="pack",
                frequency=frequency_map.get(notification.priority, "medium")
            )
            return True

        except Exception as e:
            print(f"[Howl] Error: {e}")
            return False


# =============================================================================
# NOTIFICATION HUB
# =============================================================================

class NotificationHub:
    """
    Central notification hub.

    Manages multiple channels and routes notifications.
    """

    def __init__(self):
        self.handlers: Dict[Channel, ChannelHandler] = {
            Channel.TELEGRAM: TelegramHandler(),
            Channel.DISCORD: DiscordHandler(),
            Channel.TERMUX: TermuxHandler(),
            Channel.SMS: SMSHandler(),
            Channel.DESKTOP: DesktopHandler(),
            Channel.EMAIL: EmailHandler(),
            Channel.HOWL: HowlHandler(),
        }

        self._notification_log: List[Dict] = []
        self._default_channels = [Channel.HOWL]

        # Auto-detect available channels
        self._detect_defaults()

    def _detect_defaults(self) -> None:
        """Detect and set default channels."""
        for channel, handler in self.handlers.items():
            if handler.is_configured():
                if channel not in self._default_channels:
                    if channel in [Channel.TELEGRAM, Channel.TERMUX, Channel.DESKTOP]:
                        self._default_channels.append(channel)

    def get_available_channels(self) -> List[str]:
        """Get list of configured channels."""
        return [
            ch.value for ch, handler in self.handlers.items()
            if handler.is_configured()
        ]

    def set_default_channels(self, channels: List[str]) -> None:
        """Set default notification channels."""
        self._default_channels = [
            Channel(ch) for ch in channels
            if ch in [c.value for c in Channel]
        ]

    async def send(
        self,
        message: str,
        title: str = "🐺 WOLF_AI",
        priority: str = "medium",
        channels: List[str] = None,
        data: Dict[str, Any] = None
    ) -> Dict[str, bool]:
        """
        Send notification to specified channels.

        Args:
            message: Notification message
            title: Notification title
            priority: low, medium, high, critical
            channels: Channels to send to (uses defaults if not specified)
            data: Additional data

        Returns:
            Dict mapping channel names to success status
        """
        notification = Notification(
            message=message,
            title=title,
            priority=Priority(priority),
            data=data or {}
        )

        # Determine channels
        if channels:
            target_channels = [Channel(ch) for ch in channels if ch in [c.value for c in Channel]]
        else:
            target_channels = self._default_channels

        # Send to each channel
        results = {}
        for channel in target_channels:
            handler = self.handlers.get(channel)
            if handler and handler.is_configured():
                try:
                    success = await handler.send(notification)
                    results[channel.value] = success
                except Exception as e:
                    print(f"[NotificationHub] {channel.value} error: {e}")
                    results[channel.value] = False
            else:
                results[channel.value] = False

        # Log notification
        self._log_notification(notification, results)

        return results

    async def send_priority(
        self,
        message: str,
        priority: str = "high"
    ) -> Dict[str, bool]:
        """Send high-priority notification to all available channels."""
        all_channels = self.get_available_channels()
        return await self.send(
            message=message,
            priority=priority,
            channels=all_channels
        )

    async def alert(self, message: str) -> Dict[str, bool]:
        """Send critical alert to all channels."""
        return await self.send_priority(message, "critical")

    def _log_notification(
        self,
        notification: Notification,
        results: Dict[str, bool]
    ) -> None:
        """Log notification to file."""
        log_file = BRIDGE_PATH / "notifications.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "title": notification.title,
            "message": notification.message[:200],
            "priority": notification.priority.value,
            "channels": results,
            "timestamp": notification.timestamp
        }

        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._notification_log.append(entry)

    def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get recent notifications."""
        return self._notification_log[-limit:]


# =============================================================================
# SINGLETON & CONVENIENCE
# =============================================================================

_notifier: Optional[NotificationHub] = None


def get_notifier() -> NotificationHub:
    """Get or create notifier singleton."""
    global _notifier
    if _notifier is None:
        _notifier = NotificationHub()
    return _notifier


async def notify(
    message: str,
    title: str = "🐺 WOLF_AI",
    priority: str = "medium",
    channels: List[str] = None
) -> Dict[str, bool]:
    """Quick notify function."""
    notifier = get_notifier()
    return await notifier.send(message, title, priority, channels)


async def alert(message: str) -> Dict[str, bool]:
    """Quick alert function (critical priority, all channels)."""
    notifier = get_notifier()
    return await notifier.alert(message)


# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="WOLF_AI Notification Hub")
    parser.add_argument("action", choices=["send", "alert", "channels", "test"])
    parser.add_argument("--message", "-m", help="Message to send")
    parser.add_argument("--title", "-t", default="🐺 WOLF_AI")
    parser.add_argument("--priority", "-p", default="medium",
                       choices=["low", "medium", "high", "critical"])
    parser.add_argument("--channels", "-c", nargs="+", help="Channels")
    args = parser.parse_args()

    notifier = get_notifier()

    if args.action == "channels":
        print("\n📡 Available Channels:\n")
        for ch in notifier.get_available_channels():
            print(f"  ✓ {ch}")
        print(f"\n  Defaults: {[c.value for c in notifier._default_channels]}\n")

    elif args.action == "test":
        print("\n🧪 Testing all channels...\n")
        results = await notifier.send_priority("Test notification from WOLF_AI", "medium")
        for channel, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {channel}")
        print()

    elif args.action == "send":
        if not args.message:
            print("Error: --message required")
            return

        results = await notify(
            message=args.message,
            title=args.title,
            priority=args.priority,
            channels=args.channels
        )

        for channel, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {channel}")

    elif args.action == "alert":
        if not args.message:
            print("Error: --message required")
            return

        results = await alert(args.message)
        for channel, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {channel}")


if __name__ == "__main__":
    asyncio.run(main())
