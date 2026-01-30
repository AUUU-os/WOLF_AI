"""
WOLF_AI Telegram Bot - Control your pack from your phone

Setup:
1. Talk to @BotFather on Telegram
2. Create new bot: /newbot
3. Copy the token
4. Set TELEGRAM_BOT_TOKEN in .env
5. Set TELEGRAM_ALLOWED_USERS=your_telegram_id

Commands:
/start - Welcome message
/status - Pack status
/howl <message> - Send howl to pack
/hunt <target> - Start a hunt
/wilk <message> - Ask WILK
/mode <mode> - Change WILK mode
/sync - Sync with GitHub
/help - Show help
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_ALLOWED_USERS,
    BRIDGE_PATH,
    WOLF_ROOT
)

# Telegram imports
try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("[!] python-telegram-bot not installed. Run: pip install python-telegram-bot")


class WolfBot:
    """Telegram bot for WOLF_AI control."""

    def __init__(self, token: str):
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot required")

        self.token = token
        self.allowed_users = [u.strip() for u in TELEGRAM_ALLOWED_USERS if u.strip()]
        self.wilk_mode = "chat"
        self.app = None

    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        if not self.allowed_users:
            return True  # No restrictions if not configured
        return str(user_id) in self.allowed_users

    def _howl_to_bridge(self, from_wolf: str, message: str, frequency: str = "medium"):
        """Write howl to bridge."""
        howl_data = {
            "from": from_wolf,
            "to": "pack",
            "howl": message,
            "frequency": frequency,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        howls_file = BRIDGE_PATH / "howls.jsonl"
        howls_file.parent.mkdir(parents=True, exist_ok=True)

        with open(howls_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(howl_data, ensure_ascii=False) + "\n")

        return howl_data

    def _get_pack_state(self) -> dict:
        """Get pack state."""
        state_file = BRIDGE_PATH / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"pack_status": "unknown", "wolves": {}}

    def _get_recent_howls(self, limit: int = 5) -> list:
        """Get recent howls."""
        howls_file = BRIDGE_PATH / "howls.jsonl"
        if not howls_file.exists():
            return []

        with open(howls_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        howls = []
        for line in lines[-limit:]:
            try:
                howls.append(json.loads(line.strip()))
            except:
                continue
        return howls

    # ==================== HANDLERS ====================

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        user_id = user.id

        if not self._is_authorized(user_id):
            await update.message.reply_text(
                f"Access denied. Your ID: {user_id}\n"
                "Add this ID to TELEGRAM_ALLOWED_USERS in .env"
            )
            return

        welcome = """
🐺 *WOLF_AI Command Center*

Welcome to the pack, Commander!

*Commands:*
/status - Pack status
/howl <msg> - Send howl
/hunt <target> - Start hunt
/wilk <msg> - Ask WILK AI
/mode <mode> - WILK mode (chat/hacker/hustler/bro/guardian)
/howls - Recent howls
/sync - GitHub sync
/help - This message

AUUUUUUUUUUUUUUUUUU!
        """
        await update.message.reply_text(welcome, parse_mode="Markdown")

        # Log connection
        self._howl_to_bridge(
            "telegram",
            f"Commander connected via Telegram (ID: {user_id})",
            "medium"
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Access denied.")
            return

        state = self._get_pack_state()

        # Format status
        wolves_str = ""
        for name, info in state.get("wolves", {}).items():
            emoji = "🟢" if info.get("status") == "active" else "🔴"
            wolves_str += f"\n  {emoji} {name}: {info.get('status', 'unknown')}"

        status_msg = f"""
🐺 *Pack Status*

Status: *{state.get('pack_status', 'unknown').upper()}*
Last update: {state.get('last_updated', 'N/A')[:16]}

*Wolves:*{wolves_str}

WILK Mode: {self.wilk_mode}
        """
        await update.message.reply_text(status_msg, parse_mode="Markdown")

    async def howl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /howl command."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not context.args:
            await update.message.reply_text("Usage: /howl <message>")
            return

        message = " ".join(context.args)
        howl = self._howl_to_bridge("commander", message, "high")

        await update.message.reply_text(f"🐺 Howl sent!\n\n\"{message}\"")

    async def hunt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /hunt command."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not context.args:
            await update.message.reply_text("Usage: /hunt <target>")
            return

        target = " ".join(context.args)

        try:
            from core.pack import get_pack
            pack = get_pack()
            if pack.status != "active":
                pack.form().awaken()
            pack.hunt(target)

            await update.message.reply_text(f"🎯 Hunt started!\n\nTarget: {target}\nAssigned to: hunter")
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")

    async def wilk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /wilk command - ask WILK AI."""
        if not self._is_authorized(update.effective_user.id):
            return

        if not context.args:
            await update.message.reply_text("Usage: /wilk <question>")
            return

        question = " ".join(context.args)
        await update.message.reply_text("🐺 WILK is thinking...")

        try:
            from modules.wilk import get_wilk
            wilk = get_wilk(self.wilk_mode)
            response = wilk.ask(question)

            # Split long messages
            if len(response) > 4000:
                for i in range(0, len(response), 4000):
                    await update.message.reply_text(response[i:i+4000])
            else:
                await update.message.reply_text(response)

        except Exception as e:
            await update.message.reply_text(f"WILK error: {str(e)}")

    async def mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mode command - change WILK mode."""
        if not self._is_authorized(update.effective_user.id):
            return

        modes = ["chat", "hacker", "hustler", "bro", "guardian"]

        if not context.args:
            await update.message.reply_text(
                f"Current mode: {self.wilk_mode}\n\n"
                f"Available: {', '.join(modes)}\n\n"
                "Usage: /mode <mode>"
            )
            return

        new_mode = context.args[0].lower()
        if new_mode in modes:
            self.wilk_mode = new_mode
            await update.message.reply_text(f"🔄 WILK mode changed to: {new_mode}")
        else:
            await update.message.reply_text(f"Unknown mode. Available: {', '.join(modes)}")

    async def howls(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /howls command - show recent howls."""
        if not self._is_authorized(update.effective_user.id):
            return

        recent = self._get_recent_howls(10)

        if not recent:
            await update.message.reply_text("No howls yet.")
            return

        msg = "🐺 *Recent Howls:*\n\n"
        for h in reversed(recent):
            time = h.get("timestamp", "")[:16].replace("T", " ")
            msg += f"[{time}] *{h.get('from')}*: {h.get('howl')[:100]}\n\n"

        await update.message.reply_text(msg[:4000], parse_mode="Markdown")

    async def sync(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sync command - GitHub sync."""
        if not self._is_authorized(update.effective_user.id):
            return

        await update.message.reply_text("🔄 Syncing with GitHub...")

        import subprocess
        try:
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(WOLF_ROOT),
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout or result.stderr or "Up to date"
            await update.message.reply_text(f"✅ Sync complete:\n\n{output[:1000]}")
            self._howl_to_bridge("telegram", f"GitHub sync: {output[:100]}", "medium")
        except Exception as e:
            await update.message.reply_text(f"❌ Sync failed: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages - forward to WILK."""
        if not self._is_authorized(update.effective_user.id):
            return

        # Treat any message as WILK question
        question = update.message.text
        await update.message.reply_text("🐺 WILK thinking...")

        try:
            from modules.wilk import get_wilk
            wilk = get_wilk(self.wilk_mode)
            response = wilk.ask(question)

            if len(response) > 4000:
                for i in range(0, len(response), 4000):
                    await update.message.reply_text(response[i:i+4000])
            else:
                await update.message.reply_text(response)

        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")

    def run(self):
        """Start the bot."""
        if not self.token:
            print("[!] TELEGRAM_BOT_TOKEN not set in .env")
            return

        print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🐺 WOLF_AI Telegram Bot                                 ║
    ║   ═══════════════════════════════════════                 ║
    ║                                                           ║
    ║   Bot is running. Send /start to your bot!                ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
        """)

        self.app = Application.builder().token(self.token).build()

        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.start))
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("howl", self.howl))
        self.app.add_handler(CommandHandler("hunt", self.hunt))
        self.app.add_handler(CommandHandler("wilk", self.wilk))
        self.app.add_handler(CommandHandler("mode", self.mode))
        self.app.add_handler(CommandHandler("howls", self.howls))
        self.app.add_handler(CommandHandler("sync", self.sync))

        # Handle regular messages
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_message
        ))

        # Run
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


def run_bot():
    """Run the Telegram bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("[!] Set TELEGRAM_BOT_TOKEN in .env file")
        print("    1. Talk to @BotFather on Telegram")
        print("    2. Create bot: /newbot")
        print("    3. Copy token to .env: TELEGRAM_BOT_TOKEN=your_token")
        return

    bot = WolfBot(TELEGRAM_BOT_TOKEN)
    bot.run()


if __name__ == "__main__":
    run_bot()
