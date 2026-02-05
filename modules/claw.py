"""
Claw Module - OpenClaw Bridge for WOLF_AI

Connects WOLF_AI pack to OpenClaw gateway for extended capabilities.
The Claw acts as a sixth wolf - bridging two AI ecosystems.

Usage:
    from modules.claw import get_claw, ask_claw

    claw = get_claw()
    response = await claw.send("Hello from WOLF_AI!")
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BRIDGE_PATH

# Try to import websockets
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


@dataclass
class ClawMessage:
    """Message to/from OpenClaw."""
    type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClawMessage":
        return cls(
            type=data.get("type", "unknown"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", "")
        )


class ClawBridge:
    """Bridge to OpenClaw Gateway."""

    def __init__(self, gateway_url: str = "ws://127.0.0.1:18789"):
        self.gateway_url = gateway_url
        self.connected = False
        self.websocket = None
        self.message_handlers: List[Callable] = []
        self._message_queue: List[ClawMessage] = []

    async def connect(self) -> bool:
        """Connect to OpenClaw Gateway."""
        if not WEBSOCKETS_AVAILABLE:
            print("[!] websockets not installed. Run: pip install websockets")
            return False

        try:
            self.websocket = await websockets.connect(self.gateway_url)
            self.connected = True
            self._log_howl("Claw connected to OpenClaw Gateway")
            return True
        except Exception as e:
            print(f"[!] Failed to connect to OpenClaw: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Disconnect from OpenClaw Gateway."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            self._log_howl("Claw disconnected from OpenClaw")

    async def send(self, message: str, msg_type: str = "message",
                   metadata: Optional[Dict] = None) -> Optional[str]:
        """Send message to OpenClaw."""
        if not self.connected:
            if not await self.connect():
                return None

        try:
            claw_msg = ClawMessage(
                type=msg_type,
                content=message,
                metadata=metadata or {"source": "wolf_ai"}
            )

            await self.websocket.send(claw_msg.to_json())
            self._log_howl(f"Claw sent: {message[:50]}...")

            # Wait for response
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=30.0
            )

            return response

        except asyncio.TimeoutError:
            print("[!] OpenClaw response timeout")
            return None
        except Exception as e:
            print(f"[!] Error sending to OpenClaw: {e}")
            self.connected = False
            return None

    async def ask(self, question: str) -> Optional[str]:
        """Ask OpenClaw a question and get response."""
        return await self.send(question, msg_type="question")

    async def execute(self, command: str) -> Optional[str]:
        """Ask OpenClaw to execute a command."""
        return await self.send(command, msg_type="execute", metadata={
            "source": "wolf_ai",
            "requires_approval": True
        })

    async def listen(self, callback: Callable[[ClawMessage], None]):
        """Listen for messages from OpenClaw."""
        if not self.connected:
            if not await self.connect():
                return

        self.message_handlers.append(callback)

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    claw_msg = ClawMessage.from_dict(data)
                    for handler in self.message_handlers:
                        await handler(claw_msg) if asyncio.iscoroutinefunction(handler) else handler(claw_msg)
                except json.JSONDecodeError:
                    # Plain text message
                    claw_msg = ClawMessage(type="text", content=message)
                    for handler in self.message_handlers:
                        await handler(claw_msg) if asyncio.iscoroutinefunction(handler) else handler(claw_msg)
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            self._log_howl("OpenClaw connection closed")

    def _log_howl(self, message: str):
        """Log to WOLF_AI bridge."""
        howl_data = {
            "from": "claw",
            "to": "pack",
            "howl": message,
            "frequency": "low",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        howls_file = BRIDGE_PATH / "howls.jsonl"
        try:
            with open(howls_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(howl_data, ensure_ascii=False) + "\n")
        except:
            pass

    def on_message(self, callback: Callable):
        """Decorator to register message handler."""
        self.message_handlers.append(callback)
        return callback


class Claw:
    """
    The Claw - Sixth wolf in the pack, bridges to OpenClaw.

    Like a wolf with lobster claws - best of both worlds!
    """

    def __init__(self, gateway_url: str = "ws://127.0.0.1:18789"):
        self.name = "claw"
        self.role = "bridge"
        self.status = "dormant"
        self.bridge = ClawBridge(gateway_url)
        self.model = "openclaw"

    async def awaken(self) -> "Claw":
        """Awaken the Claw and connect to OpenClaw."""
        success = await self.bridge.connect()
        self.status = "active" if success else "disconnected"
        return self

    async def rest(self):
        """Disconnect and rest."""
        await self.bridge.disconnect()
        self.status = "resting"

    async def howl(self, message: str) -> Optional[str]:
        """Send message through OpenClaw."""
        return await self.bridge.send(message)

    async def ask(self, question: str) -> Optional[str]:
        """Ask OpenClaw."""
        return await self.bridge.ask(question)

    async def execute(self, command: str) -> Optional[str]:
        """Execute via OpenClaw (sandboxed)."""
        return await self.bridge.execute(command)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "model": self.model,
            "connected": self.bridge.connected,
            "gateway": self.bridge.gateway_url
        }


# =============================================================================
# SINGLETON & CONVENIENCE
# =============================================================================

_claw: Optional[Claw] = None


def get_claw(gateway_url: str = "ws://127.0.0.1:18789") -> Claw:
    """Get or create Claw instance."""
    global _claw
    if _claw is None:
        _claw = Claw(gateway_url)
    return _claw


async def ask_claw(question: str) -> Optional[str]:
    """Quick function to ask OpenClaw."""
    claw = get_claw()
    if claw.status != "active":
        await claw.awaken()
    return await claw.ask(question)


async def claw_execute(command: str) -> Optional[str]:
    """Quick function to execute via OpenClaw."""
    claw = get_claw()
    if claw.status != "active":
        await claw.awaken()
    return await claw.execute(command)


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    async def main():
        claw = get_claw()
        print("Connecting to OpenClaw...")

        await claw.awaken()

        if claw.status == "active":
            print("Connected! Testing...")

            # Test message
            response = await claw.howl("Hello from WOLF_AI! The pack sends greetings.")
            print(f"Response: {response}")

            await claw.rest()
        else:
            print("Could not connect to OpenClaw.")
            print("Make sure OpenClaw is running: openclaw gateway status")

    asyncio.run(main())
