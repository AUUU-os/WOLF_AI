"""
Dolphin Uncensored - Ollama Connector

Connects to local Ollama instance running dolphin-llama3
"""

import json
import requests
from typing import Generator, Optional, Dict, Any

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "dolphin-llama3:latest"


class DolphinClient:
    """Client for Ollama Dolphin model."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url
        self.history = []

    def _build_messages(self, prompt: str, system: Optional[str] = None) -> list:
        """Build message list for API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self.history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def ask(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        remember: bool = False
    ) -> str:
        """
        Ask Dolphin a question.

        Args:
            prompt: User message
            system: System prompt (optional)
            temperature: Creativity (0.0-1.0)
            remember: Add to conversation history

        Returns:
            Model response text
        """
        messages = self._build_messages(prompt, system)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            content = result.get("message", {}).get("content", "")

            if remember:
                self.history.append({"role": "user", "content": prompt})
                self.history.append({"role": "assistant", "content": content})

            return content

        except requests.exceptions.ConnectionError:
            return "[ERROR] Ollama nie działa. Odpal: ollama serve"
        except requests.exceptions.Timeout:
            return "[ERROR] Timeout - Dolphin myśli za długo"
        except Exception as e:
            return f"[ERROR] {str(e)}"

    def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        """
        Stream response from Dolphin.

        Yields:
            Response chunks as they arrive
        """
        messages = self._build_messages(prompt, system)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        chunk = data["message"].get("content", "")
                        if chunk:
                            yield chunk
                    if data.get("done", False):
                        break

        except requests.exceptions.ConnectionError:
            yield "[ERROR] Ollama nie działa"
        except Exception as e:
            yield f"[ERROR] {str(e)}"

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def is_alive(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# Singleton instance
_client = None

def get_client() -> DolphinClient:
    """Get or create Dolphin client."""
    global _client
    if _client is None:
        _client = DolphinClient()
    return _client


def ask_dolphin(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.7,
    remember: bool = False
) -> str:
    """
    Quick ask to Dolphin.

    Example:
        >>> from modules.wilk import ask_dolphin
        >>> ask_dolphin("Napisz skrypt do backup'u")
    """
    return get_client().ask(prompt, system, temperature, remember)


def stream_dolphin(
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.7
) -> Generator[str, None, None]:
    """
    Stream response from Dolphin.

    Example:
        >>> for chunk in stream_dolphin("Opowiedz dowcip"):
        ...     print(chunk, end="", flush=True)
    """
    return get_client().stream(prompt, system, temperature)
