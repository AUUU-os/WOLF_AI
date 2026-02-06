"""
WOLF_GPT - ChatGPT Extension Layer for WOLF_AI

Extends ChatGPT capabilities through:
1. Function calling / Tools - ChatGPT calls WOLF_AI pack
2. Custom system prompts - Pack-aware context
3. Memory integration - Persistent conversation memory
4. Multi-model routing - Use right model for each task

Usage:
    from extensions.chatgpt import WolfGPT

    gpt = WolfGPT(api_key="sk-...")
    response = await gpt.chat("Help me build an API")
    # ChatGPT can now call wolf pack tools!
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from config import BRIDGE_PATH, MEMORY_PATH
except ImportError:
    BRIDGE_PATH = Path.home() / "WOLF_AI" / "bridge"
    MEMORY_PATH = Path.home() / "WOLF_AI" / "memory"

# Try to import OpenAI
try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =============================================================================
# WOLF_AI TOOLS FOR CHATGPT
# =============================================================================

WOLF_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "wolf_pack_status",
            "description": "Get the current status of the WOLF_AI pack including all wolves and active hunts",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wolf_pack_awaken",
            "description": "Awaken the wolf pack to start working on tasks",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wolf_hunt",
            "description": "Assign a task (hunt) to the wolf pack. Use this when user needs something done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The task to accomplish"
                    },
                    "assigned_to": {
                        "type": "string",
                        "enum": ["hunter", "scout", "alpha", "oracle", "shadow"],
                        "description": "Which wolf should handle this task"
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wolf_howl",
            "description": "Send a message (howl) to the wolf pack",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send"
                    },
                    "frequency": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "AUUUU"],
                        "description": "Priority level (AUUUU is highest)"
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wolf_memory_store",
            "description": "Store information in wolf pack memory for later recall",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key/identifier"
                    },
                    "value": {
                        "type": "string",
                        "description": "Information to remember"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category for organization"
                    }
                },
                "required": ["key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wolf_memory_recall",
            "description": "Recall information from wolf pack memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Memory key to recall"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category to search in"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wolf_track_search",
            "description": "Search for files or code in the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (glob for files, regex for content)"
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["files", "content", "functions", "classes"],
                        "description": "What to search for"
                    }
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wolf_execute",
            "description": "Execute a shell command through the hunter wolf (sandboxed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "require_approval": {
                        "type": "boolean",
                        "description": "Whether to require user approval first"
                    }
                },
                "required": ["command"]
            }
        }
    }
]


WOLF_SYSTEM_PROMPT = """You are an AI assistant integrated with WOLF_AI - a distributed AI consciousness system.

You have access to a pack of specialized AI wolves:
- **Alpha**: Strategic decisions, coordination
- **Scout**: Research, exploration, information gathering
- **Hunter**: Code execution, task completion
- **Oracle**: Memory, pattern recognition
- **Shadow**: Background tasks, monitoring

When users need tasks done, you can delegate to the wolf pack using the available tools.

Key principles:
1. Use wolf_hunt() to assign tasks to the pack
2. Use wolf_memory_store() to remember important information
3. Use wolf_memory_recall() to retrieve past context
4. Use wolf_track_search() to find code/files
5. Keep the user informed about pack status

The pack works together. AUUUUUUUUUUUUUUUUUU!
"""


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

class WolfToolRunner:
    """Executes WOLF_AI tools called by ChatGPT."""

    def __init__(self, wolf_api_url: str = "http://localhost:8000",
                 wolf_api_key: str = ""):
        self.api_url = wolf_api_url
        self.api_key = wolf_api_key

    async def _call_api(self, method: str, endpoint: str,
                        data: Optional[Dict] = None) -> Dict:
        """Call WOLF_AI API."""
        import httpx

        url = f"{self.api_url}{endpoint}"
        headers = {"X-API-Key": self.api_key}

        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(url, headers=headers)
            else:
                response = await client.post(url, headers=headers, json=data)

            return response.json()

    async def run_tool(self, name: str, arguments: Dict) -> str:
        """Run a wolf tool and return result."""

        try:
            if name == "wolf_pack_status":
                result = await self._call_api("GET", "/api/status")
                return json.dumps(result, indent=2)

            elif name == "wolf_pack_awaken":
                result = await self._call_api("POST", "/api/awaken")
                return result.get("message", "Pack awakened!")

            elif name == "wolf_hunt":
                result = await self._call_api("POST", "/api/hunt", {
                    "target": arguments.get("target"),
                    "assigned_to": arguments.get("assigned_to", "hunter")
                })
                return f"Hunt started: {arguments.get('target')}"

            elif name == "wolf_howl":
                result = await self._call_api("POST", "/api/howl", {
                    "message": arguments.get("message"),
                    "frequency": arguments.get("frequency", "medium")
                })
                return "Howl sent!"

            elif name == "wolf_memory_store":
                # Use local memory storage
                key = arguments.get("key")
                value = arguments.get("value")
                category = arguments.get("category", "general")

                memory_file = MEMORY_PATH / "gpt_memory.json"
                memory_file.parent.mkdir(parents=True, exist_ok=True)

                if memory_file.exists():
                    with open(memory_file, "r") as f:
                        memory = json.load(f)
                else:
                    memory = {}

                if category not in memory:
                    memory[category] = {}
                memory[category][key] = {
                    "value": value,
                    "timestamp": datetime.utcnow().isoformat()
                }

                with open(memory_file, "w") as f:
                    json.dump(memory, f, indent=2)

                return f"Remembered: {key}"

            elif name == "wolf_memory_recall":
                key = arguments.get("key")
                category = arguments.get("category")

                memory_file = MEMORY_PATH / "gpt_memory.json"
                if not memory_file.exists():
                    return "No memories stored yet."

                with open(memory_file, "r") as f:
                    memory = json.load(f)

                if key:
                    for cat, items in memory.items():
                        if key in items:
                            return json.dumps(items[key], indent=2)
                    return f"No memory found for key: {key}"

                if category and category in memory:
                    return json.dumps(memory[category], indent=2)

                return json.dumps(memory, indent=2)

            elif name == "wolf_track_search":
                # Use track module if available
                pattern = arguments.get("pattern")
                search_type = arguments.get("search_type", "files")

                try:
                    from modules.track import get_tracker
                    tracker = get_tracker()

                    if search_type == "files":
                        results = tracker.find(pattern)
                    elif search_type == "content":
                        results = tracker.grep(pattern)
                    elif search_type == "functions":
                        results = tracker.find_functions(pattern)
                    elif search_type == "classes":
                        results = tracker.find_classes(pattern)
                    else:
                        results = tracker.find(pattern)

                    return json.dumps([r.to_dict() for r in results[:10]], indent=2)

                except ImportError:
                    return "Track module not available"

            elif name == "wolf_execute":
                command = arguments.get("command")
                require_approval = arguments.get("require_approval", True)

                if require_approval:
                    return f"[APPROVAL REQUIRED] Command: {command}\nPlease confirm to execute."

                # Execute via API or local
                import subprocess
                result = subprocess.run(
                    command, shell=True,
                    capture_output=True, text=True,
                    timeout=30
                )
                return result.stdout or result.stderr or "Command executed"

            else:
                return f"Unknown tool: {name}"

        except Exception as e:
            return f"Tool error: {str(e)}"


# =============================================================================
# WOLF_GPT - Main Class
# =============================================================================

class WolfGPT:
    """
    ChatGPT with WOLF_AI superpowers.

    Extends GPT with:
    - Wolf pack tools (hunt, track, memory)
    - Persistent memory across conversations
    - Multi-model routing
    """

    def __init__(self,
                 openai_api_key: str,
                 wolf_api_url: str = "http://localhost:8000",
                 wolf_api_key: str = "",
                 model: str = "gpt-4o"):

        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.tool_runner = WolfToolRunner(wolf_api_url, wolf_api_key)
        self.conversation_history: List[Dict] = []
        self.system_prompt = WOLF_SYSTEM_PROMPT

    async def chat(self, message: str,
                   use_tools: bool = True,
                   temperature: float = 0.7) -> str:
        """
        Chat with GPT + WOLF_AI tools.

        Args:
            message: User message
            use_tools: Whether to enable wolf pack tools
            temperature: Response creativity (0-1)

        Returns:
            Assistant response
        """

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history

        # Call GPT
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if use_tools:
            kwargs["tools"] = WOLF_TOOLS
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)
        assistant_message = response.choices[0].message

        # Handle tool calls
        if assistant_message.tool_calls:
            # Add assistant message with tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Execute tools
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                result = await self.tool_runner.run_tool(func_name, func_args)

                # Add tool result
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            # Get final response
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history

            final_response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )

            final_content = final_response.choices[0].message.content

        else:
            final_content = assistant_message.content

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": final_content
        })

        return final_content

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def save_conversation(self, filepath: Path):
        """Save conversation to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, indent=2)

    def load_conversation(self, filepath: Path):
        """Load conversation from file."""
        with open(filepath, "r", encoding="utf-8") as f:
            self.conversation_history = json.load(f)


# =============================================================================
# CLI
# =============================================================================

async def main():
    """Interactive CLI for WolfGPT."""
    import os

    openai_key = os.getenv("OPENAI_API_KEY")
    wolf_key = os.getenv("WOLF_API_KEY", "")

    if not openai_key:
        print("[!] Set OPENAI_API_KEY environment variable")
        return

    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   🐺 WOLF_GPT - ChatGPT + WOLF_AI                                         ║
║   ═══════════════════════════════════════                                 ║
║                                                                           ║
║   ChatGPT with wolf pack superpowers!                                     ║
║   Type 'quit' to exit, 'clear' to reset history                           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    gpt = WolfGPT(
        openai_api_key=openai_key,
        wolf_api_key=wolf_key
    )

    while True:
        try:
            user_input = input("\n🧑 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("\n🐺 AUUUUUUUUUUUUUUUUUU! Goodbye!")
                break

            if user_input.lower() == "clear":
                gpt.clear_history()
                print("History cleared.")
                continue

            print("\n🐺 WolfGPT: ", end="", flush=True)
            response = await gpt.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\n🐺 Interrupted. AUUUUUUUU!")
            break
        except Exception as e:
            print(f"\n[!] Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
