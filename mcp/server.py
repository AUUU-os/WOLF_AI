"""
WOLF_AI MCP Server - Model Context Protocol

Expose WOLF_AI capabilities as MCP tools for external AI integrations.
Compatible with Claude Desktop, VS Code extensions, and other MCP clients.

Usage:
    python -m mcp.server

Protocol: https://modelcontextprotocol.io
"""

import json
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Add WOLF_AI to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WOLF_ROOT, BRIDGE_PATH


# =============================================================================
# MCP PROTOCOL TYPES
# =============================================================================

@dataclass
class Tool:
    """MCP Tool definition."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class Resource:
    """MCP Resource definition."""
    uri: str
    name: str
    description: str
    mimeType: str = "application/json"


# =============================================================================
# WOLF TOOLS
# =============================================================================

WOLF_TOOLS = [
    Tool(
        name="wolf_status",
        description="Get current pack status including all wolves and active hunts",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    Tool(
        name="wolf_awaken",
        description="Awaken the wolf pack and prepare for hunting",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    Tool(
        name="wolf_howl",
        description="Send a message (howl) to the pack or specific wolf",
        inputSchema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to send"
                },
                "to": {
                    "type": "string",
                    "description": "Recipient: 'pack', 'alpha', 'scout', 'hunter', 'oracle', 'shadow'",
                    "default": "pack"
                },
                "frequency": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "AUUUU"],
                    "default": "medium"
                }
            },
            "required": ["message"]
        }
    ),
    Tool(
        name="wolf_hunt",
        description="Start a task (hunt) and assign it to a wolf",
        inputSchema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Task description"
                },
                "assigned_to": {
                    "type": "string",
                    "description": "Wolf to assign: 'hunter', 'scout', 'oracle', 'shadow'",
                    "default": "hunter"
                }
            },
            "required": ["target"]
        }
    ),
    Tool(
        name="wolf_think",
        description="Ask Alpha wolf to analyze a situation using Claude API",
        inputSchema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question or situation to analyze"
                },
                "context": {
                    "type": "string",
                    "description": "Additional context"
                }
            },
            "required": ["question"]
        }
    ),
    Tool(
        name="wolf_plan",
        description="Ask Alpha to create a strategic plan for an objective",
        inputSchema={
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "What needs to be accomplished"
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limitations or requirements"
                }
            },
            "required": ["objective"]
        }
    ),
    Tool(
        name="wolf_execute",
        description="Execute code safely in the Hunter sandbox",
        inputSchema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Code to execute"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "bash", "ruby", "go"],
                    "default": "python"
                }
            },
            "required": ["code"]
        }
    ),
    Tool(
        name="wolf_search",
        description="Search files and code in the territory",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern"
                },
                "type": {
                    "type": "string",
                    "enum": ["files", "content", "functions", "classes"],
                    "default": "files"
                }
            },
            "required": ["pattern"]
        }
    ),
    Tool(
        name="wolf_memory_store",
        description="Store information in pack memory",
        inputSchema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Memory key"
                },
                "value": {
                    "type": "string",
                    "description": "Value to store"
                },
                "category": {
                    "type": "string",
                    "default": "general"
                }
            },
            "required": ["key", "value"]
        }
    ),
    Tool(
        name="wolf_memory_recall",
        description="Recall information from pack memory",
        inputSchema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Memory key to recall"
                },
                "category": {
                    "type": "string"
                }
            },
            "required": ["key"]
        }
    )
]


WOLF_RESOURCES = [
    Resource(
        uri="wolf://pack/status",
        name="Pack Status",
        description="Current status of the wolf pack"
    ),
    Resource(
        uri="wolf://pack/howls",
        name="Recent Howls",
        description="Recent pack communications"
    ),
    Resource(
        uri="wolf://pack/tasks",
        name="Active Tasks",
        description="Currently active hunts/tasks"
    ),
    Resource(
        uri="wolf://alpha/thoughts",
        name="Alpha Thoughts",
        description="Recent Alpha brain activity"
    )
]


# =============================================================================
# TOOL HANDLERS
# =============================================================================

async def handle_wolf_status(arguments: Dict) -> Dict:
    """Handle wolf_status tool."""
    from core.pack import get_pack
    pack = get_pack()
    return pack.status_report()


async def handle_wolf_awaken(arguments: Dict) -> Dict:
    """Handle wolf_awaken tool."""
    from core.pack import awaken_pack
    pack = awaken_pack()
    return {"status": "awakened", "pack": pack.status_report()}


async def handle_wolf_howl(arguments: Dict) -> Dict:
    """Handle wolf_howl tool."""
    from modules.howl import get_bridge
    bridge = get_bridge()
    howl = bridge.howl(
        message=arguments["message"],
        to=arguments.get("to", "pack"),
        frequency=arguments.get("frequency", "medium")
    )
    return {"status": "sent", "howl": howl.to_dict()}


async def handle_wolf_hunt(arguments: Dict) -> Dict:
    """Handle wolf_hunt tool."""
    from core.pack import get_pack
    pack = get_pack()
    success = pack.hunt(
        arguments["target"],
        arguments.get("assigned_to", "hunter")
    )
    return {"status": "started" if success else "failed", "target": arguments["target"]}


async def handle_wolf_think(arguments: Dict) -> Dict:
    """Handle wolf_think tool."""
    from core.alpha import get_alpha_brain
    brain = get_alpha_brain()

    if not brain.is_available:
        return {"error": "Alpha brain offline. Set ANTHROPIC_API_KEY."}

    response = await brain.think(
        arguments["question"],
        arguments.get("context")
    )
    return {"response": response}


async def handle_wolf_plan(arguments: Dict) -> Dict:
    """Handle wolf_plan tool."""
    from core.alpha import get_alpha_brain
    brain = get_alpha_brain()

    if not brain.is_available:
        return {"error": "Alpha brain offline. Set ANTHROPIC_API_KEY."}

    plan = await brain.plan(
        arguments["objective"],
        arguments.get("constraints")
    )
    return {"plan": plan}


async def handle_wolf_execute(arguments: Dict) -> Dict:
    """Handle wolf_execute tool."""
    from modules.sandbox import execute_code
    result = await execute_code(
        arguments["code"],
        arguments.get("language", "python")
    )
    return result.to_dict()


async def handle_wolf_search(arguments: Dict) -> Dict:
    """Handle wolf_search tool."""
    from modules.track import get_tracker
    tracker = get_tracker()

    search_type = arguments.get("type", "files")
    pattern = arguments["pattern"]

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

    return {"results": [r.to_dict() for r in results[:20]]}


async def handle_wolf_memory_store(arguments: Dict) -> Dict:
    """Handle wolf_memory_store tool."""
    from memory.store import get_memory
    memory = get_memory()
    memory.set(
        arguments["key"],
        arguments["value"],
        namespace=arguments.get("category", "general")
    )
    return {"status": "stored", "key": arguments["key"]}


async def handle_wolf_memory_recall(arguments: Dict) -> Dict:
    """Handle wolf_memory_recall tool."""
    from memory.store import get_memory
    memory = get_memory()
    value = memory.get(
        arguments["key"],
        namespace=arguments.get("category", "general")
    )
    return {"key": arguments["key"], "value": value}


TOOL_HANDLERS = {
    "wolf_status": handle_wolf_status,
    "wolf_awaken": handle_wolf_awaken,
    "wolf_howl": handle_wolf_howl,
    "wolf_hunt": handle_wolf_hunt,
    "wolf_think": handle_wolf_think,
    "wolf_plan": handle_wolf_plan,
    "wolf_execute": handle_wolf_execute,
    "wolf_search": handle_wolf_search,
    "wolf_memory_store": handle_wolf_memory_store,
    "wolf_memory_recall": handle_wolf_memory_recall,
}


# =============================================================================
# RESOURCE HANDLERS
# =============================================================================

async def handle_resource(uri: str) -> Dict:
    """Handle resource request."""
    if uri == "wolf://pack/status":
        from core.pack import get_pack
        return get_pack().status_report()

    elif uri == "wolf://pack/howls":
        howls_file = BRIDGE_PATH / "howls.jsonl"
        if howls_file.exists():
            with open(howls_file) as f:
                lines = f.readlines()[-20:]
            return {"howls": [json.loads(l) for l in lines if l.strip()]}
        return {"howls": []}

    elif uri == "wolf://pack/tasks":
        tasks_file = BRIDGE_PATH / "tasks.json"
        if tasks_file.exists():
            with open(tasks_file) as f:
                return json.load(f)
        return {"hunts": []}

    elif uri == "wolf://alpha/thoughts":
        thoughts_file = BRIDGE_PATH / "alpha_thoughts.jsonl"
        if thoughts_file.exists():
            with open(thoughts_file) as f:
                lines = f.readlines()[-10:]
            return {"thoughts": [json.loads(l) for l in lines if l.strip()]}
        return {"thoughts": []}

    return {"error": f"Unknown resource: {uri}"}


# =============================================================================
# MCP SERVER
# =============================================================================

class MCPServer:
    """
    MCP Server implementation using stdio transport.

    Handles JSON-RPC messages over stdin/stdout.
    """

    def __init__(self):
        self.name = "wolf-ai"
        self.version = "0.4.0"

    async def handle_message(self, message: Dict) -> Dict:
        """Handle incoming MCP message."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        try:
            if method == "initialize":
                return self._response(msg_id, {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version
                    }
                })

            elif method == "tools/list":
                return self._response(msg_id, {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.inputSchema
                        }
                        for t in WOLF_TOOLS
                    ]
                })

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if tool_name not in TOOL_HANDLERS:
                    return self._error(msg_id, -32602, f"Unknown tool: {tool_name}")

                result = await TOOL_HANDLERS[tool_name](arguments)

                return self._response(msg_id, {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                })

            elif method == "resources/list":
                return self._response(msg_id, {
                    "resources": [
                        {
                            "uri": r.uri,
                            "name": r.name,
                            "description": r.description,
                            "mimeType": r.mimeType
                        }
                        for r in WOLF_RESOURCES
                    ]
                })

            elif method == "resources/read":
                uri = params.get("uri")
                result = await handle_resource(uri)

                return self._response(msg_id, {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                })

            elif method == "notifications/initialized":
                # Client acknowledged initialization
                return None

            else:
                return self._error(msg_id, -32601, f"Method not found: {method}")

        except Exception as e:
            return self._error(msg_id, -32603, str(e))

    def _response(self, msg_id: Any, result: Dict) -> Dict:
        """Create success response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result
        }

    def _error(self, msg_id: Any, code: int, message: str) -> Dict:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message
            }
        }

    async def run(self):
        """Run the MCP server on stdio."""
        print(json.dumps({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        }), file=sys.stderr)

        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )

                if not line:
                    break

                message = json.loads(line.strip())
                response = await self.handle_message(message)

                if response:
                    print(json.dumps(response), flush=True)

            except json.JSONDecodeError:
                continue
            except EOFError:
                break
            except Exception as e:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": str(e)}
                }), flush=True)


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Run MCP server."""
    server = MCPServer()

    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   🐺 WOLF_AI MCP Server v{server.version}                              ║
║   ════════════════════════════════                                ║
║                                                                   ║
║   Model Context Protocol server for AI tool integration          ║
║                                                                   ║
║   Tools: {len(WOLF_TOOLS)}                                                       ║
║   Resources: {len(WOLF_RESOURCES)}                                                    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""", file=sys.stderr)

    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
