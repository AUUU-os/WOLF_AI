"""
WOLF_AI MCP - Model Context Protocol Server

Expose WOLF_AI capabilities to external AI tools.
"""

from .server import MCPServer, WOLF_TOOLS, WOLF_RESOURCES

__all__ = ["MCPServer", "WOLF_TOOLS", "WOLF_RESOURCES"]
