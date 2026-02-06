"""
ChatGPT Extension for WOLF_AI

Extends ChatGPT with wolf pack capabilities through function calling.
"""

from .wolf_gpt import WolfGPT, WolfToolRunner, WOLF_TOOLS, WOLF_SYSTEM_PROMPT

__all__ = ["WolfGPT", "WolfToolRunner", "WOLF_TOOLS", "WOLF_SYSTEM_PROMPT"]
