"""
WOLF_AI Memory System

Persistent storage for wolf knowledge and experiences.
Supports multiple backends: JSON, SQLite, ChromaDB (vector).
"""

from .store import MemoryStore, get_store
from .knowledge import KnowledgeBase, get_knowledge

__all__ = [
    "MemoryStore",
    "get_store",
    "KnowledgeBase",
    "get_knowledge"
]
