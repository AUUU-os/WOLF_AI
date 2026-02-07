"""
Knowledge Base - Structured knowledge storage for WOLF_AI

Supports:
- Fact storage and retrieval
- Topic organization
- Knowledge graphs (simple)
- Search and query
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MEMORY_PATH


@dataclass
class Fact:
    """A single fact/piece of knowledge."""
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source: Optional[str] = None
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "confidence": self.confidence,
            "source": self.source,
            "created": self.created,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        return cls(**data)

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


@dataclass
class Topic:
    """A knowledge topic/category."""
    name: str
    description: str = ""
    parent: Optional[str] = None
    facts: List[str] = field(default_factory=list)  # Fact IDs
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parent": self.parent,
            "facts": self.facts,
            "created": self.created
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Topic":
        return cls(**data)


class KnowledgeBase:
    """Knowledge base for structured information."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.filepath = MEMORY_PATH / f"knowledge_{name}.json"
        self._ensure_file()

    def _ensure_file(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self._save({"facts": {}, "topics": {}, "relations": []})

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"facts": {}, "topics": {}, "relations": []}

    def _save(self, data: Dict[str, Any]):
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # FACTS
    # =========================================================================

    def add_fact(self, subject: str, predicate: str, obj: str,
                 confidence: float = 1.0, source: Optional[str] = None,
                 tags: List[str] = None) -> Fact:
        """Add a fact to the knowledge base."""
        data = self._load()
        facts = data.get("facts", {})

        # Generate ID
        fact_id = f"fact_{len(facts) + 1:05d}"

        fact = Fact(
            id=fact_id,
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source=source,
            tags=tags or []
        )

        facts[fact_id] = fact.to_dict()
        data["facts"] = facts
        self._save(data)

        return fact

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID."""
        data = self._load()
        fact_data = data.get("facts", {}).get(fact_id)
        return Fact.from_dict(fact_data) if fact_data else None

    def query_facts(self, subject: Optional[str] = None,
                    predicate: Optional[str] = None,
                    obj: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    min_confidence: float = 0.0) -> List[Fact]:
        """Query facts with filters."""
        data = self._load()
        facts = []

        for fact_data in data.get("facts", {}).values():
            fact = Fact.from_dict(fact_data)

            if subject and fact.subject.lower() != subject.lower():
                continue
            if predicate and fact.predicate.lower() != predicate.lower():
                continue
            if obj and fact.object.lower() != obj.lower():
                continue
            if fact.confidence < min_confidence:
                continue
            if tags and not any(t in fact.tags for t in tags):
                continue

            facts.append(fact)

        return facts

    def search_facts(self, query: str, limit: int = 20) -> List[Fact]:
        """Search facts by text content."""
        data = self._load()
        query_lower = query.lower()
        results = []

        for fact_data in data.get("facts", {}).values():
            fact = Fact.from_dict(fact_data)
            text = f"{fact.subject} {fact.predicate} {fact.object}".lower()

            if query_lower in text:
                results.append(fact)

        return results[:limit]

    def get_facts_about(self, subject: str) -> List[Fact]:
        """Get all facts about a subject."""
        return self.query_facts(subject=subject)

    # =========================================================================
    # TOPICS
    # =========================================================================

    def add_topic(self, name: str, description: str = "",
                  parent: Optional[str] = None) -> Topic:
        """Add a topic to the knowledge base."""
        data = self._load()
        topics = data.get("topics", {})

        topic = Topic(name=name, description=description, parent=parent)
        topics[name] = topic.to_dict()
        data["topics"] = topics
        self._save(data)

        return topic

    def get_topic(self, name: str) -> Optional[Topic]:
        """Get a topic by name."""
        data = self._load()
        topic_data = data.get("topics", {}).get(name)
        return Topic.from_dict(topic_data) if topic_data else None

    def add_fact_to_topic(self, topic_name: str, fact_id: str):
        """Associate a fact with a topic."""
        data = self._load()
        topics = data.get("topics", {})

        if topic_name in topics:
            if fact_id not in topics[topic_name].get("facts", []):
                topics[topic_name]["facts"].append(fact_id)
                data["topics"] = topics
                self._save(data)

    def get_topic_facts(self, topic_name: str) -> List[Fact]:
        """Get all facts in a topic."""
        topic = self.get_topic(topic_name)
        if not topic:
            return []

        facts = []
        for fact_id in topic.facts:
            fact = self.get_fact(fact_id)
            if fact:
                facts.append(fact)

        return facts

    def list_topics(self) -> List[Topic]:
        """List all topics."""
        data = self._load()
        return [Topic.from_dict(t) for t in data.get("topics", {}).values()]

    # =========================================================================
    # RELATIONS (Simple Knowledge Graph)
    # =========================================================================

    def add_relation(self, from_entity: str, relation: str, to_entity: str):
        """Add a relation between entities."""
        data = self._load()
        relations = data.get("relations", [])

        relation_entry = {
            "from": from_entity,
            "relation": relation,
            "to": to_entity,
            "created": datetime.utcnow().isoformat() + "Z"
        }

        # Avoid duplicates
        for r in relations:
            if r["from"] == from_entity and r["relation"] == relation and r["to"] == to_entity:
                return

        relations.append(relation_entry)
        data["relations"] = relations
        self._save(data)

    def get_relations(self, entity: str,
                      direction: str = "both") -> List[Dict[str, str]]:
        """Get relations for an entity."""
        data = self._load()
        relations = data.get("relations", [])

        results = []
        for r in relations:
            if direction in ["from", "both"] and r["from"] == entity:
                results.append(r)
            elif direction in ["to", "both"] and r["to"] == entity:
                results.append(r)

        return results

    def find_path(self, from_entity: str, to_entity: str,
                  max_depth: int = 3) -> Optional[List[Dict]]:
        """Find a path between two entities (simple BFS)."""
        data = self._load()
        relations = data.get("relations", [])

        # Build adjacency
        adj: Dict[str, List[Dict]] = {}
        for r in relations:
            if r["from"] not in adj:
                adj[r["from"]] = []
            adj[r["from"]].append(r)

        # BFS
        visited: Set[str] = {from_entity}
        queue = [(from_entity, [])]

        while queue:
            current, path = queue.pop(0)

            if current == to_entity:
                return path

            if len(path) >= max_depth:
                continue

            for rel in adj.get(current, []):
                next_entity = rel["to"]
                if next_entity not in visited:
                    visited.add(next_entity)
                    queue.append((next_entity, path + [rel]))

        return None

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        data = self._load()

        facts = data.get("facts", {})
        topics = data.get("topics", {})
        relations = data.get("relations", [])

        # Count by predicate
        predicates: Dict[str, int] = {}
        for f in facts.values():
            pred = f.get("predicate", "unknown")
            predicates[pred] = predicates.get(pred, 0) + 1

        return {
            "total_facts": len(facts),
            "total_topics": len(topics),
            "total_relations": len(relations),
            "predicates": predicates,
            "average_confidence": sum(f.get("confidence", 1) for f in facts.values()) / max(len(facts), 1)
        }


# =============================================================================
# CONVENIENCE
# =============================================================================

_knowledge_bases: Dict[str, KnowledgeBase] = {}


def get_knowledge(name: str = "default") -> KnowledgeBase:
    """Get or create a knowledge base."""
    if name not in _knowledge_bases:
        _knowledge_bases[name] = KnowledgeBase(name)
    return _knowledge_bases[name]


def know(subject: str, predicate: str, obj: str,
         kb_name: str = "default") -> Dict[str, Any]:
    """Quick function to add knowledge."""
    kb = get_knowledge(kb_name)
    fact = kb.add_fact(subject, predicate, obj)
    return fact.to_dict()


def ask(subject: str, kb_name: str = "default") -> List[Dict[str, Any]]:
    """Quick function to query about a subject."""
    kb = get_knowledge(kb_name)
    facts = kb.get_facts_about(subject)
    return [f.to_dict() for f in facts]
