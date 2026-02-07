"""
Evolve Module - Self-improvement system for WOLF_AI

Handles:
- Learning from hunt results
- Pattern recognition
- Strategy optimization
- Performance tracking
- Evolution metrics
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BRIDGE_PATH, MEMORY_PATH, ARENA_PATH


@dataclass
class Lesson:
    """A learned lesson from experience."""
    id: str
    category: str
    trigger: str  # What triggered this lesson
    insight: str  # What was learned
    confidence: float = 0.5  # 0.0 to 1.0
    applications: int = 0  # Times this lesson was applied
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_applied: Optional[str] = None
    source_hunt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "trigger": self.trigger,
            "insight": self.insight,
            "confidence": self.confidence,
            "applications": self.applications,
            "created": self.created,
            "last_applied": self.last_applied,
            "source_hunt": self.source_hunt
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Lesson":
        return cls(**data)


@dataclass
class Evolution:
    """Tracks evolution/improvement over time."""
    wolf_name: str
    metric: str
    value: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wolf_name": self.wolf_name,
            "metric": self.metric,
            "value": self.value,
            "timestamp": self.timestamp
        }


class EvolutionTracker:
    """Tracks and manages pack evolution."""

    def __init__(self):
        self.lessons_file = MEMORY_PATH / "lessons.json"
        self.metrics_file = MEMORY_PATH / "metrics.jsonl"
        self.patterns_file = MEMORY_PATH / "patterns.json"
        self._ensure_files()

    def _ensure_files(self):
        """Ensure evolution files exist."""
        MEMORY_PATH.mkdir(parents=True, exist_ok=True)

        if not self.lessons_file.exists():
            self._save_lessons([])

        if not self.metrics_file.exists():
            self.metrics_file.touch()

        if not self.patterns_file.exists():
            self._save_patterns({})

    # =========================================================================
    # LESSONS
    # =========================================================================

    def _load_lessons(self) -> List[Lesson]:
        """Load all lessons."""
        if not self.lessons_file.exists():
            return []

        with open(self.lessons_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [Lesson.from_dict(l) for l in data.get("lessons", [])]

    def _save_lessons(self, lessons: List[Lesson]):
        """Save lessons."""
        data = {
            "version": "1.0",
            "updated": datetime.utcnow().isoformat() + "Z",
            "lessons": [l.to_dict() for l in lessons]
        }
        with open(self.lessons_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def learn(self, category: str, trigger: str, insight: str,
              source_hunt: Optional[str] = None) -> Lesson:
        """Record a new lesson."""
        lessons = self._load_lessons()

        # Check for similar existing lesson
        for lesson in lessons:
            if lesson.trigger == trigger and lesson.category == category:
                # Reinforce existing lesson
                lesson.confidence = min(1.0, lesson.confidence + 0.1)
                lesson.applications += 1
                lesson.last_applied = datetime.utcnow().isoformat() + "Z"
                self._save_lessons(lessons)
                return lesson

        # Create new lesson
        lesson_id = f"lesson_{len(lessons) + 1:04d}"
        lesson = Lesson(
            id=lesson_id,
            category=category,
            trigger=trigger,
            insight=insight,
            source_hunt=source_hunt
        )
        lessons.append(lesson)
        self._save_lessons(lessons)

        return lesson

    def apply_lesson(self, lesson_id: str) -> Optional[Lesson]:
        """Mark a lesson as applied, increasing confidence."""
        lessons = self._load_lessons()

        for lesson in lessons:
            if lesson.id == lesson_id:
                lesson.applications += 1
                lesson.confidence = min(1.0, lesson.confidence + 0.05)
                lesson.last_applied = datetime.utcnow().isoformat() + "Z"
                self._save_lessons(lessons)
                return lesson

        return None

    def get_lessons(self, category: Optional[str] = None,
                    min_confidence: float = 0.0) -> List[Lesson]:
        """Get lessons, optionally filtered."""
        lessons = self._load_lessons()

        if category:
            lessons = [l for l in lessons if l.category == category]

        lessons = [l for l in lessons if l.confidence >= min_confidence]

        # Sort by confidence
        return sorted(lessons, key=lambda l: l.confidence, reverse=True)

    def find_relevant_lessons(self, context: str, limit: int = 5) -> List[Lesson]:
        """Find lessons relevant to a context."""
        lessons = self._load_lessons()
        context_lower = context.lower()

        # Simple relevance scoring
        scored = []
        for lesson in lessons:
            score = 0
            if any(word in lesson.trigger.lower() for word in context_lower.split()):
                score += 2
            if any(word in lesson.insight.lower() for word in context_lower.split()):
                score += 1
            score += lesson.confidence

            if score > 0:
                scored.append((lesson, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [l for l, _ in scored[:limit]]

    # =========================================================================
    # METRICS
    # =========================================================================

    def record_metric(self, wolf_name: str, metric: str, value: float):
        """Record an evolution metric."""
        evolution = Evolution(wolf_name=wolf_name, metric=metric, value=value)

        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(evolution.to_dict()) + "\n")

    def get_metrics(self, wolf_name: Optional[str] = None,
                    metric: Optional[str] = None,
                    hours: int = 24) -> List[Evolution]:
        """Get metrics with filters."""
        if not self.metrics_file.exists():
            return []

        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        metrics = []

        with open(self.metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("timestamp", "") >= since:
                        evo = Evolution(**data)
                        if wolf_name and evo.wolf_name != wolf_name:
                            continue
                        if metric and evo.metric != metric:
                            continue
                        metrics.append(evo)
                except:
                    continue

        return metrics

    def get_metric_trend(self, wolf_name: str, metric: str,
                         hours: int = 168) -> Dict[str, Any]:
        """Get trend for a specific metric."""
        metrics = self.get_metrics(wolf_name, metric, hours)

        if not metrics:
            return {"trend": "unknown", "change": 0, "samples": 0}

        values = [m.value for m in metrics]
        avg = sum(values) / len(values)

        # Simple trend detection
        if len(values) >= 2:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            change = second_avg - first_avg
            if change > 0.1:
                trend = "improving"
            elif change < -0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
            change = 0

        return {
            "trend": trend,
            "change": change,
            "current": values[-1] if values else 0,
            "average": avg,
            "samples": len(values)
        }

    # =========================================================================
    # PATTERNS
    # =========================================================================

    def _load_patterns(self) -> Dict[str, Any]:
        """Load recognized patterns."""
        if not self.patterns_file.exists():
            return {}

        with open(self.patterns_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_patterns(self, patterns: Dict[str, Any]):
        """Save patterns."""
        with open(self.patterns_file, "w", encoding="utf-8") as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)

    def recognize_pattern(self, name: str, description: str,
                          occurrences: int = 1) -> Dict[str, Any]:
        """Record or update a recognized pattern."""
        patterns = self._load_patterns()

        if name in patterns:
            patterns[name]["occurrences"] += occurrences
            patterns[name]["last_seen"] = datetime.utcnow().isoformat() + "Z"
        else:
            patterns[name] = {
                "description": description,
                "occurrences": occurrences,
                "first_seen": datetime.utcnow().isoformat() + "Z",
                "last_seen": datetime.utcnow().isoformat() + "Z"
            }

        self._save_patterns(patterns)
        return patterns[name]

    def get_patterns(self, min_occurrences: int = 1) -> Dict[str, Any]:
        """Get all recognized patterns."""
        patterns = self._load_patterns()
        return {
            k: v for k, v in patterns.items()
            if v.get("occurrences", 0) >= min_occurrences
        }

    # =========================================================================
    # EVOLUTION SUMMARY
    # =========================================================================

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get overall evolution summary."""
        lessons = self._load_lessons()
        patterns = self._load_patterns()

        # Calculate overall progress
        if lessons:
            avg_confidence = sum(l.confidence for l in lessons) / len(lessons)
            total_applications = sum(l.applications for l in lessons)
        else:
            avg_confidence = 0
            total_applications = 0

        return {
            "total_lessons": len(lessons),
            "average_confidence": avg_confidence,
            "total_applications": total_applications,
            "patterns_recognized": len(patterns),
            "top_categories": self._get_top_categories(lessons),
            "evolution_stage": self._calculate_stage(len(lessons), avg_confidence)
        }

    def _get_top_categories(self, lessons: List[Lesson]) -> Dict[str, int]:
        """Get top lesson categories."""
        categories = defaultdict(int)
        for lesson in lessons:
            categories[lesson.category] += 1
        return dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5])

    def _calculate_stage(self, lesson_count: int, avg_confidence: float) -> str:
        """Calculate evolution stage."""
        score = lesson_count * 0.1 + avg_confidence * 10

        if score < 2:
            return "pup"
        elif score < 5:
            return "yearling"
        elif score < 10:
            return "hunter"
        elif score < 20:
            return "veteran"
        else:
            return "alpha"


# =============================================================================
# ARENA - Training Ground
# =============================================================================

class Arena:
    """Training ground for wolves."""

    def __init__(self):
        self.challenges_file = ARENA_PATH / "challenges.json"
        self.results_file = ARENA_PATH / "results.jsonl"
        ARENA_PATH.mkdir(parents=True, exist_ok=True)
        self._ensure_files()

    def _ensure_files(self):
        if not self.challenges_file.exists():
            self._save_challenges([])

    def _save_challenges(self, challenges: List[Dict]):
        with open(self.challenges_file, "w", encoding="utf-8") as f:
            json.dump({"challenges": challenges}, f, indent=2)

    def _load_challenges(self) -> List[Dict]:
        if not self.challenges_file.exists():
            return []
        with open(self.challenges_file, "r", encoding="utf-8") as f:
            return json.load(f).get("challenges", [])

    def add_challenge(self, name: str, description: str,
                      difficulty: str = "medium",
                      test_cases: List[Dict] = None) -> Dict[str, Any]:
        """Add a training challenge."""
        challenges = self._load_challenges()

        challenge = {
            "id": f"challenge_{len(challenges) + 1:03d}",
            "name": name,
            "description": description,
            "difficulty": difficulty,
            "test_cases": test_cases or [],
            "created": datetime.utcnow().isoformat() + "Z"
        }

        challenges.append(challenge)
        self._save_challenges(challenges)
        return challenge

    def record_attempt(self, challenge_id: str, wolf_name: str,
                       success: bool, score: float,
                       notes: Optional[str] = None):
        """Record a challenge attempt."""
        result = {
            "challenge_id": challenge_id,
            "wolf_name": wolf_name,
            "success": success,
            "score": score,
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

    def get_leaderboard(self, challenge_id: Optional[str] = None) -> List[Dict]:
        """Get arena leaderboard."""
        if not self.results_file.exists():
            return []

        scores = defaultdict(lambda: {"attempts": 0, "successes": 0, "total_score": 0})

        with open(self.results_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    if challenge_id and result.get("challenge_id") != challenge_id:
                        continue
                    wolf = result["wolf_name"]
                    scores[wolf]["attempts"] += 1
                    if result["success"]:
                        scores[wolf]["successes"] += 1
                    scores[wolf]["total_score"] += result.get("score", 0)
                except:
                    continue

        leaderboard = []
        for wolf, data in scores.items():
            leaderboard.append({
                "wolf": wolf,
                "attempts": data["attempts"],
                "successes": data["successes"],
                "success_rate": data["successes"] / data["attempts"] if data["attempts"] > 0 else 0,
                "total_score": data["total_score"],
                "avg_score": data["total_score"] / data["attempts"] if data["attempts"] > 0 else 0
            })

        return sorted(leaderboard, key=lambda x: x["total_score"], reverse=True)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_tracker: Optional[EvolutionTracker] = None
_arena: Optional[Arena] = None


def get_tracker() -> EvolutionTracker:
    """Get or create evolution tracker."""
    global _tracker
    if _tracker is None:
        _tracker = EvolutionTracker()
    return _tracker


def get_arena() -> Arena:
    """Get or create arena."""
    global _arena
    if _arena is None:
        _arena = Arena()
    return _arena


def learn(category: str, trigger: str, insight: str) -> Dict[str, Any]:
    """Quick learn function."""
    tracker = get_tracker()
    lesson = tracker.learn(category, trigger, insight)
    return lesson.to_dict()


def evolve_metric(wolf_name: str, metric: str, value: float):
    """Record an evolution metric."""
    tracker = get_tracker()
    tracker.record_metric(wolf_name, metric, value)


def get_evolution() -> Dict[str, Any]:
    """Get evolution summary."""
    tracker = get_tracker()
    return tracker.get_evolution_summary()
