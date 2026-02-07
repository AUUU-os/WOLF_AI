"""
Track Module - Navigation and search for WOLF_AI

Handles:
- File system navigation
- Code search (grep-like)
- Pattern matching
- Territory mapping
"""

import os
import re
import fnmatch
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WOLF_ROOT, BRIDGE_PATH


@dataclass
class TrackResult:
    """Result of a tracking operation."""
    path: str
    type: str  # file, directory, match
    line: Optional[int] = None
    content: Optional[str] = None
    context: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "type": self.type,
            "line": self.line,
            "content": self.content,
            "context": self.context
        }


class Tracker:
    """Tracks files, patterns, and code in the territory."""

    def __init__(self, root: Optional[Path] = None):
        self.root = root or WOLF_ROOT
        self.ignore_patterns = [
            ".git", "__pycache__", "*.pyc", "node_modules",
            ".env", "*.log", "venv", ".venv"
        ]

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        name = path.name
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    # =========================================================================
    # FILE SEARCH
    # =========================================================================

    def find_files(self, pattern: str, path: Optional[Path] = None,
                   max_results: int = 100) -> List[TrackResult]:
        """Find files matching a glob pattern."""
        search_path = path or self.root
        results = []

        for root, dirs, files in os.walk(search_path):
            # Filter ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(Path(root) / d)]

            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    filepath = Path(root) / filename
                    if not self._should_ignore(filepath):
                        results.append(TrackResult(
                            path=str(filepath),
                            type="file"
                        ))
                        if len(results) >= max_results:
                            return results

        return results

    def find_dirs(self, pattern: str, path: Optional[Path] = None,
                  max_results: int = 50) -> List[TrackResult]:
        """Find directories matching a pattern."""
        search_path = path or self.root
        results = []

        for root, dirs, _ in os.walk(search_path):
            dirs[:] = [d for d in dirs if not self._should_ignore(Path(root) / d)]

            for dirname in dirs:
                if fnmatch.fnmatch(dirname, pattern):
                    results.append(TrackResult(
                        path=str(Path(root) / dirname),
                        type="directory"
                    ))
                    if len(results) >= max_results:
                        return results

        return results

    # =========================================================================
    # CONTENT SEARCH (grep-like)
    # =========================================================================

    def grep(self, pattern: str, path: Optional[Path] = None,
             file_pattern: str = "*", context_lines: int = 0,
             max_results: int = 100, case_sensitive: bool = True) -> List[TrackResult]:
        """Search for pattern in files (grep-like)."""
        search_path = path or self.root
        results = []

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            # Treat as literal string if invalid regex
            regex = re.compile(re.escape(pattern), flags)

        for root, dirs, files in os.walk(search_path):
            dirs[:] = [d for d in dirs if not self._should_ignore(Path(root) / d)]

            for filename in files:
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue

                filepath = Path(root) / filename
                if self._should_ignore(filepath):
                    continue

                try:
                    matches = self._search_file(filepath, regex, context_lines)
                    results.extend(matches)
                    if len(results) >= max_results:
                        return results[:max_results]
                except (UnicodeDecodeError, PermissionError, IOError):
                    continue

        return results

    def _search_file(self, filepath: Path, regex: re.Pattern,
                     context_lines: int = 0) -> List[TrackResult]:
        """Search a single file for pattern matches."""
        results = []

        try:
            lines = filepath.read_text(encoding="utf-8").splitlines()
        except:
            return results

        for i, line in enumerate(lines, 1):
            if regex.search(line):
                context = None
                if context_lines > 0:
                    start = max(0, i - 1 - context_lines)
                    end = min(len(lines), i + context_lines)
                    context = lines[start:end]

                results.append(TrackResult(
                    path=str(filepath),
                    type="match",
                    line=i,
                    content=line.strip(),
                    context=context
                ))

        return results

    # =========================================================================
    # CODE ANALYSIS
    # =========================================================================

    def find_functions(self, name_pattern: str = "*",
                       path: Optional[Path] = None) -> List[TrackResult]:
        """Find function definitions in Python files."""
        name_regex = name_pattern.replace('*', r'\w*')
        pattern = rf"^\s*def\s+{name_regex}\s*\("
        return self.grep(pattern, path, "*.py")

    def find_classes(self, name_pattern: str = "*",
                     path: Optional[Path] = None) -> List[TrackResult]:
        """Find class definitions in Python files."""
        name_regex = name_pattern.replace('*', r'\w*')
        pattern = rf"^\s*class\s+{name_regex}\s*[:\(]"
        return self.grep(pattern, path, "*.py")

    def find_imports(self, module_name: str,
                     path: Optional[Path] = None) -> List[TrackResult]:
        """Find import statements for a module."""
        patterns = [
            rf"^\s*import\s+{module_name}",
            rf"^\s*from\s+{module_name}\s+import"
        ]
        results = []
        for pattern in patterns:
            results.extend(self.grep(pattern, path, "*.py"))
        return results

    def find_todos(self, path: Optional[Path] = None) -> List[TrackResult]:
        """Find TODO, FIXME, XXX comments."""
        return self.grep(r"#\s*(TODO|FIXME|XXX|HACK|BUG)", path, "*.py",
                        case_sensitive=False)

    # =========================================================================
    # TERRITORY MAPPING
    # =========================================================================

    def map_territory(self, path: Optional[Path] = None,
                      max_depth: int = 3) -> Dict[str, Any]:
        """Create a map of the territory (directory structure)."""
        search_path = path or self.root

        def build_tree(current_path: Path, depth: int) -> Dict[str, Any]:
            if depth > max_depth:
                return {"type": "...", "truncated": True}

            if self._should_ignore(current_path):
                return {"type": "ignored"}

            if current_path.is_file():
                size = current_path.stat().st_size
                return {
                    "type": "file",
                    "size": size,
                    "extension": current_path.suffix
                }

            if current_path.is_dir():
                children = {}
                try:
                    for child in sorted(current_path.iterdir()):
                        if not self._should_ignore(child):
                            children[child.name] = build_tree(child, depth + 1)
                except PermissionError:
                    return {"type": "directory", "error": "permission denied"}

                return {
                    "type": "directory",
                    "children": children,
                    "count": len(children)
                }

            return {"type": "unknown"}

        return {
            "root": str(search_path),
            "tree": build_tree(search_path, 0)
        }

    def get_stats(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Get statistics about the territory."""
        search_path = path or self.root

        stats = {
            "total_files": 0,
            "total_dirs": 0,
            "total_size": 0,
            "by_extension": {},
            "largest_files": []
        }

        files_with_size = []

        for root, dirs, files in os.walk(search_path):
            dirs[:] = [d for d in dirs if not self._should_ignore(Path(root) / d)]
            stats["total_dirs"] += len(dirs)

            for filename in files:
                filepath = Path(root) / filename
                if self._should_ignore(filepath):
                    continue

                stats["total_files"] += 1
                try:
                    size = filepath.stat().st_size
                    stats["total_size"] += size
                    files_with_size.append((str(filepath), size))

                    ext = filepath.suffix or "(no extension)"
                    if ext not in stats["by_extension"]:
                        stats["by_extension"][ext] = {"count": 0, "size": 0}
                    stats["by_extension"][ext]["count"] += 1
                    stats["by_extension"][ext]["size"] += size
                except:
                    pass

        # Top 10 largest files
        files_with_size.sort(key=lambda x: x[1], reverse=True)
        stats["largest_files"] = [
            {"path": p, "size": s} for p, s in files_with_size[:10]
        ]

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_tracker: Optional[Tracker] = None


def get_tracker() -> Tracker:
    """Get or create the tracker."""
    global _tracker
    if _tracker is None:
        _tracker = Tracker()
    return _tracker


def track(pattern: str, search_type: str = "files") -> List[Dict[str, Any]]:
    """Quick tracking function."""
    tracker = get_tracker()

    if search_type == "files":
        results = tracker.find_files(pattern)
    elif search_type == "dirs":
        results = tracker.find_dirs(pattern)
    elif search_type == "grep":
        results = tracker.grep(pattern)
    elif search_type == "functions":
        results = tracker.find_functions(pattern)
    elif search_type == "classes":
        results = tracker.find_classes(pattern)
    else:
        results = tracker.grep(pattern)

    return [r.to_dict() for r in results]


def find(pattern: str) -> List[str]:
    """Find files matching pattern, return paths only."""
    tracker = get_tracker()
    results = tracker.find_files(pattern)
    return [r.path for r in results]


def grep(pattern: str, file_pattern: str = "*") -> List[Dict[str, Any]]:
    """Search for pattern in files."""
    tracker = get_tracker()
    results = tracker.grep(pattern, file_pattern=file_pattern, context_lines=1)
    return [r.to_dict() for r in results]


def map_territory() -> Dict[str, Any]:
    """Map the current territory."""
    tracker = get_tracker()
    return tracker.map_territory()
