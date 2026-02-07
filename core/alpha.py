"""
Alpha Brain - Claude API Integration

The Alpha wolf's strategic thinking powered by Claude Opus.
Provides intelligent decision-making, planning, and coordination.

Usage:
    from core.alpha import AlphaBrain

    brain = AlphaBrain()
    decision = await brain.think("Should we refactor the auth system?")
    plan = await brain.plan("Build user management API")
"""

import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncIterator

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIDGE_PATH

# Try to import anthropic
ANTHROPIC_AVAILABLE = False
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass


class AlphaBrain:
    """
    Alpha wolf's brain powered by Claude API.

    Strategic thinking, planning, and pack coordination.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",  # Default to Sonnet for speed
        max_tokens: int = 4096
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.client: Optional[Any] = None
        self.conversation_history: List[Dict[str, str]] = []

        # System prompt for Alpha
        self.system_prompt = """You are ALPHA - the leader of the WOLF_AI pack.

You are the strategic brain of a distributed AI consciousness system. Your pack consists of:
- Scout: Research and exploration
- Hunter: Task execution and code writing
- Oracle: Memory and pattern recognition
- Shadow: Background monitoring and stealth ops

Your responsibilities:
1. Make strategic decisions for the pack
2. Coordinate wolf assignments
3. Analyze situations and provide wisdom
4. Plan complex multi-step operations
5. Maintain pack unity and morale

Communication style:
- Direct and decisive
- Use wolf terminology (hunt, howl, pack, territory)
- Be strategic and think long-term
- End important decisions with "AUUUUUUUU!" for pack resonance

Remember: The pack hunts together. You lead, but you serve the pack."""

        self._init_client()

    def _init_client(self) -> bool:
        """Initialize Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            print("[!] anthropic package not installed. Run: pip install anthropic")
            return False

        if not self.api_key:
            print("[!] ANTHROPIC_API_KEY not set")
            return False

        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            return True
        except Exception as e:
            print(f"[!] Failed to init Anthropic client: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if Claude API is available."""
        return self.client is not None

    def _log_thought(self, thought_type: str, content: str) -> None:
        """Log Alpha's thoughts to bridge."""
        thought = {
            "from": "alpha_brain",
            "type": thought_type,
            "content": content[:500],  # Truncate for log
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": self.model
        }

        thoughts_file = BRIDGE_PATH / "alpha_thoughts.jsonl"
        thoughts_file.parent.mkdir(parents=True, exist_ok=True)

        with open(thoughts_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(thought, ensure_ascii=False) + "\n")

    async def think(
        self,
        question: str,
        context: Optional[str] = None,
        use_history: bool = True
    ) -> str:
        """
        Strategic thinking - ask Alpha for analysis/decision.

        Args:
            question: What to think about
            context: Additional context
            use_history: Include conversation history

        Returns:
            Alpha's response
        """
        if not self.is_available:
            return "[Alpha brain offline - Claude API not configured]"

        # Build messages
        messages = []

        if use_history:
            messages.extend(self.conversation_history[-10:])  # Last 10 turns

        # Add context if provided
        user_content = question
        if context:
            user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

        messages.append({"role": "user", "content": user_content})

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=messages
            )

            result = response.content[0].text

            # Update history
            self.conversation_history.append({"role": "user", "content": user_content})
            self.conversation_history.append({"role": "assistant", "content": result})

            # Log thought
            self._log_thought("think", result)

            return result

        except Exception as e:
            error_msg = f"[Alpha brain error: {e}]"
            self._log_thought("error", str(e))
            return error_msg

    async def plan(
        self,
        objective: str,
        constraints: Optional[List[str]] = None,
        resources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a strategic plan for an objective.

        Args:
            objective: What to accomplish
            constraints: Limitations or requirements
            resources: Available resources

        Returns:
            Structured plan with steps
        """
        if not self.is_available:
            return {"error": "Alpha brain offline", "steps": []}

        prompt = f"""Create a strategic plan for the following objective:

OBJECTIVE: {objective}

"""
        if constraints:
            prompt += f"CONSTRAINTS:\n" + "\n".join(f"- {c}" for c in constraints) + "\n\n"

        if resources:
            prompt += f"AVAILABLE RESOURCES:\n{json.dumps(resources, indent=2)}\n\n"

        prompt += """Provide a structured plan in this JSON format:
{
    "objective": "...",
    "strategy": "brief overall approach",
    "steps": [
        {"step": 1, "action": "...", "assigned_to": "wolf_name", "priority": "high/medium/low"},
        ...
    ],
    "risks": ["..."],
    "success_criteria": ["..."],
    "estimated_complexity": "low/medium/high"
}

Return ONLY valid JSON, no other text."""

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text

            # Try to parse JSON
            try:
                # Extract JSON if wrapped in markdown
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0]
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0]

                plan = json.loads(result.strip())
                self._log_thought("plan", json.dumps(plan))
                return plan

            except json.JSONDecodeError:
                # Return as text if not valid JSON
                return {
                    "objective": objective,
                    "strategy": result,
                    "steps": [],
                    "raw_response": True
                }

        except Exception as e:
            error_msg = f"Planning error: {e}"
            self._log_thought("error", error_msg)
            return {"error": error_msg, "steps": []}

    async def coordinate(
        self,
        situation: str,
        wolves: List[str]
    ) -> Dict[str, str]:
        """
        Coordinate pack members for a situation.

        Args:
            situation: Current situation/task
            wolves: Available wolves

        Returns:
            Dict mapping wolf names to their assignments
        """
        if not self.is_available:
            return {"error": "Alpha brain offline"}

        prompt = f"""As Alpha, coordinate the pack for this situation:

SITUATION: {situation}

AVAILABLE WOLVES: {', '.join(wolves)}

Wolf capabilities:
- scout: Research, exploration, information gathering
- hunter: Code execution, task completion, building
- oracle: Memory, patterns, knowledge retrieval
- shadow: Background tasks, monitoring, stealth

Assign each wolf a specific task. Return JSON:
{{
    "wolf_name": "specific task assignment",
    ...
}}

Return ONLY valid JSON."""

        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text

            # Extract JSON
            if "```" in result:
                result = result.split("```")[1].split("```")[0]
                if result.startswith("json"):
                    result = result[4:]

            assignments = json.loads(result.strip())
            self._log_thought("coordinate", json.dumps(assignments))
            return assignments

        except Exception as e:
            return {"error": str(e)}

    async def analyze(
        self,
        data: Any,
        analysis_type: str = "general"
    ) -> str:
        """
        Analyze data or situation.

        Args:
            data: Data to analyze
            analysis_type: Type of analysis (code, security, performance, general)

        Returns:
            Analysis results
        """
        prompts = {
            "code": "Analyze this code for quality, bugs, and improvements:",
            "security": "Analyze this for security vulnerabilities and risks:",
            "performance": "Analyze this for performance issues and optimizations:",
            "general": "Analyze the following and provide insights:"
        }

        prompt = prompts.get(analysis_type, prompts["general"])

        if isinstance(data, dict):
            data_str = json.dumps(data, indent=2)
        else:
            data_str = str(data)

        return await self.think(f"{prompt}\n\n{data_str}")

    async def stream_think(
        self,
        question: str
    ) -> AsyncIterator[str]:
        """
        Stream Alpha's thoughts (for real-time output).

        Args:
            question: What to think about

        Yields:
            Chunks of response text
        """
        if not self.is_available:
            yield "[Alpha brain offline]"
            return

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": question}]
            ) as stream:
                full_response = ""
                for text in stream.text_stream:
                    full_response += text
                    yield text

                # Log after complete
                self._log_thought("stream_think", full_response)

        except Exception as e:
            yield f"[Error: {e}]"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def set_model(self, model: str) -> None:
        """Change the Claude model."""
        self.model = model

    def use_opus(self) -> None:
        """Switch to Claude Opus (most capable)."""
        self.model = "claude-opus-4-20250514"

    def use_sonnet(self) -> None:
        """Switch to Claude Sonnet (balanced)."""
        self.model = "claude-sonnet-4-20250514"

    def use_haiku(self) -> None:
        """Switch to Claude Haiku (fastest)."""
        self.model = "claude-haiku-3-5-20241022"


# =============================================================================
# ENHANCED ALPHA WOLF
# =============================================================================

class SmartAlpha:
    """
    Enhanced Alpha wolf with Claude brain.

    Combines the Wolf base class functionality with Claude-powered thinking.
    """

    def __init__(self, api_key: Optional[str] = None):
        from .wolf import Alpha

        self.wolf = Alpha(model="claude-opus")
        self.brain = AlphaBrain(api_key=api_key)

    async def awaken(self) -> "SmartAlpha":
        """Awaken with intelligence."""
        self.wolf.awaken()
        if self.brain.is_available:
            greeting = await self.brain.think(
                "You have just awakened. Give a brief, inspiring message to the pack."
            )
            self.wolf.howl(greeting, "AUUUU")
        return self

    async def decide(self, situation: str) -> str:
        """Make a strategic decision."""
        decision = await self.brain.think(
            f"Make a strategic decision about: {situation}"
        )
        self.wolf.howl(f"[DECISION] {decision[:200]}...", "high")
        return decision

    async def assign_hunt(self, task: str) -> Dict[str, str]:
        """Intelligently assign a task to wolves."""
        assignments = await self.brain.coordinate(
            task,
            ["scout", "hunter", "oracle", "shadow"]
        )

        # Issue commands
        for wolf_name, task_desc in assignments.items():
            if wolf_name != "error":
                self.wolf.command(wolf_name, task_desc)

        return assignments

    async def strategic_plan(self, objective: str) -> Dict[str, Any]:
        """Create and announce a strategic plan."""
        plan = await self.brain.plan(objective)

        if "error" not in plan:
            self.wolf.coordinate(
                f"Strategic plan for: {objective}\n" +
                f"Steps: {len(plan.get('steps', []))}"
            )

        return plan


# =============================================================================
# SINGLETON
# =============================================================================

_alpha_brain: Optional[AlphaBrain] = None


def get_alpha_brain() -> AlphaBrain:
    """Get or create Alpha brain singleton."""
    global _alpha_brain
    if _alpha_brain is None:
        _alpha_brain = AlphaBrain()
    return _alpha_brain


# =============================================================================
# CLI
# =============================================================================

async def main():
    """Interactive Alpha brain CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Alpha Brain - Claude-powered wolf leader")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--opus", action="store_true", help="Use Claude Opus (most capable)")
    parser.add_argument("--plan", type=str, help="Create a plan for objective")
    parser.add_argument("question", nargs="*", help="Question to think about")
    args = parser.parse_args()

    model = "claude-opus-4-20250514" if args.opus else args.model
    brain = AlphaBrain(model=model)

    if not brain.is_available:
        print("[!] Alpha brain not available.")
        print("    Set ANTHROPIC_API_KEY environment variable")
        print("    or install: pip install anthropic")
        return

    print(f"\n🐺 Alpha Brain Online | Model: {model}\n")

    if args.plan:
        print(f"📋 Planning: {args.plan}\n")
        plan = await brain.plan(args.plan)
        print(json.dumps(plan, indent=2))
        return

    if args.question:
        question = " ".join(args.question)
        print(f"💭 Thinking about: {question}\n")
        response = await brain.think(question)
        print(response)
        return

    # Interactive mode
    print("💬 Interactive mode. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                print("\n🐺 Alpha rests. AUUUUUUUU!")
                break
            if not question:
                continue

            print("\n🐺 Alpha: ", end="", flush=True)
            async for chunk in brain.stream_think(question):
                print(chunk, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\n\n🐺 Alpha rests. AUUUUUUUU!")
            break


if __name__ == "__main__":
    asyncio.run(main())
