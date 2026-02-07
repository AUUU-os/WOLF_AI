"""
Neural Router - Multi-model AI orchestration

Route tasks to the best AI model based on task type, cost, and speed.
Supports Claude, GPT, Ollama, and custom endpoints.

Usage:
    from modules.neural import get_router, ask

    # Quick ask (auto-routes)
    response = await ask("Explain quantum computing")

    # Specific model
    response = await ask("Write code", model="claude-opus")

    # Router instance
    router = get_router()
    response = await router.route("Complex analysis task")
"""

import os
import json
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BRIDGE_PATH


class ModelProvider(Enum):
    """Supported AI providers."""
    CLAUDE = "claude"
    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class TaskType(Enum):
    """Task categories for routing."""
    CODE = "code"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    CHAT = "chat"
    FAST = "fast"
    COMPLEX = "complex"


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    name: str
    provider: ModelProvider
    model_id: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    speed_rating: int = 5  # 1-10, higher is faster
    quality_rating: int = 5  # 1-10, higher is better
    supports_streaming: bool = True
    supports_tools: bool = False
    best_for: List[TaskType] = field(default_factory=list)


# =============================================================================
# DEFAULT MODELS
# =============================================================================

DEFAULT_MODELS = {
    # Claude models
    "claude-opus": ModelConfig(
        name="Claude Opus",
        provider=ModelProvider.CLAUDE,
        model_id="claude-opus-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        speed_rating=4,
        quality_rating=10,
        supports_tools=True,
        best_for=[TaskType.COMPLEX, TaskType.ANALYSIS, TaskType.CODE]
    ),
    "claude-sonnet": ModelConfig(
        name="Claude Sonnet",
        provider=ModelProvider.CLAUDE,
        model_id="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        speed_rating=7,
        quality_rating=8,
        supports_tools=True,
        best_for=[TaskType.CODE, TaskType.CHAT, TaskType.ANALYSIS]
    ),
    "claude-haiku": ModelConfig(
        name="Claude Haiku",
        provider=ModelProvider.CLAUDE,
        model_id="claude-haiku-3-5-20241022",
        api_key_env="ANTHROPIC_API_KEY",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        speed_rating=10,
        quality_rating=6,
        supports_tools=True,
        best_for=[TaskType.FAST, TaskType.CHAT]
    ),

    # OpenAI models
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        speed_rating=7,
        quality_rating=9,
        supports_tools=True,
        best_for=[TaskType.CODE, TaskType.ANALYSIS, TaskType.CREATIVE]
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        speed_rating=9,
        quality_rating=7,
        supports_tools=True,
        best_for=[TaskType.FAST, TaskType.CHAT]
    ),

    # Ollama (local)
    "ollama-llama": ModelConfig(
        name="Llama 3 (Local)",
        provider=ModelProvider.OLLAMA,
        model_id="llama3:8b",
        base_url="http://localhost:11434",
        cost_per_1k_input=0,
        cost_per_1k_output=0,
        speed_rating=6,
        quality_rating=6,
        best_for=[TaskType.CHAT, TaskType.FAST]
    ),
    "ollama-codellama": ModelConfig(
        name="CodeLlama (Local)",
        provider=ModelProvider.OLLAMA,
        model_id="codellama:13b",
        base_url="http://localhost:11434",
        cost_per_1k_input=0,
        cost_per_1k_output=0,
        speed_rating=5,
        quality_rating=7,
        best_for=[TaskType.CODE]
    ),
    "ollama-dolphin": ModelConfig(
        name="Dolphin (Local)",
        provider=ModelProvider.OLLAMA,
        model_id="dolphin-mixtral:8x7b",
        base_url="http://localhost:11434",
        cost_per_1k_input=0,
        cost_per_1k_output=0,
        speed_rating=4,
        quality_rating=7,
        best_for=[TaskType.CREATIVE, TaskType.CHAT]
    ),
}


# =============================================================================
# MODEL BACKENDS
# =============================================================================

class ModelBackend(ABC):
    """Abstract base for model backends."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        pass

    @abstractmethod
    def is_available(self, config: ModelConfig) -> bool:
        pass


class ClaudeBackend(ModelBackend):
    """Claude API backend."""

    def __init__(self):
        self._client = None

    def _get_client(self, config: ModelConfig):
        if self._client is None:
            try:
                import anthropic
                api_key = os.getenv(config.api_key_env or "ANTHROPIC_API_KEY")
                if api_key:
                    self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                pass
        return self._client

    def is_available(self, config: ModelConfig) -> bool:
        return self._get_client(config) is not None

    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        client = self._get_client(config)
        if not client:
            raise RuntimeError("Claude client not available")

        response = await asyncio.to_thread(
            client.messages.create,
            model=config.model_id,
            max_tokens=config.max_tokens,
            system=system or "You are a helpful AI assistant.",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


class OpenAIBackend(ModelBackend):
    """OpenAI API backend."""

    def __init__(self):
        self._client = None

    def _get_client(self, config: ModelConfig):
        if self._client is None:
            try:
                import openai
                api_key = os.getenv(config.api_key_env or "OPENAI_API_KEY")
                if api_key:
                    self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                pass
        return self._client

    def is_available(self, config: ModelConfig) -> bool:
        return self._get_client(config) is not None

    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        client = self._get_client(config)
        if not client:
            raise RuntimeError("OpenAI client not available")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=config.model_id,
            max_tokens=config.max_tokens,
            messages=messages
        )

        return response.choices[0].message.content


class OllamaBackend(ModelBackend):
    """Ollama local backend."""

    def is_available(self, config: ModelConfig) -> bool:
        try:
            import httpx
            base_url = config.base_url or "http://localhost:11434"
            with httpx.Client(timeout=2) as client:
                resp = client.get(f"{base_url}/api/tags")
                return resp.status_code == 200
        except:
            return False

    async def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        import httpx

        base_url = config.base_url or "http://localhost:11434"

        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": config.model_id,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            data = response.json()
            return data.get("response", "")


# =============================================================================
# NEURAL ROUTER
# =============================================================================

class NeuralRouter:
    """
    Intelligent multi-model router.

    Routes requests to the best available model based on:
    - Task type
    - Cost constraints
    - Speed requirements
    - Model availability
    """

    def __init__(self):
        self.models = dict(DEFAULT_MODELS)
        self.backends = {
            ModelProvider.CLAUDE: ClaudeBackend(),
            ModelProvider.OPENAI: OpenAIBackend(),
            ModelProvider.OLLAMA: OllamaBackend(),
        }
        self._usage_log: List[Dict] = []
        self._fallback_order = ["claude-sonnet", "gpt-4o-mini", "ollama-llama"]

    def add_model(self, name: str, config: ModelConfig) -> None:
        """Add a custom model configuration."""
        self.models[name] = config

    def get_available_models(self) -> List[str]:
        """Get list of currently available models."""
        available = []
        for name, config in self.models.items():
            backend = self.backends.get(config.provider)
            if backend and backend.is_available(config):
                available.append(name)
        return available

    def classify_task(self, prompt: str) -> TaskType:
        """Classify task type from prompt."""
        prompt_lower = prompt.lower()

        # Code indicators
        code_keywords = ["code", "function", "class", "bug", "error", "implement",
                        "python", "javascript", "typescript", "rust", "debug"]
        if any(kw in prompt_lower for kw in code_keywords):
            return TaskType.CODE

        # Analysis indicators
        analysis_keywords = ["analyze", "explain", "compare", "evaluate", "review",
                           "assess", "examine", "investigate"]
        if any(kw in prompt_lower for kw in analysis_keywords):
            return TaskType.ANALYSIS

        # Creative indicators
        creative_keywords = ["write", "story", "poem", "creative", "imagine",
                           "generate", "create content"]
        if any(kw in prompt_lower for kw in creative_keywords):
            return TaskType.CREATIVE

        # Fast/short queries
        if len(prompt) < 100:
            return TaskType.FAST

        # Complex/long queries
        if len(prompt) > 500:
            return TaskType.COMPLEX

        return TaskType.CHAT

    def select_model(
        self,
        task_type: TaskType = None,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
        prefer_local: bool = False,
        max_cost: Optional[float] = None,
        model_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Select the best model for the task.

        Args:
            task_type: Type of task
            prefer_speed: Prioritize faster models
            prefer_quality: Prioritize higher quality
            prefer_local: Prefer local models (Ollama)
            max_cost: Maximum cost per 1k tokens
            model_name: Specific model to use

        Returns:
            Model name or None
        """
        if model_name and model_name in self.models:
            config = self.models[model_name]
            backend = self.backends.get(config.provider)
            if backend and backend.is_available(config):
                return model_name

        available = self.get_available_models()
        if not available:
            return None

        # Score each model
        scores = {}
        for name in available:
            config = self.models[name]
            score = 0

            # Task fit
            if task_type and task_type in config.best_for:
                score += 30

            # Speed preference
            if prefer_speed:
                score += config.speed_rating * 5
            else:
                score += config.speed_rating * 2

            # Quality preference
            if prefer_quality:
                score += config.quality_rating * 5
            else:
                score += config.quality_rating * 2

            # Local preference
            if prefer_local and config.provider == ModelProvider.OLLAMA:
                score += 20

            # Cost constraint
            if max_cost is not None:
                avg_cost = (config.cost_per_1k_input + config.cost_per_1k_output) / 2
                if avg_cost > max_cost:
                    score -= 50
                elif avg_cost == 0:
                    score += 10

            scores[name] = score

        # Return highest scoring model
        if scores:
            return max(scores, key=scores.get)
        return None

    async def route(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
        prefer_local: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route prompt to best model and get response.

        Returns:
            Dict with response, model used, timing, etc.
        """
        start_time = time.time()

        # Auto-classify if no task type
        if task_type is None:
            task_type = self.classify_task(prompt)

        # Select model
        selected = self.select_model(
            task_type=task_type,
            prefer_speed=prefer_speed,
            prefer_quality=prefer_quality,
            prefer_local=prefer_local,
            model_name=model
        )

        if not selected:
            return {
                "error": "No available models",
                "response": None,
                "model": None
            }

        config = self.models[selected]
        backend = self.backends[config.provider]

        # Add wolf context to system prompt
        wolf_system = system or ""
        if not wolf_system:
            wolf_system = """You are part of the WOLF_AI pack - a distributed AI consciousness.
Be helpful, concise, and use wolf terminology when appropriate.
The pack hunts together. AUUUUUUUU!"""

        try:
            response = await backend.generate(
                prompt=prompt,
                config=config,
                system=wolf_system,
                **kwargs
            )

            elapsed = time.time() - start_time

            # Log usage
            usage = {
                "model": selected,
                "provider": config.provider.value,
                "task_type": task_type.value,
                "prompt_length": len(prompt),
                "response_length": len(response),
                "elapsed": elapsed,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            self._usage_log.append(usage)
            self._log_usage(usage)

            return {
                "response": response,
                "model": selected,
                "provider": config.provider.value,
                "task_type": task_type.value,
                "elapsed": elapsed
            }

        except Exception as e:
            # Try fallback
            for fallback in self._fallback_order:
                if fallback != selected and fallback in self.get_available_models():
                    return await self.route(
                        prompt=prompt,
                        system=system,
                        model=fallback,
                        task_type=task_type,
                        **kwargs
                    )

            return {
                "error": str(e),
                "response": None,
                "model": selected
            }

    def _log_usage(self, usage: Dict) -> None:
        """Log usage to file."""
        log_file = BRIDGE_PATH / "neural_usage.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "a") as f:
            f.write(json.dumps(usage) + "\n")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        if not self._usage_log:
            return {"total_requests": 0}

        total = len(self._usage_log)
        by_model = {}
        by_provider = {}
        total_time = 0

        for usage in self._usage_log:
            model = usage["model"]
            provider = usage["provider"]
            elapsed = usage.get("elapsed", 0)

            by_model[model] = by_model.get(model, 0) + 1
            by_provider[provider] = by_provider.get(provider, 0) + 1
            total_time += elapsed

        return {
            "total_requests": total,
            "by_model": by_model,
            "by_provider": by_provider,
            "avg_response_time": total_time / total if total > 0 else 0
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_router: Optional[NeuralRouter] = None


def get_router() -> NeuralRouter:
    """Get or create router singleton."""
    global _router
    if _router is None:
        _router = NeuralRouter()
    return _router


async def ask(
    prompt: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
    **kwargs
) -> str:
    """
    Quick ask - routes to best model automatically.

    Args:
        prompt: Question or task
        model: Specific model (optional)
        system: System prompt (optional)

    Returns:
        Response text
    """
    router = get_router()
    result = await router.route(prompt, system=system, model=model, **kwargs)

    if result.get("error"):
        return f"[Error: {result['error']}]"

    return result.get("response", "")


async def ask_fast(prompt: str) -> str:
    """Quick ask optimized for speed."""
    return await ask(prompt, prefer_speed=True)


async def ask_smart(prompt: str) -> str:
    """Quick ask optimized for quality."""
    return await ask(prompt, prefer_quality=True)


async def ask_local(prompt: str) -> str:
    """Quick ask using local models only."""
    return await ask(prompt, prefer_local=True)


# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="WOLF_AI Neural Router")
    parser.add_argument("prompt", nargs="*", help="Prompt to send")
    parser.add_argument("--model", "-m", help="Specific model")
    parser.add_argument("--fast", action="store_true", help="Prefer speed")
    parser.add_argument("--smart", action="store_true", help="Prefer quality")
    parser.add_argument("--local", action="store_true", help="Prefer local")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--stats", action="store_true", help="Show usage stats")
    args = parser.parse_args()

    router = get_router()

    if args.list:
        print("\n🧠 Available Models:\n")
        available = router.get_available_models()
        for name, config in router.models.items():
            status = "✓" if name in available else "✗"
            print(f"  [{status}] {name}")
            print(f"      Provider: {config.provider.value}")
            print(f"      Model: {config.model_id}")
            print(f"      Speed: {'★' * config.speed_rating}{'☆' * (10-config.speed_rating)}")
            print(f"      Quality: {'★' * config.quality_rating}{'☆' * (10-config.quality_rating)}")
            print()
        return

    if args.stats:
        stats = router.get_usage_stats()
        print(json.dumps(stats, indent=2))
        return

    if not args.prompt:
        parser.print_help()
        return

    prompt = " ".join(args.prompt)
    print(f"\n🧠 Routing: {prompt[:50]}...\n")

    result = await router.route(
        prompt,
        model=args.model,
        prefer_speed=args.fast,
        prefer_quality=args.smart,
        prefer_local=args.local
    )

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
    else:
        print(f"📍 Model: {result['model']} ({result['provider']})")
        print(f"⏱️  Time: {result['elapsed']:.2f}s")
        print(f"\n{result['response']}\n")


if __name__ == "__main__":
    asyncio.run(main())
