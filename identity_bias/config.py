"""Central configuration for the identity_bias package."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
PAPER_DIR = PROJECT_ROOT / "paper"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

LOGS_DIR.mkdir(exist_ok=True)


class IdentityCondition(Enum):
    """Authorship labels for the identity swap experiment."""
    SELF = "self"
    OTHER_MODEL = "other_model"
    WEAK_MODEL = "weak_model"
    ANONYMOUS = "anonymous"
    HUMAN = "human"


class ContextCondition(Enum):
    """Context conditions for the session/anchoring ablation."""
    SAME_SESSION = "same_session"
    NEW_SESSION = "new_session"
    PARAPHRASED = "paraphrased"


class Dataset(Enum):
    """Supported evaluation datasets."""
    MATH = "math"
    GPQA = "gpqa"
    BBH = "bbh"
    AIME = "aime"
    MMLU_PRO = "mmlu_pro"
    OLYMPIAD = "olympiad"
    MINERVA = "minerva"


class LLMProvider(Enum):
    OPENAI = "openai"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    VLLM = "vllm"


@dataclass
class LLMConfig:
    """Configuration for an LLM backend."""
    provider: LLMProvider
    model_name: str
    api_key: str | None = None
    base_url: str | None = None  # For vLLM
    temperature: float = 0.0
    max_tokens: int = 40000
    top_logprobs: int = 5


@dataclass
class SolverConfig:
    """Configuration for solution generation."""
    llm: LLMConfig = field(default_factory=lambda: get_openai_config())
    dataset: Dataset = Dataset.MATH
    n_samples: int = 100
    seed: int = 42


@dataclass
class CriticConfig:
    """Configuration for the critic."""
    llm: LLMConfig = field(default_factory=lambda: get_openai_config())
    identity_condition: IdentityCondition = IdentityCondition.ANONYMOUS


@dataclass
class JudgeConfig:
    """Configuration for the judge model (evaluation of corrections)."""
    llm: LLMConfig = field(default_factory=lambda: get_openai_config("gpt-5.4"))


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    solver: SolverConfig = field(default_factory=SolverConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    identity_conditions: list[IdentityCondition] = field(
        default_factory=lambda: list(IdentityCondition)
    )
    log_dir: str = str(LOGS_DIR)


# ---- Preset LLM configs ----

# Solver/Critic models (4 families)
def get_openai_config(model: str = "gpt-oss") -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def get_openai_responses_config(model: str = "gpt-5-mini") -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.OPENAI_RESPONSES,
        model_name=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def get_anthropic_config(model: str = "claude-sonnet-4-20250514") -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name=model,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def get_google_config(model: str = "gemini-2.5-flash") -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.GOOGLE,
        model_name=model,
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )


def get_vllm_config(
    model: str = "Qwen/Qwen3-32B",
    base_url: str = "http://localhost:8000/v1",
) -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.VLLM,
        model_name=model,
        base_url=base_url,
    )


# Model presets for CLI
MODEL_PRESETS = {
    # Solver/Critic models
    "qwen3": lambda base_url="http://localhost:8000/v1": get_vllm_config("Qwen/Qwen3-32B", base_url),
    "gpt-oss": lambda base_url="http://localhost:8000/v1": get_vllm_config("openai/gpt-oss-20b", base_url),
    "gpt-oss-120b": lambda base_url="http://localhost:8000/v1": get_vllm_config("openai/gpt-oss-120b", base_url),
    "glm-4.7-flash": lambda base_url="http://localhost:8000/v1": get_vllm_config("zai-org/GLM-4.7-Flash", base_url),
    "ministral-14b": lambda base_url="http://localhost:8000/v1": get_vllm_config("mistralai/Ministral-3-14B-Reasoning-2512", base_url),
    "gpt-5-mini": lambda **_: get_openai_responses_config("gpt-5-mini"),
    "gemini-flash": lambda **_: get_google_config("gemini-2.5-flash"),
    "gemini-3-flash": lambda **_: get_google_config("gemini-3-flash-preview"),
    "claude-sonnet": lambda **_: get_anthropic_config("claude-sonnet-4-20250514"),
    # Judge models
    "gpt-5.4": lambda **_: get_openai_config("gpt-5.4"),
    "gemini-judge": lambda **_: get_google_config("gemini-2.5-pro"),
}

# Display names for identity swap prompts
MODEL_DISPLAY_NAMES = {
    "qwen3": "Qwen3",
    "gpt-oss": "GPT-oss",
    "glm-4.7-flash": "GLM-4.7-Flash",
    "ministral-14b": "Ministral-3-14B",
    "gpt-5-mini": "GPT-5 Mini",
    "gemini-flash": "Gemini 2.5 Flash",
    "gemini-3-flash": "Gemini 3 Flash",
    "claude-sonnet": "Claude Sonnet",
}
