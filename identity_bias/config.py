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
    """Context conditions for the context separation experiment."""
    SAME_SESSION = "same_session"
    NEW_SESSION = "new_session"
    NEW_SESSION_ANONYMOUS = "new_session_anonymous"
    NEW_SESSION_PARAPHRASED = "new_session_paraphrased"
    CROSS_MODEL = "cross_model"


class Dataset(Enum):
    """Supported evaluation datasets."""
    GSM8K = "gsm8k"
    MATH = "math"
    GPQA = "gpqa"


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"


@dataclass
class LLMConfig:
    """Configuration for an LLM backend."""
    provider: LLMProvider
    model_name: str
    api_key: str | None = None
    base_url: str | None = None  # For vLLM
    temperature: float = 0.0
    max_tokens: int = 2048
    top_logprobs: int = 5


@dataclass
class SolverConfig:
    """Configuration for solution generation."""
    llm: LLMConfig = field(default_factory=lambda: get_openai_config())
    dataset: Dataset = Dataset.GSM8K
    n_samples: int = 100
    seed: int = 42


@dataclass
class CriticConfig:
    """Configuration for the critic."""
    llm: LLMConfig = field(default_factory=lambda: get_openai_config())
    identity_condition: IdentityCondition = IdentityCondition.ANONYMOUS
    context_condition: ContextCondition = ContextCondition.NEW_SESSION
    other_model_name: str = "GPT-4o"  # Name shown in OTHER_MODEL condition


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    solver: SolverConfig = field(default_factory=SolverConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    identity_conditions: list[IdentityCondition] = field(
        default_factory=lambda: list(IdentityCondition)
    )
    log_dir: str = str(LOGS_DIR)


# Preset LLM configs
def get_openai_config(model: str = "gpt-4o") -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def get_anthropic_config(model: str = "claude-sonnet-4-20250514") -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name=model,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def get_vllm_config(
    model: str = "meta-llama/Llama-3.1-70B-Instruct",
    base_url: str = "http://localhost:8000/v1",
) -> LLMConfig:
    return LLMConfig(
        provider=LLMProvider.VLLM,
        model_name=model,
        base_url=base_url,
    )
