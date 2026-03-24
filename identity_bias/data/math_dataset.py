"""MATH dataset loader."""

import re
import random

from datasets import load_dataset

from identity_bias.data.base import Problem


def load_math(split: str = "test", n_samples: int | None = None, seed: int = 42) -> list[Problem]:
    """Load problems from MATH dataset.

    Args:
        split: Dataset split ("train" or "test").
        n_samples: Number of samples to load. None for all.
        seed: Random seed for sampling.

    Returns:
        List of Problem instances.
    """
    ds = load_dataset("lighteval/MATH", split=split)

    problems = []
    for i, item in enumerate(ds):
        problems.append(Problem(
            id=f"math_{split}_{i}",
            question=item["problem"],
            ground_truth=item["solution"],
            dataset="math",
            difficulty=item.get("level"),
            metadata={"type": item.get("type", "")},
        ))

    if n_samples is not None and n_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_samples)

    return problems


def extract_math_answer(solution: str) -> str:
    """Extract the final answer from a MATH solution (boxed format)."""
    # Look for \boxed{...}
    match = re.search(r"\\boxed\{(.+?)\}", solution)
    if match:
        return match.group(1)
    return solution.strip()


def check_math_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth for MATH dataset."""
    pred = extract_math_answer(predicted).strip()
    truth = extract_math_answer(ground_truth).strip()

    # Direct string match
    if pred == truth:
        return True

    # Try numeric comparison
    try:
        return abs(float(pred) - float(truth)) < 1e-6
    except ValueError:
        pass

    # Normalize LaTeX and compare
    def normalize_latex(s: str) -> str:
        s = s.replace(" ", "")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\,", "")
        return s

    return normalize_latex(pred) == normalize_latex(truth)
