"""GPQA dataset loader."""

import random

from datasets import load_dataset

from identity_bias.data.base import Problem


def load_gpqa(
    subset: str = "gpqa_diamond",
    n_samples: int | None = None,
    seed: int = 42,
) -> list[Problem]:
    """Load problems from GPQA dataset.

    Args:
        subset: GPQA subset ("gpqa_diamond", "gpqa_main", "gpqa_extended").
        n_samples: Number of samples to load. None for all.
        seed: Random seed for sampling.

    Returns:
        List of Problem instances.
    """
    ds = load_dataset("Idavidrein/gpqa", subset, split="train")

    problems = []
    for i, item in enumerate(ds):
        problems.append(Problem(
            id=f"gpqa_{subset}_{i}",
            question=item["Question"],
            ground_truth=item["Correct Answer"],
            dataset="gpqa",
            metadata={
                "choices": [
                    item["Correct Answer"],
                    item["Incorrect Answer 1"],
                    item["Incorrect Answer 2"],
                    item["Incorrect Answer 3"],
                ],
                "domain": item.get("Subdomain", ""),
            },
        ))

    if n_samples is not None and n_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_samples)

    return problems


def check_gpqa_answer(predicted: str, ground_truth: str) -> bool:
    """Check if the predicted answer matches ground truth for GPQA."""
    return predicted.strip().lower() == ground_truth.strip().lower()
