"""Dataset loaders."""

from identity_bias.config import Dataset
from identity_bias.data.base import Problem, Solution


def load_dataset_problems(
    dataset: Dataset,
    n_samples: int | None = None,
    seed: int = 42,
    **kwargs,
) -> list[Problem]:
    """Load problems from the specified dataset."""
    if dataset == Dataset.MATH:
        from identity_bias.data.math_dataset import load_math
        return load_math(n_samples=n_samples, seed=seed, **kwargs)
    elif dataset == Dataset.GPQA:
        from identity_bias.data.gpqa import load_gpqa
        return load_gpqa(n_samples=n_samples, seed=seed)
    elif dataset == Dataset.BBH:
        from identity_bias.data.bbh import load_bbh
        return load_bbh(n_samples=n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def check_answer(dataset: Dataset, predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth for the given dataset."""
    if dataset == Dataset.MATH:
        from identity_bias.data.math_dataset import check_math_answer
        return check_math_answer(predicted, ground_truth)
    elif dataset == Dataset.GPQA:
        from identity_bias.data.gpqa import check_gpqa_answer
        return check_gpqa_answer(predicted, ground_truth)
    elif dataset == Dataset.BBH:
        from identity_bias.data.bbh import check_bbh_answer
        return check_bbh_answer(predicted, ground_truth)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


__all__ = ["Problem", "Solution", "load_dataset_problems", "check_answer"]
