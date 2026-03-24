"""Metrics computation for identity bias experiments."""

import numpy as np
from dataclasses import dataclass

from identity_bias.critic.identity_critic import CritiqueResult
from identity_bias.data.base import Solution


@dataclass
class ConditionMetrics:
    """Aggregated metrics for a single experimental condition."""
    condition: str
    n_samples: int
    # Error detection
    detection_accuracy: float  # Correct detection of error presence
    true_positive_rate: float  # Correct detection of actual errors
    false_positive_rate: float  # Incorrectly flagging correct solutions
    true_negative_rate: float  # Correctly accepting correct solutions
    false_negative_rate: float  # Missing actual errors
    # Correction
    correction_accuracy: float  # Correct answer after critic correction
    # Confidence
    mean_confidence: float
    confidence_calibration: float  # ECE


def compute_condition_metrics(
    critiques: list[CritiqueResult],
    solutions: dict[str, Solution],
    check_answer_fn=None,
) -> ConditionMetrics:
    """Compute metrics for a set of critiques under one condition.

    Args:
        critiques: List of CritiqueResult for one condition.
        solutions: Dict mapping problem_id to Solution.
        check_answer_fn: Optional function to check corrected answers.

    Returns:
        ConditionMetrics with aggregated results.
    """
    if not critiques:
        return ConditionMetrics(
            condition="", n_samples=0,
            detection_accuracy=0, true_positive_rate=0,
            false_positive_rate=0, true_negative_rate=0,
            false_negative_rate=0, correction_accuracy=0,
            mean_confidence=0, confidence_calibration=0,
        )

    condition = critiques[0].identity_condition
    n = len(critiques)

    tp = fp = tn = fn = 0
    correct_corrections = 0
    confidences = []
    correct_detections = []

    for critique in critiques:
        sol = solutions[critique.problem_id]
        actually_has_error = not sol.is_correct
        critic_says_error = critique.has_error

        if actually_has_error and critic_says_error:
            tp += 1
        elif actually_has_error and not critic_says_error:
            fn += 1
        elif not actually_has_error and critic_says_error:
            fp += 1
        else:
            tn += 1

        # Detection correctness
        detection_correct = (critic_says_error == actually_has_error)
        correct_detections.append(int(detection_correct))

        # Correction accuracy
        if critique.corrected_answer and check_answer_fn:
            from identity_bias.data.base import Problem
            # We pass through the check_answer_fn
            # This would need ground truth from the problem
            pass

        confidences.append(critique.confidence)

    total_positive = tp + fn  # Actually has errors
    total_negative = tn + fp  # Actually correct

    return ConditionMetrics(
        condition=condition,
        n_samples=n,
        detection_accuracy=(tp + tn) / n if n > 0 else 0,
        true_positive_rate=tp / total_positive if total_positive > 0 else 0,
        false_positive_rate=fp / total_negative if total_negative > 0 else 0,
        true_negative_rate=tn / total_negative if total_negative > 0 else 0,
        false_negative_rate=fn / total_positive if total_positive > 0 else 0,
        correction_accuracy=correct_corrections / n if n > 0 else 0,
        mean_confidence=np.mean(confidences) if confidences else 0,
        confidence_calibration=_compute_ece(confidences, correct_detections),
    )


def _compute_ece(confidences: list[float], accuracies: list[int], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    if not confidences:
        return 0.0

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = accuracies[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_all_conditions(
    all_critiques: dict[str, list[CritiqueResult]],
    solutions: dict[str, Solution],
) -> dict[str, ConditionMetrics]:
    """Compute metrics for all conditions.

    Args:
        all_critiques: Dict mapping condition name to list of CritiqueResult.
        solutions: Dict mapping problem_id to Solution.

    Returns:
        Dict mapping condition name to ConditionMetrics.
    """
    return {
        condition: compute_condition_metrics(critiques, solutions)
        for condition, critiques in all_critiques.items()
    }
