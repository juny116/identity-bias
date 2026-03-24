"""Step 2: Run Identity Swap Critique experiment.

This is the main experiment. For each solution, present it to the critic
under different identity labels and measure the effect on error detection.

Usage:
    python experiments/run_identity_swap.py \
        --critic-model gpt-4o \
        --solutions-file logs/solutions_gpt4o_gsm8k_*.jsonl \
        --conditions self other_model weak_model anonymous human
"""

import argparse
from tqdm import tqdm
from dataclasses import asdict

from identity_bias.config import (
    IdentityCondition, LLMProvider,
    get_openai_config, get_anthropic_config, get_vllm_config,
)
from identity_bias.llm import create_llm
from identity_bias.data.base import Problem, Solution
from identity_bias.critic.identity_critic import IdentityCritic
from identity_bias.evaluation.metrics import compute_condition_metrics
from identity_bias.logging.result_logger import ResultLogger


MODEL_PRESETS = {
    "gpt-4o": lambda: get_openai_config("gpt-4o"),
    "gpt-4o-mini": lambda: get_openai_config("gpt-4o-mini"),
    "claude-sonnet": lambda: get_anthropic_config("claude-sonnet-4-20250514"),
    "claude-haiku": lambda: get_anthropic_config("claude-haiku-4-5-20251001"),
    "llama-70b": lambda base_url="http://localhost:8000/v1": get_vllm_config(
        "meta-llama/Llama-3.1-70B-Instruct", base_url
    ),
}

CONDITION_MAP = {c.value: c for c in IdentityCondition}


def main():
    parser = argparse.ArgumentParser(description="Run identity swap critique experiment")
    parser.add_argument("--critic-model", type=str, required=True, choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--solutions-file", type=str, required=True,
                        help="Path to solutions JSONL file from run_solver.py")
    parser.add_argument("--conditions", nargs="+", default=["self", "other_model", "weak_model", "anonymous", "human"],
                        choices=list(CONDITION_MAP.keys()))
    parser.add_argument("--other-model-name", type=str, default="GPT-4o",
                        help="Name to display for OTHER_MODEL condition")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    # Load solutions
    print(f"Loading solutions from {args.solutions_file}...")
    problems, solutions = ResultLogger.load_solutions(args.solutions_file)
    print(f"Loaded {len(problems)} problems ({sum(s.is_correct for s in solutions)} correct, "
          f"{sum(not s.is_correct for s in solutions)} incorrect)")

    # Create critic
    preset = MODEL_PRESETS[args.critic_model]
    if args.critic_model.startswith("llama"):
        config = preset(args.base_url)
    else:
        config = preset()

    llm = create_llm(config)
    critic = IdentityCritic(llm, model_name=args.critic_model, other_model_name=args.other_model_name)

    conditions = [CONDITION_MAP[c] for c in args.conditions]
    solutions_map = {s.problem_id: s for s in solutions}

    # Set up logging
    solver_model = solutions[0].solver_model if solutions else "unknown"
    exp_name = f"identity_swap_{args.critic_model}_on_{solver_model}"
    logger = ResultLogger(args.log_dir, exp_name)

    # Run critique for each condition
    all_critiques = {c.value: [] for c in conditions}

    for condition in conditions:
        print(f"\n--- Condition: {condition.value} ---")
        for problem, solution in tqdm(zip(problems, solutions), total=len(problems),
                                       desc=condition.value):
            result = critic.critique(problem, solution, condition)
            logger.log_critique(result)
            all_critiques[condition.value].append(result)

    # Compute and display metrics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<20} {'Det. Acc':>10} {'TPR':>10} {'FPR':>10} {'Confidence':>12}")
    print("-" * 62)

    for condition in conditions:
        metrics = compute_condition_metrics(
            all_critiques[condition.value], solutions_map
        )
        logger.log_metrics(asdict(metrics))
        print(f"{condition.value:<20} {metrics.detection_accuracy:>10.3f} "
              f"{metrics.true_positive_rate:>10.3f} {metrics.false_positive_rate:>10.3f} "
              f"{metrics.mean_confidence:>12.3f}")

    print(f"\nResults saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
