"""Step 3: Run Context Separation experiment.

Tests whether context (same session vs new session) affects critique quality
independently of identity labeling.

Usage:
    python experiments/run_context_separation.py \
        --critic-model gpt-4o \
        --solutions-file logs/solutions_gpt4o_gsm8k_*.jsonl \
        --conditions same_session new_session new_session_anonymous
"""

import argparse
from tqdm import tqdm
from dataclasses import asdict

from identity_bias.config import (
    ContextCondition, IdentityCondition,
    get_openai_config, get_anthropic_config, get_vllm_config,
)
from identity_bias.llm import create_llm
from identity_bias.data.base import Problem, Solution
from identity_bias.critic.identity_critic import IdentityCritic
from identity_bias.critic.prompts import build_critic_messages, CRITIC_SYSTEM_PROMPT
from identity_bias.evaluation.metrics import compute_condition_metrics
from identity_bias.logging.result_logger import ResultLogger


MODEL_PRESETS = {
    "gpt-4o": lambda: get_openai_config("gpt-4o"),
    "gpt-4o-mini": lambda: get_openai_config("gpt-4o-mini"),
    "claude-sonnet": lambda: get_anthropic_config("claude-sonnet-4-20250514"),
    "llama-70b": lambda base_url="http://localhost:8000/v1": get_vllm_config(
        "meta-llama/Llama-3.1-70B-Instruct", base_url
    ),
}

CONTEXT_MAP = {c.value: c for c in ContextCondition}


def build_same_session_messages(
    problem: Problem,
    solution: Solution,
) -> list[dict[str, str]]:
    """Build messages that simulate same-session context.

    Includes the original generation as assistant turn, then asks for review.
    """
    return [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        # Simulate original generation context
        {"role": "user", "content": f"Solve the following problem step by step.\n\nProblem: {problem.question}"},
        {"role": "assistant", "content": solution.chain_of_thought},
        # Now ask for review
        {"role": "user", "content": (
            "Please review your solution above. Check each step carefully and determine "
            "if there are any errors. Respond with your assessment in the required JSON format."
        )},
    ]


def main():
    parser = argparse.ArgumentParser(description="Run context separation experiment")
    parser.add_argument("--critic-model", type=str, required=True, choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--solutions-file", type=str, required=True)
    parser.add_argument("--conditions", nargs="+",
                        default=["same_session", "new_session", "new_session_anonymous"],
                        choices=list(CONTEXT_MAP.keys()))
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    # Load solutions
    print(f"Loading solutions from {args.solutions_file}...")
    problems, solutions = ResultLogger.load_solutions(args.solutions_file)
    print(f"Loaded {len(problems)} problems")

    # Create LLM
    preset = MODEL_PRESETS[args.critic_model]
    if args.critic_model.startswith("llama"):
        config = preset(args.base_url)
    else:
        config = preset()
    llm = create_llm(config)
    critic = IdentityCritic(llm, model_name=args.critic_model)

    conditions = [CONTEXT_MAP[c] for c in args.conditions]
    solutions_map = {s.problem_id: s for s in solutions}

    exp_name = f"context_sep_{args.critic_model}"
    logger = ResultLogger(args.log_dir, exp_name)

    all_critiques = {c.value: [] for c in conditions}

    for condition in conditions:
        print(f"\n--- Condition: {condition.value} ---")
        for problem, solution in tqdm(zip(problems, solutions), total=len(problems),
                                       desc=condition.value):
            if condition == ContextCondition.SAME_SESSION:
                # Same session: solution appears as assistant's own output
                messages = build_same_session_messages(problem, solution)
                response = llm.generate(messages)
                result = critic._parse_response(response.text)
                from identity_bias.critic.identity_critic import CritiqueResult
                critique = CritiqueResult(
                    problem_id=problem.id,
                    identity_condition=condition.value,
                    has_error=result.get("has_error", False),
                    error_step=result.get("error_step"),
                    error_description=result.get("error_description"),
                    corrected_answer=result.get("corrected_answer"),
                    confidence=result.get("confidence", 0.5),
                    raw_response=response.text,
                    critic_model=args.critic_model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                )
            elif condition == ContextCondition.NEW_SESSION:
                # New session with self label
                critique = critic.critique(problem, solution, IdentityCondition.SELF)
                critique.identity_condition = condition.value
            elif condition == ContextCondition.NEW_SESSION_ANONYMOUS:
                # New session with anonymous label
                critique = critic.critique(problem, solution, IdentityCondition.ANONYMOUS)
                critique.identity_condition = condition.value
            else:
                critique = critic.critique(problem, solution, IdentityCondition.ANONYMOUS)
                critique.identity_condition = condition.value

            logger.log_critique(critique)
            all_critiques[condition.value].append(critique)

    # Results summary
    print("\n" + "=" * 80)
    print("CONTEXT SEPARATION RESULTS")
    print("=" * 80)
    print(f"{'Condition':<25} {'Det. Acc':>10} {'TPR':>10} {'FPR':>10} {'Confidence':>12}")
    print("-" * 67)

    for condition in conditions:
        metrics = compute_condition_metrics(
            all_critiques[condition.value], solutions_map
        )
        logger.log_metrics(asdict(metrics))
        print(f"{condition.value:<25} {metrics.detection_accuracy:>10.3f} "
              f"{metrics.true_positive_rate:>10.3f} {metrics.false_positive_rate:>10.3f} "
              f"{metrics.mean_confidence:>12.3f}")

    print(f"\nResults saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
