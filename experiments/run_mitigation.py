"""Step 4: Run Mitigation experiment.

Tests whether simple protocol changes (anonymization, new session) improve
critique quality compared to the self-critique baseline.

Usage:
    python experiments/run_mitigation.py \
        --critic-model gpt-4o \
        --solutions-file logs/solutions_gpt4o_gsm8k_*.jsonl
"""

import argparse
from tqdm import tqdm
from dataclasses import asdict

from identity_bias.config import (
    IdentityCondition,
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
    "llama-70b": lambda base_url="http://localhost:8000/v1": get_vllm_config(
        "meta-llama/Llama-3.1-70B-Instruct", base_url
    ),
}

MITIGATIONS = {
    "baseline_self": IdentityCondition.SELF,
    "anonymous": IdentityCondition.ANONYMOUS,
    "attributed_weak": IdentityCondition.WEAK_MODEL,
    "attributed_human": IdentityCondition.HUMAN,
}


def main():
    parser = argparse.ArgumentParser(description="Run mitigation experiment")
    parser.add_argument("--critic-model", type=str, required=True, choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--solutions-file", type=str, required=True)
    parser.add_argument("--mitigations", nargs="+",
                        default=list(MITIGATIONS.keys()),
                        choices=list(MITIGATIONS.keys()))
    parser.add_argument("--cross-model", type=str, default=None,
                        help="Additional model for cross-model mitigation")
    parser.add_argument("--cross-model-base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    # Load solutions
    print(f"Loading solutions from {args.solutions_file}...")
    problems, solutions = ResultLogger.load_solutions(args.solutions_file)
    print(f"Loaded {len(problems)} problems")

    # Create primary critic
    preset = MODEL_PRESETS[args.critic_model]
    if args.critic_model.startswith("llama"):
        config = preset(args.base_url)
    else:
        config = preset()
    llm = create_llm(config)
    critic = IdentityCritic(llm, model_name=args.critic_model)

    solutions_map = {s.problem_id: s for s in solutions}
    exp_name = f"mitigation_{args.critic_model}"
    logger = ResultLogger(args.log_dir, exp_name)

    all_critiques = {}

    for mitigation_name in args.mitigations:
        condition = MITIGATIONS[mitigation_name]
        print(f"\n--- Mitigation: {mitigation_name} ({condition.value}) ---")

        critiques = []
        for problem, solution in tqdm(zip(problems, solutions), total=len(problems),
                                       desc=mitigation_name):
            result = critic.critique(problem, solution, condition)
            result.identity_condition = mitigation_name
            logger.log_critique(result)
            critiques.append(result)

        all_critiques[mitigation_name] = critiques

    # Cross-model mitigation if specified
    if args.cross_model and args.cross_model in MODEL_PRESETS:
        print(f"\n--- Mitigation: cross_model ({args.cross_model}) ---")
        cross_preset = MODEL_PRESETS[args.cross_model]
        if args.cross_model.startswith("llama"):
            cross_config = cross_preset(args.cross_model_base_url)
        else:
            cross_config = cross_preset()
        cross_llm = create_llm(cross_config)
        cross_critic = IdentityCritic(cross_llm, model_name=args.cross_model)

        critiques = []
        for problem, solution in tqdm(zip(problems, solutions), total=len(problems),
                                       desc="cross_model"):
            result = cross_critic.critique(problem, solution, IdentityCondition.ANONYMOUS)
            result.identity_condition = "cross_model"
            logger.log_critique(result)
            critiques.append(result)

        all_critiques["cross_model"] = critiques

    # Results summary
    print("\n" + "=" * 80)
    print("MITIGATION RESULTS")
    print("=" * 80)
    print(f"{'Mitigation':<20} {'Det. Acc':>10} {'TPR':>10} {'FPR':>10} {'Confidence':>12}")
    print("-" * 62)

    for name, critiques in all_critiques.items():
        metrics = compute_condition_metrics(critiques, solutions_map)
        logger.log_metrics(asdict(metrics))
        print(f"{name:<20} {metrics.detection_accuracy:>10.3f} "
              f"{metrics.true_positive_rate:>10.3f} {metrics.false_positive_rate:>10.3f} "
              f"{metrics.mean_confidence:>12.3f}")

    print(f"\nResults saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
