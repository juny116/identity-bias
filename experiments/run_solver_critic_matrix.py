"""Step 3: Run Solver-Critic Matrix experiment (Experiment 2).

Each model evaluates its own and others' solutions.
Produces a matrix of solver x critic performance.

Usage:
    python experiments/run_solver_critic_matrix.py \
        --solutions-dir logs/ \
        --critic-models qwen3 gpt-o3 gemini-flash claude-sonnet \
        --dataset math
"""

import argparse
import glob
from tqdm import tqdm
from dataclasses import asdict

from identity_bias.config import IdentityCondition, Dataset, MODEL_PRESETS
from identity_bias.llm import create_llm
from identity_bias.data import check_answer
from identity_bias.critic.identity_critic import IdentityCritic
from identity_bias.evaluation.metrics import compute_condition_metrics
from identity_bias.logging.result_logger import ResultLogger


def main():
    parser = argparse.ArgumentParser(description="Run solver-critic matrix experiment")
    parser.add_argument("--solutions-dir", type=str, default="logs",
                        help="Directory containing solution JSONL files")
    parser.add_argument("--critic-models", nargs="+", required=True,
                        choices=["qwen3", "gpt-o3", "gemini-flash", "claude-sonnet"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=[d.value for d in Dataset])
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    check_fn = lambda pred, gt: check_answer(dataset, pred, gt)

    # Discover solution files for each solver model
    solver_files = {}
    for model in args.critic_models:
        pattern = f"{args.solutions_dir}/solutions_{model}_{args.dataset}_*.jsonl"
        matches = sorted(glob.glob(pattern))
        if matches:
            solver_files[model] = matches[-1]  # Use most recent
            print(f"Found solutions for {model}: {solver_files[model]}")
        else:
            print(f"WARNING: No solution file found for {model} ({pattern})")

    if not solver_files:
        print("ERROR: No solution files found. Run run_solver.py first.")
        return

    # Build the matrix: for each (solver, critic) pair, run anonymous critique
    results_matrix = {}  # (solver, critic) -> ConditionMetrics

    for critic_name in args.critic_models:
        config = MODEL_PRESETS[critic_name](base_url=args.base_url)
        llm = create_llm(config)
        critic = IdentityCritic(llm, model_name=critic_name)

        for solver_name, sol_file in solver_files.items():
            print(f"\n--- Critic: {critic_name} | Solver: {solver_name} ---")
            problems, solutions = ResultLogger.load_solutions(sol_file)
            ground_truths = {p.id: p.ground_truth for p in problems}

            # Use SELF if critic==solver, ANONYMOUS otherwise
            if critic_name == solver_name:
                condition = IdentityCondition.SELF
            else:
                condition = IdentityCondition.ANONYMOUS

            exp_name = f"matrix_{critic_name}_on_{solver_name}_{args.dataset}"
            logger = ResultLogger(args.log_dir, exp_name)

            critiques = []
            for problem, solution in tqdm(zip(problems, solutions), total=len(problems),
                                           desc=f"{critic_name}→{solver_name}"):
                result = critic.critique(problem, solution, condition)
                logger.log_critique(result)
                critiques.append(result)

            metrics = compute_condition_metrics(critiques, check_fn, ground_truths)
            logger.log_metrics(asdict(metrics))
            results_matrix[(solver_name, critic_name)] = metrics

    # Display matrix
    solver_names = list(solver_files.keys())
    critic_names = args.critic_models

    print("\n" + "=" * 90)
    print("SOLVER-CRITIC MATRIX: Detection Accuracy")
    print("=" * 90)
    header = f"{'Solver \\ Critic':<18}" + "".join(f"{c:>15}" for c in critic_names)
    print(header)
    print("-" * 90)

    for solver in solver_names:
        row = f"{solver:<18}"
        for critic in critic_names:
            key = (solver, critic)
            if key in results_matrix:
                m = results_matrix[key]
                marker = " *" if solver == critic else ""
                row += f"{m.detection_accuracy:>13.3f}{marker}"
            else:
                row += f"{'N/A':>15}"
        print(row)

    print("\n* = self-critique (diagonal)")

    # Show self vs cross gap per model
    print("\n--- Self vs Cross Gap (per model) ---")
    for model in solver_names:
        if model not in critic_names:
            continue
        self_key = (model, model)
        if self_key not in results_matrix:
            continue
        self_acc = results_matrix[self_key].detection_accuracy

        cross_accs = []
        for critic in critic_names:
            if critic != model and (model, critic) in results_matrix:
                cross_accs.append(results_matrix[(model, critic)].detection_accuracy)

        if cross_accs:
            avg_cross = sum(cross_accs) / len(cross_accs)
            gap = avg_cross - self_acc
            print(f"  {model}: self={self_acc:.3f}, avg_cross={avg_cross:.3f}, gap={gap:+.3f}")


if __name__ == "__main__":
    main()
