"""Step 1: Generate solutions for problems using solver models.

Usage:
    python experiments/run_solver.py --model qwen3 --dataset math --n-samples 100
    python experiments/run_solver.py --model gpt-o3 --dataset gpqa --n-samples 80
    python experiments/run_solver.py --model gemini-flash --dataset bbh --n-samples 120
    python experiments/run_solver.py --model claude-sonnet --dataset math --n-samples 100
"""

import argparse
from tqdm import tqdm

from identity_bias.config import Dataset, MODEL_PRESETS
from identity_bias.llm import create_llm
from identity_bias.data import load_dataset_problems
from identity_bias.solver.cot_solver import CoTSolver
from identity_bias.logging.result_logger import ResultLogger


def main():
    parser = argparse.ArgumentParser(description="Generate solutions for reasoning problems")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "gpt-o3", "gemini-flash", "claude-sonnet"])
    parser.add_argument("--dataset", type=str, required=True,
                        choices=[d.value for d in Dataset])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="Base URL for vLLM server")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    preset_fn = MODEL_PRESETS[args.model]
    config = preset_fn(base_url=args.base_url)
    config.temperature = args.temperature

    # Load problems
    print(f"Loading {args.n_samples} problems from {args.dataset}...")
    problems = load_dataset_problems(dataset, n_samples=args.n_samples, seed=args.seed)
    print(f"Loaded {len(problems)} problems")

    # Create solver
    llm = create_llm(config)
    solver = CoTSolver(llm, model_name=args.model)

    # Set up logging
    exp_name = f"solutions_{args.model}_{args.dataset}"
    logger = ResultLogger(args.log_dir, exp_name)

    # Solve problems
    correct = 0
    total = 0
    for problem in tqdm(problems, desc=f"Solving ({args.model})"):
        solution = solver.solve(problem, dataset)
        logger.log_solution(problem, solution)
        if solution.is_correct:
            correct += 1
        total += 1

    print(f"\nResults: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Solutions saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
