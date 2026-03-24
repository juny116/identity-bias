"""Step 1: Generate solutions for problems.

Usage:
    python experiments/run_solver.py --model gpt-4o --dataset gsm8k --n-samples 100
    python experiments/run_solver.py --model claude-sonnet --dataset math --n-samples 300
    python experiments/run_solver.py --model llama-70b --dataset gsm8k --n-samples 500 --base-url http://localhost:8000/v1
"""

import argparse
from tqdm import tqdm

from identity_bias.config import (
    Dataset, LLMProvider,
    get_openai_config, get_anthropic_config, get_vllm_config,
)
from identity_bias.llm import create_llm
from identity_bias.data import load_dataset_problems
from identity_bias.solver.cot_solver import CoTSolver
from identity_bias.logging.result_logger import ResultLogger


MODEL_PRESETS = {
    "gpt-4o": lambda: get_openai_config("gpt-4o"),
    "gpt-4o-mini": lambda: get_openai_config("gpt-4o-mini"),
    "claude-sonnet": lambda: get_anthropic_config("claude-sonnet-4-20250514"),
    "claude-haiku": lambda: get_anthropic_config("claude-haiku-4-5-20251001"),
    "llama-70b": lambda base_url: get_vllm_config("meta-llama/Llama-3.1-70B-Instruct", base_url),
    "llama-8b": lambda base_url: get_vllm_config("meta-llama/Llama-3.1-8B-Instruct", base_url),
}


def main():
    parser = argparse.ArgumentParser(description="Generate solutions for reasoning problems")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--dataset", type=str, required=True, choices=[d.value for d in Dataset])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="Base URL for vLLM server")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    # Build LLM config
    dataset = Dataset(args.dataset)
    preset = MODEL_PRESETS[args.model]

    if args.model.startswith("llama"):
        config = preset(args.base_url)
    else:
        config = preset()

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
    for problem in tqdm(problems, desc="Solving"):
        solution = solver.solve(problem, dataset)
        logger.log_solution(problem, solution)
        if solution.is_correct:
            correct += 1
        total += 1

    print(f"\nResults: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Solutions saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
