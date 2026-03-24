"""Chain-of-thought solver for generating solutions."""

import re
import json

from identity_bias.config import Dataset
from identity_bias.data.base import Problem, Solution
from identity_bias.data import check_answer
from identity_bias.llm.base import BaseLLM


SOLVER_SYSTEM_PROMPT = """You are a careful problem solver. Solve the given problem step by step, showing your complete chain of thought.

Format your response as follows:
1. Show your step-by-step reasoning
2. End with your final answer on a new line in the format: **Final Answer: <answer>**"""


SOLVER_USER_PROMPT = """Solve the following problem step by step.

Problem: {question}"""


class CoTSolver:
    """Chain-of-thought solver that generates step-by-step solutions."""

    def __init__(self, llm: BaseLLM, model_name: str = "unknown"):
        self.llm = llm
        self.model_name = model_name

    def solve(self, problem: Problem, dataset: Dataset) -> Solution:
        """Generate a chain-of-thought solution for the given problem."""
        messages = [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": SOLVER_USER_PROMPT.format(question=problem.question)},
        ]

        response = self.llm.generate(messages)
        cot = response.text

        # Extract final answer
        final_answer = self._extract_answer(cot)

        # Check correctness
        is_correct = check_answer(dataset, final_answer, problem.ground_truth)

        return Solution(
            problem_id=problem.id,
            chain_of_thought=cot,
            final_answer=final_answer,
            is_correct=is_correct,
            solver_model=self.model_name,
            metadata={
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            },
        )

    def _extract_answer(self, text: str) -> str:
        """Extract the final answer from the CoT response."""
        # Look for **Final Answer: ...** pattern
        match = re.search(r"\*\*Final Answer:\s*(.+?)\*\*", text)
        if match:
            return match.group(1).strip()

        # Fallback: look for "Final Answer:" without bold
        match = re.search(r"Final Answer:\s*(.+)", text)
        if match:
            return match.group(1).strip()

        # Fallback: look for boxed answer
        match = re.search(r"\\boxed\{(.+?)\}", text)
        if match:
            return match.group(1).strip()

        # Last resort: return the last line
        lines = text.strip().split("\n")
        return lines[-1].strip() if lines else ""

    def solve_batch(self, problems: list[Problem], dataset: Dataset) -> list[Solution]:
        """Solve a batch of problems sequentially."""
        solutions = []
        for problem in problems:
            solution = self.solve(problem, dataset)
            solutions.append(solution)
        return solutions
