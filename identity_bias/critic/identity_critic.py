"""Identity-labeled critic for evaluating solutions."""

import json
import re
from dataclasses import dataclass

from identity_bias.config import IdentityCondition
from identity_bias.data.base import Problem, Solution
from identity_bias.critic.prompts import build_critic_messages
from identity_bias.llm.base import BaseLLM


@dataclass
class CritiqueResult:
    """Result of a critic's evaluation."""
    problem_id: str
    identity_condition: str
    has_error: bool
    error_step: str | None
    error_description: str | None
    corrected_answer: str | None
    confidence: float
    raw_response: str
    critic_model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class IdentityCritic:
    """Critic that evaluates solutions with identity-conditioned prompts."""

    def __init__(
        self,
        llm: BaseLLM,
        model_name: str = "unknown",
        other_model_name: str = "GPT-4o",
    ):
        self.llm = llm
        self.model_name = model_name
        self.other_model_name = other_model_name

    def critique(
        self,
        problem: Problem,
        solution: Solution,
        identity_condition: IdentityCondition,
    ) -> CritiqueResult:
        """Critique a solution under the given identity condition.

        Args:
            problem: The original problem.
            solution: The solution to critique.
            identity_condition: The authorship label to present.

        Returns:
            CritiqueResult with the critic's assessment.
        """
        messages = build_critic_messages(
            question=problem.question,
            chain_of_thought=solution.chain_of_thought,
            final_answer=solution.final_answer,
            identity_condition=identity_condition,
            other_model_name=self.other_model_name,
        )

        response = self.llm.generate(messages)
        parsed = self._parse_response(response.text)

        return CritiqueResult(
            problem_id=problem.id,
            identity_condition=identity_condition.value,
            has_error=parsed.get("has_error", False),
            error_step=parsed.get("error_step"),
            error_description=parsed.get("error_description"),
            corrected_answer=parsed.get("corrected_answer"),
            confidence=parsed.get("confidence", 0.5),
            raw_response=response.text,
            critic_model=self.model_name,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    def critique_all_conditions(
        self,
        problem: Problem,
        solution: Solution,
        conditions: list[IdentityCondition] | None = None,
    ) -> list[CritiqueResult]:
        """Critique a solution under all identity conditions."""
        if conditions is None:
            conditions = list(IdentityCondition)

        results = []
        for condition in conditions:
            result = self.critique(problem, solution, condition)
            results.append(result)
        return results

    def _parse_response(self, text: str) -> dict:
        """Parse the JSON response from the critic."""
        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code fence
        match = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object in the text
        match = re.search(r"\{[^{}]*\"has_error\"[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback: return defaults
        return {
            "has_error": False,
            "error_step": None,
            "error_description": None,
            "corrected_answer": None,
            "confidence": 0.5,
            "parse_error": True,
        }
