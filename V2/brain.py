import json
from collections.abc import Callable
from dataclasses import dataclass

import requests

try:
    from .environment import GameState
    from .memory import WorldMemory
    from .prompts import build_decision_prompt
except ImportError:
    from environment import GameState
    from memory import WorldMemory
    from prompts import build_decision_prompt


@dataclass(frozen=True)
class Decision:
    goal: str
    action: str
    reason: str


class OllamaBrain:
    """One model call chooses one grounded micro-plan and action."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        url: str = "http://127.0.0.1:11434/api/generate",
        timeout: int = 120,
        prompt_callback: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.prompt_callback = prompt_callback

    def set_prompt_callback(
        self,
        callback: Callable[[str], None] | None,
    ) -> None:
        self.prompt_callback = callback

    def decide(
        self,
        state: GameState,
        memory: WorldMemory,
        correction: str | None = None,
    ) -> Decision:
        candidates = memory.candidate_actions(state)
        prompt = build_decision_prompt(state, memory, correction)
        if self.prompt_callback:
            self.prompt_callback(prompt)
        data = self._generate(prompt)

        decision = Decision(
            goal=data.get("goal", "").strip(),
            action=data.get("action", "").strip(),
            reason=data.get("reason", "").strip(),
        )
        if not decision.goal or not decision.action or not decision.reason:
            raise ValueError("Model returned an incomplete decision.")
        if decision.action not in candidates:
            raise ValueError(
                f"Action is not a candidate: {decision.action}. "
                f"Candidates: {list(candidates)}"
            )
        return decision

    def _generate(self, prompt: str) -> dict:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "action": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["goal", "action", "reason"],
                    "additionalProperties": False,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return json.loads(response.json()["response"])
