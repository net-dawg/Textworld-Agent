from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from .brain import Decision, OllamaBrain
    from .environment import GameState, TextWorldEnvironment
    from .memory import WorldMemory
    from .planner import PlanManager
except ImportError:
    from brain import Decision, OllamaBrain
    from environment import GameState, TextWorldEnvironment
    from memory import WorldMemory
    from planner import PlanManager


@dataclass(frozen=True)
class RunResult:
    won: bool
    lost: bool
    moves: int
    score: int
    reason: str


class Agent:
    """Simple loop: observe, remember, plan one action, execute, verify."""

    DIVIDER = "-" * 60

    def __init__(
        self,
        environment: TextWorldEnvironment,
        brain: OllamaBrain,
        max_moves: int = 100,
        max_model_attempts: int = 3,
        prompt_log_directory: str | Path = "prompt_logs",
    ):
        self.environment = environment
        self.brain = brain
        self.max_moves = max_moves
        self.max_model_attempts = max_model_attempts
        self.prompt_log_directory = Path(prompt_log_directory)
        self.prompt_log_path: Path | None = None
        self.prompt_count = 0
        self.memory = WorldMemory()
        self.plans = PlanManager()

    def run(self) -> RunResult:
        state = self.environment.reset()
        self.memory = WorldMemory()
        self.plans = PlanManager()
        self.memory.observe_initial_state(state)
        self._start_prompt_log()

        print("OBJECTIVE:")
        print(f"{state.objective}\n")
        print(f"Prompt log: {self.prompt_log_path}\n")
        print(self.DIVIDER)

        while state.moves < self.max_moves and not state.won and not state.lost:
            candidates = self.memory.candidate_actions(state)
            if not candidates:
                return self._result(state, "No useful candidate actions remain.")

            decision = self._request_decision(state)
            plan = self.plans.create(
                goal=decision.goal,
                action=decision.action,
                reason=decision.reason,
                move=state.moves,
            )
            self._print_plan(plan)

            step_result = self.environment.step(plan.action)
            record = self.memory.record_action(plan.action, state, step_result)
            plan_result = self.plans.evaluate(plan, record)
            state = step_result.state

            self._print_result(state, plan_result.succeeded, record.result)

        if state.won:
            return self._result(state, "The game was won.")
        if state.lost:
            return self._result(state, "The game was lost.")
        return self._result(state, f"Maximum of {self.max_moves} moves reached.")

    def _request_decision(self, state: GameState) -> Decision:
        error = None
        for attempt in range(1, self.max_model_attempts + 1):
            try:
                return self.brain.decide(
                    state,
                    self.memory,
                    correction=str(error) if error else None,
                )
            except (ValueError, KeyError) as caught:
                error = caught
                print(
                    f"Invalid model decision "
                    f"({attempt}/{self.max_model_attempts}): {caught}"
                )
        raise RuntimeError("The model did not return a valid decision.") from error

    def _start_prompt_log(self) -> None:
        started_at = datetime.now(timezone.utc)
        run_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ")
        self.prompt_log_directory.mkdir(parents=True, exist_ok=True)
        self.prompt_log_path = (
            self.prompt_log_directory / f"prompts_{run_id}.txt"
        )
        self.prompt_count = 0
        self.prompt_log_path.write_text(
            "TEXTWORLD AGENT — EXACT MODEL PROMPTS\n"
            f"Run started: {started_at.isoformat()}\n",
            encoding="utf-8",
        )

        if hasattr(self.brain, "set_prompt_callback"):
            self.brain.set_prompt_callback(self._log_prompt)

    def _log_prompt(self, prompt: str) -> None:
        self.prompt_count += 1
        separator = "=" * 80
        with self.prompt_log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"\n\n{separator}\n"
                f"MODEL PROMPT {self.prompt_count}\n"
                f"{separator}\n\n"
                f"{prompt}\n"
            )

    @classmethod
    def _print_plan(cls, plan) -> None:
        print(f"\nPLAN FOR MOVE {plan.created_at_move + 1}:")
        print(f"Goal: {plan.goal}\n")
        print(f"Action: {plan.action}\n")
        print(f"Expected: {plan.expected_outcome.value}\n")
        print(f"Why: {plan.reason}\n")

    @classmethod
    def _print_result(cls, state: GameState, succeeded: bool, evidence: str) -> None:
        print(f"Result: {evidence}\n")
        print(f"Plan verified: {'yes' if succeeded else 'no'}\n")
        print(
            f"Location: {state.location} | Score: {state.score} | Moves: {state.moves}"
        )
        print(cls.DIVIDER)

    @staticmethod
    def _result(state: GameState, reason: str) -> RunResult:
        return RunResult(
            won=state.won,
            lost=state.lost,
            moves=state.moves,
            score=state.score,
            reason=reason,
        )
