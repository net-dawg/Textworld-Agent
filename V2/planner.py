from dataclasses import dataclass
from enum import Enum

try:
    from .memory import ActionRecord, OutcomeKind
except ImportError:
    from memory import ActionRecord, OutcomeKind


class ExpectedOutcome(str, Enum):
    LEARN = "learn something from one observation"
    MOVE = "move to another location"
    ACQUIRE = "add an item to inventory"
    CHANGE = "produce a useful world change"


@dataclass(frozen=True)
class Plan:
    """A one-action plan whose result can be checked by code."""

    goal: str
    action: str
    reason: str
    expected_outcome: ExpectedOutcome
    created_at_move: int


@dataclass(frozen=True)
class PlanResult:
    plan: Plan
    succeeded: bool
    evidence: str


class PlanManager:
    """Creates grounded micro-plans and verifies their actual outcomes."""

    def __init__(self):
        self.history: list[PlanResult] = []

    def create(self, goal: str, action: str, reason: str, move: int) -> Plan:
        goal = goal.strip()
        action = action.strip()
        reason = reason.strip()
        if not goal or not action or not reason:
            raise ValueError("A plan requires a goal, action, and reason.")

        return Plan(
            goal=goal,
            action=action,
            reason=reason,
            expected_outcome=self._expected_outcome(action),
            created_at_move=move,
        )

    def evaluate(self, plan: Plan, record: ActionRecord) -> PlanResult:
        expected = plan.expected_outcome

        if expected is ExpectedOutcome.LEARN:
            succeeded = bool(record.result)
        elif expected is ExpectedOutcome.MOVE:
            succeeded = (
                record.state_before.location != record.state_after.location
            )
        elif expected is ExpectedOutcome.ACQUIRE:
            succeeded = (
                record.state_before.inventory != record.state_after.inventory
            )
        else:
            succeeded = record.outcome in (
                OutcomeKind.WORLD_CHANGE,
                OutcomeKind.GOAL_EVENT,
            )

        evidence = record.result
        result = PlanResult(plan=plan, succeeded=succeeded, evidence=evidence)
        self.history.append(result)
        return result

    @staticmethod
    def _expected_outcome(action: str) -> ExpectedOutcome:
        verb = action.split(" ", 1)[0]
        if verb == "examine":
            return ExpectedOutcome.LEARN
        if verb == "go":
            return ExpectedOutcome.MOVE
        if verb == "take":
            return ExpectedOutcome.ACQUIRE
        return ExpectedOutcome.CHANGE
