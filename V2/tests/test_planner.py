import unittest
from dataclasses import replace

from V2.environment import GameState
from V2.memory import ActionRecord, OutcomeKind
from V2.planner import ExpectedOutcome, PlanManager


def make_state(**changes):
    state = GameState(
        objective="Find the treasure.",
        location="Hall",
        description="-= Hall =-",
        feedback="Nothing yet.",
        inventory="You are carrying nothing.",
        admissible_commands=("go west",),
        score=0,
        moves=0,
        won=False,
        lost=False,
    )
    return replace(state, **changes)


class PlanManagerTests(unittest.TestCase):
    def test_expected_outcome_is_derived_from_action(self):
        manager = PlanManager()

        plan = manager.create("Reach the Library", "go west", "Unexplored", 0)

        self.assertEqual(plan.expected_outcome, ExpectedOutcome.MOVE)

    def test_code_verifies_movement(self):
        manager = PlanManager()
        before = make_state()
        after = make_state(location="Library", moves=1)
        plan = manager.create("Reach the Library", "go west", "Unexplored", 0)
        record = ActionRecord(
            command="go west",
            state_before=before,
            state_after=after,
            result="You enter the Library.",
            outcome=OutcomeKind.NAVIGATION,
            reward=0,
            done=False,
        )

        result = manager.evaluate(plan, record)

        self.assertTrue(result.succeeded)


if __name__ == "__main__":
    unittest.main()
