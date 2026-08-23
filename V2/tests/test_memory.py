import unittest
from dataclasses import replace

from V2.environment import GameState, StepResult
from V2.memory import OutcomeKind, WorldMemory


def make_state(**changes):
    state = GameState(
        objective="Find the sunstone and return it to the ceremonial plinth.",
        location="Hall",
        description="-= Hall =-\nA plain hall.",
        feedback="No action has been taken yet.",
        inventory="You are carrying nothing.",
        admissible_commands=("examine journal", "go west"),
        score=0,
        moves=0,
        won=False,
        lost=False,
    )
    return replace(state, **changes)


class WorldMemoryTests(unittest.TestCase):
    def test_examination_preserves_exact_evidence_and_is_not_repeated(self):
        memory = WorldMemory()
        before = make_state()
        after = make_state(
            feedback="Most pages are unreadable.",
            moves=1,
        )

        record = memory.record_action(
            "examine journal",
            before,
            StepResult(after, reward=0, done=False),
        )

        self.assertEqual(record.outcome, OutcomeKind.OBSERVATION)
        self.assertEqual(
            memory.findings[("Hall", "examine journal")],
            "Most pages are unreadable.",
        )
        self.assertNotIn("examine journal", memory.candidate_actions(after))

    def test_navigation_remembers_both_directions(self):
        memory = WorldMemory()
        before = make_state()
        after = make_state(
            location="Library",
            admissible_commands=("go east",),
            moves=1,
        )

        memory.record_action(
            "go west",
            before,
            StepResult(after, reward=0, done=False),
        )

        self.assertEqual(memory.routes[("Hall", "go west")], "Library")
        self.assertEqual(memory.routes[("Library", "go east")], "Hall")

    def test_unrelated_items_cannot_be_stored(self):
        memory = WorldMemory()
        state = make_state(
            inventory="You are carrying a faded journal and a brass key.",
            admissible_commands=(
                "insert faded journal into plinth",
                "insert brass key into desk",
                "go west",
            ),
        )

        self.assertEqual(memory.candidate_actions(state), ("go west",))

    def test_objective_item_can_be_placed(self):
        memory = WorldMemory()
        state = make_state(
            inventory="You are carrying the sunstone.",
            admissible_commands=("insert sunstone into ceremonial plinth",),
        )

        self.assertEqual(
            memory.candidate_actions(state),
            ("insert sunstone into ceremonial plinth",),
        )

    def test_objective_item_cannot_be_returned_to_wrong_container(self):
        memory = WorldMemory()
        state = make_state(
            location="Hidden Vault",
            inventory="You are carrying the sunstone.",
            admissible_commands=(
                "insert sunstone into stone coffer",
                "go west",
            ),
        )

        self.assertEqual(memory.candidate_actions(state), ("go west",))

    def test_unlocking_with_carried_key_outranks_backtracking(self):
        memory = WorldMemory()
        state = make_state(
            inventory="You are carrying a brass key.",
            admissible_commands=(
                "go west",
                "unlock carved oak door with brass key",
            ),
        )

        self.assertEqual(
            memory.candidate_actions(state),
            ("unlock carved oak door with brass key",),
        )

    def test_junction_avoids_route_just_backtracked_from(self):
        memory = WorldMemory()
        junction = make_state(
            location="Launderette",
            admissible_commands=("go east", "go north", "go south"),
        )
        study = make_state(
            location="Study",
            admissible_commands=("go west",),
            moves=1,
        )
        returned = replace(junction, moves=2)
        memory.observe_initial_state(junction)
        memory.record_action(
            "go east",
            junction,
            StepResult(study, reward=0, done=False),
        )
        memory.record_action(
            "go west",
            study,
            StepResult(returned, reward=0, done=False),
        )
        memory.routes[("Launderette", "go north")] = "Courtyard"
        memory.routes[("Launderette", "go south")] = "Great Hall"
        memory.route_traversals[("Launderette", "go north")] = 1
        memory.location_visits["Courtyard"] = 1
        memory.location_visits["Great Hall"] = 1

        self.assertEqual(memory.candidate_actions(returned), ("go south",))


if __name__ == "__main__":
    unittest.main()
