import unittest
from types import SimpleNamespace

from agent import Agent


class FakeEnvironment:
    def get_location(self, state=None):
        return state.location


class ActionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(brain=None, environment=FakeEnvironment())

    def test_unrelated_state_change_does_not_release_locked_door(self):
        state = SimpleNamespace(
            location="Great Hall",
            admissible_commands=[
                "open carved oak door",
                "take book",
            ],
        )
        self.agent.remember_failed_action(
            "Great Hall",
            "open carved oak door",
            "You have to unlock the carved oak door with the brass key first.",
        )

        self.agent.resolve_failed_actions("take book", made_progress=True)

        self.assertEqual(
            self.agent.get_candidate_commands(state),
            ["take book"],
        )

    def test_unlocking_door_releases_failed_open_action(self):
        state = SimpleNamespace(
            location="Great Hall",
            admissible_commands=["open carved oak door"],
        )
        self.agent.remember_failed_action(
            "Great Hall",
            "open carved oak door",
            "You have to unlock the carved oak door with the brass key first.",
        )

        self.agent.resolve_failed_actions(
            "unlock carved oak door with brass key",
            made_progress=True,
        )

        self.assertEqual(
            self.agent.get_candidate_commands(state),
            ["open carved oak door"],
        )

    def test_failures_are_scoped_to_the_location(self):
        self.agent.remember_failed_action(
            "Great Hall",
            "open door",
            "The door is locked.",
        )
        other_room = SimpleNamespace(
            location="Library",
            admissible_commands=["open door"],
        )

        self.assertEqual(
            self.agent.get_candidate_commands(other_room),
            ["open door"],
        )


if __name__ == "__main__":
    unittest.main()
