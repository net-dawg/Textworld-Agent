import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory

from V2.agent import Agent
from V2.brain import Decision
from V2.environment import GameState, StepResult


def make_state(**changes):
    state = GameState(
        objective="Leave the room.",
        location="Hall",
        description="-= Hall =-\nA plain hall.",
        feedback="No action has been taken yet.",
        inventory="You are carrying nothing.",
        admissible_commands=("go west",),
        score=0,
        moves=0,
        won=False,
        lost=False,
    )
    return replace(state, **changes)


class FakeEnvironment:
    def __init__(self):
        self.initial = make_state()

    def reset(self):
        return self.initial

    def step(self, command):
        state = make_state(
            location="Outside",
            description="-= Outside =-\nYou made it.",
            feedback="You leave the room.",
            admissible_commands=(),
            score=1,
            moves=1,
            won=True,
        )
        return StepResult(state=state, reward=1, done=True)


class FakeBrain:
    def decide(self, state, memory, correction=None):
        return Decision(
            goal="Leave the Hall",
            action="go west",
            reason="The west exit is available",
        )


class AgentTests(unittest.TestCase):
    def test_agent_connects_plan_action_environment_and_memory(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        agent = Agent(
            FakeEnvironment(),
            FakeBrain(),
            max_moves=100,
            prompt_log_directory=temporary_directory.name,
        )

        result = agent.run()

        self.assertTrue(result.won)
        self.assertEqual(result.moves, 1)
        self.assertEqual(len(agent.memory.action_history), 1)
        self.assertEqual(
            agent.memory.known_routes[("Hall", "go west")],
            "Outside",
        )
        self.assertTrue(agent.prompt_log_path.exists())


if __name__ == "__main__":
    unittest.main()
