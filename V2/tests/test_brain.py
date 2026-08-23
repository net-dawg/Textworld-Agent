import unittest

from V2.brain import OllamaBrain
from V2.environment import GameState
from V2.memory import WorldMemory


def make_state():
    return GameState(
        objective="Find the treasure.",
        location="Hall",
        description="-= Hall =-\nA plain hall.",
        feedback="No action has been taken yet.",
        inventory="You are carrying nothing.",
        admissible_commands=("examine door", "go west"),
        score=0,
        moves=0,
        won=False,
        lost=False,
    )


class StubBrain(OllamaBrain):
    def __init__(self, response):
        super().__init__()
        self.response = response

    def _generate(self, prompt):
        return self.response


class BrainTests(unittest.TestCase):
    def test_valid_candidate_is_accepted(self):
        brain = StubBrain({
            "goal": "Explore west",
            "action": "go west",
            "reason": "The west route is unexplored.",
        })

        decision = brain.decide(make_state(), WorldMemory())

        self.assertEqual(decision.action, "go west")

    def test_non_candidate_is_rejected(self):
        brain = StubBrain({
            "goal": "Do something unsupported",
            "action": "invent command",
            "reason": "No verified reason.",
        })

        with self.assertRaises(ValueError):
            brain.decide(make_state(), WorldMemory())

    def test_exact_prompt_is_sent_to_callback(self):
        prompts = []
        brain = StubBrain({
            "goal": "Explore west",
            "action": "go west",
            "reason": "The west route is unexplored.",
        })
        brain.set_prompt_callback(prompts.append)

        brain.decide(make_state(), WorldMemory())

        self.assertEqual(len(prompts), 1)
        self.assertIn("OBJECTIVE:\nFind the treasure.", prompts[0])


if __name__ == "__main__":
    unittest.main()
