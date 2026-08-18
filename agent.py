from prompts import SYSTEM_PROMPT

class Agent:
    def __init__(self, brain, environment, max_steps=20):
        self.brain = brain
        self.environment = environment
        self.max_steps = max_steps

    def build_prompt(self, state):
        commands = "\n".join(
            f"- {command}"
            for command in state.admissible_commands
        )

        prompt = f"""
{SYSTEM_PROMPT}

OBJECTIVE:
{state.objective}

CURRENT OBSERVATION:
{state.feedback}

INVENTORY:
{state.inventory}

AVAILABLE COMMANDS:
{commands}

COMMAND:
"""

        return prompt

    def run(self):
        state = self.environment.reset()

        print("\nOBJECTIVE:")
        print(state.objective)

        print("\nSTARTING OBSERVATION:")
        print(state.feedback)

        for step_number in range(1, self.max_steps + 1):

            print("\n" + "=" * 60)
            print(f"AGENT STEP {step_number}")
            print("=" * 60)

            print(f"Location: {self.environment.get_location()}")

            prompt = self.build_prompt(state)

            action = self.brain.generate(prompt)

            print(f"Qwen chose: {action}")

            if action not in state.admissible_commands:
                print("Invalid model response.")
                print("Available commands were:")

                for command in state.admissible_commands:
                    print(f"  - {command}")

                continue

            state, reward, done = self.environment.step(action)

            print("\nTEXTWORLD RESPONSE:")
            print(state.feedback)

            print(f"Reward: {reward}")
            print(f"Score: {state.score}")

            if done:
                if state.won:
                    print("\n*** AGENT WON ***")
                elif state.lost:
                    print("\n*** AGENT LOST ***")
                else:
                    print("\n*** EPISODE ENDED ***")

                return

        print("\nMaximum number of steps reached.")
