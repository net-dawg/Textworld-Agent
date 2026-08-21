from environment import TextWorldEnvironment
from brain import OllamaBrain
from agent import Agent


GAME_FILE = "games/manor_mystery.z8"


def main():
    environment = TextWorldEnvironment(GAME_FILE)

    brain = OllamaBrain(
        model="qwen3:8b"
    )

    agent = Agent(
        brain=brain,
        environment=environment,
        max_steps=100,
    )

    try:
        agent.run()

    finally:
        environment.close()


if __name__ == "__main__":
    main()
