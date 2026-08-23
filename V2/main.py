from pathlib import Path

try:
    from .agent import Agent
    from .brain import OllamaBrain
    from .environment import TextWorldEnvironment
except ImportError:
    from agent import Agent
    from brain import OllamaBrain
    from environment import TextWorldEnvironment


GAME_FILE = Path(__file__).resolve().parent.parent / "games" / "manor_mystery.z8"
MODEL = "qwen3:8b"
MAX_MOVES = 100


def main() -> None:
    environment = TextWorldEnvironment(GAME_FILE)
    brain = OllamaBrain(model=MODEL)
    agent = Agent(
        environment=environment,
        brain=brain,
        max_moves=MAX_MOVES,
    )

    try:
        result = agent.run()
        print("\nRUN FINISHED:")
        print(f"Reason: {result.reason}")
        print(f"Won: {result.won}")
        print(f"Score: {result.score}")
        print(f"Moves: {result.moves}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
