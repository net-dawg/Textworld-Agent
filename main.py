from pathlib import Path

try:
    from .agent import Agent
    from .brain import OllamaBrain
    from .environment import TextWorldEnvironment
except ImportError:
    from agent import Agent
    from brain import OllamaBrain
    from environment import TextWorldEnvironment


PROJECT_DIRECTORY = Path(__file__).resolve().parent
GAME_FILE = PROJECT_DIRECTORY / "games" / "manor_mystery.z8"
PROMPT_LOG_DIRECTORY = PROJECT_DIRECTORY / "prompt_logs"
MODEL = "qwen3:8b"
MAX_MOVES = 100


def main() -> None:
    environment = TextWorldEnvironment(GAME_FILE)
    brain = OllamaBrain(model=MODEL)
    agent = Agent(
        environment=environment,
        brain=brain,
        max_moves=MAX_MOVES,
        prompt_log_directory=PROMPT_LOG_DIRECTORY,
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
