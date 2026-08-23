import re
from dataclasses import dataclass
from pathlib import Path

import textworld


@dataclass(frozen=True)
class GameState:
    """Authoritative information reported by TextWorld for the current turn."""

    objective: str
    location: str | None
    description: str
    feedback: str
    inventory: str
    admissible_commands: tuple[str, ...]
    score: int
    moves: int
    won: bool
    lost: bool


@dataclass(frozen=True)
class StepResult:
    """The new game state plus the outcome of the executed command."""

    state: GameState
    reward: int
    done: bool


class TextWorldEnvironment:
    """Runs TextWorld and converts its raw state into a clean snapshot."""

    def __init__(self, game_file: str | Path):
        self.game_file = Path(game_file)
        self._environment = None
        self._state = None

    def start(self) -> None:
        if self._environment is not None:
            return

        requested_information = textworld.EnvInfos(
            description=True,
            inventory=True,
            objective=True,
            admissible_commands=True,
            score=True,
            moves=True,
            won=True,
            lost=True,
        )

        self._environment = textworld.start(
            str(self.game_file),
            request_infos=requested_information,
        )

    def reset(self) -> GameState:
        self.start()
        self._state = self._environment.reset()
        return self._snapshot(self._state)

    def step(self, command: str) -> StepResult:
        if self._environment is None or self._state is None:
            raise RuntimeError("Reset the environment before taking an action.")

        if command not in self._state.admissible_commands:
            raise ValueError(f"Command is not currently admissible: {command}")

        self._state, reward, done = self._environment.step(command)

        return StepResult(
            state=self._snapshot(self._state),
            reward=reward,
            done=done,
        )

    def close(self) -> None:
        if self._environment is not None:
            self._environment.close()

        self._environment = None
        self._state = None

    def _snapshot(self, state) -> GameState:
        return GameState(
            objective=state.objective.strip(),
            location=self._extract_location(state.description),
            description=state.description.strip(),
            feedback=self._clean_feedback(state.feedback, state.moves),
            inventory=state.inventory.strip(),
            admissible_commands=tuple(sorted(state.admissible_commands)),
            score=state.score,
            moves=state.moves,
            won=state.won,
            lost=state.lost,
        )

    @staticmethod
    def _extract_location(description: str) -> str | None:
        match = re.search(r"-=\s*(.*?)\s*=-", description)
        return match.group(1) if match else None

    @staticmethod
    def _clean_feedback(feedback: str, moves: int) -> str:
        if moves == 0:
            return "No action has been taken yet."

        cleaned = re.sub(
            r"\n>\s*-=.*$",
            "",
            feedback,
            flags=re.DOTALL,
        )
        return cleaned.strip()
