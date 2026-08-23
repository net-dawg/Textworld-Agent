from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

try:
    from .environment import GameState, StepResult
except ImportError:
    from environment import GameState, StepResult


class OutcomeKind(str, Enum):
    OBSERVATION = "information observed"
    NO_EFFECT = "no effect"
    WORLD_CHANGE = "world changed"
    NAVIGATION = "location changed"
    CYCLE = "returned to an earlier state"
    GOAL_EVENT = "score or episode changed"


@dataclass(frozen=True)
class ActionRecord:
    command: str
    state_before: GameState
    state_after: GameState
    result: str
    outcome: OutcomeKind
    reward: int
    done: bool


@dataclass
class WorldMemory:
    """Verified observations and transitions; never model-generated guesses."""

    history: list[ActionRecord] = field(default_factory=list)
    visited_locations: set[str] = field(default_factory=set)
    location_visits: Counter = field(default_factory=Counter)
    routes: dict[tuple[str, str], str] = field(default_factory=dict)
    route_traversals: Counter = field(default_factory=Counter)
    findings: dict[tuple[str, str], str] = field(default_factory=dict)
    exhausted_observations: set[tuple[str, str]] = field(default_factory=set)
    ineffective_actions: set[tuple[tuple, str]] = field(default_factory=set)
    cyclic_actions: set[tuple[tuple, str]] = field(default_factory=set)
    state_visits: Counter = field(default_factory=Counter)

    @property
    def action_history(self) -> list[ActionRecord]:
        return self.history

    @property
    def known_routes(self) -> dict[tuple[str, str], str]:
        return self.routes

    def observe_initial_state(self, state: GameState) -> None:
        if state.location:
            self.visited_locations.add(state.location)
            self.location_visits[state.location] += 1
        self.state_visits[self.signature(state)] += 1

    def record_action(
        self,
        command: str,
        state_before: GameState,
        step_result: StepResult,
    ) -> ActionRecord:
        state_after = step_result.state
        before_key = self.signature(state_before)
        after_key = self.signature(state_after)
        location = state_before.location or "Unknown"
        first_observation = (location, command) not in self.findings

        if step_result.reward or step_result.done:
            outcome = OutcomeKind.GOAL_EVENT
        elif state_before.location != state_after.location:
            outcome = OutcomeKind.NAVIGATION
        elif after_key != before_key and self.state_visits[after_key]:
            outcome = OutcomeKind.CYCLE
        elif after_key != before_key:
            outcome = OutcomeKind.WORLD_CHANGE
        elif command.startswith("examine ") and first_observation:
            outcome = OutcomeKind.OBSERVATION
        else:
            outcome = OutcomeKind.NO_EFFECT

        record = ActionRecord(
            command=command,
            state_before=state_before,
            state_after=state_after,
            result=state_after.feedback,
            outcome=outcome,
            reward=step_result.reward,
            done=step_result.done,
        )
        self.history.append(record)

        if state_before.location:
            self.visited_locations.add(state_before.location)
        if state_after.location:
            self.visited_locations.add(state_after.location)
            if state_before.location != state_after.location:
                self.location_visits[state_after.location] += 1

        if state_before.location != state_after.location:
            self._remember_route(state_before, command, state_after)

        if command.startswith("examine "):
            self.findings[(location, command)] = record.result
            self.exhausted_observations.add((location, command))

        if outcome is OutcomeKind.NO_EFFECT:
            self.ineffective_actions.add((before_key, command))

        if outcome is OutcomeKind.CYCLE:
            self.cyclic_actions.add((before_key, command))
            self._remember_two_action_cycle(record)

        self.state_visits[after_key] += 1
        return record

    def candidate_actions(self, state: GameState) -> tuple[str, ...]:
        available = [
            command
            for command in state.admissible_commands
            if self._is_candidate(state, command)
        ]
        if not available:
            return ()

        ranked = sorted(available, key=lambda command: self._priority(state, command))
        best_priority = self._priority(state, ranked[0])[:-1]
        return tuple(
            command
            for command in ranked
            if self._priority(state, command)[:-1] == best_priority
        )

    def rejected_actions(self, state: GameState) -> tuple[str, ...]:
        candidates = set(self.candidate_actions(state))
        return tuple(
            command for command in state.admissible_commands if command not in candidates
        )

    def _is_candidate(self, state: GameState, command: str) -> bool:
        location = state.location or "Unknown"
        key = self.signature(state)

        if command in ("look", "inventory"):
            return False
        if (location, command) in self.exhausted_observations:
            return False
        if (key, command) in self.ineffective_actions:
            return False
        if (key, command) in self.cyclic_actions:
            return False

        verb = command.split(" ", 1)[0]
        if verb in ("close", "lock", "drop"):
            return self._action_required_by_objective(command, state.objective)
        if verb in ("insert", "put"):
            placement = self._placement(command)
            if not placement:
                return False
            item, destination = placement
            objective = state.objective.lower()
            return item in objective and destination in objective

        return True

    def _priority(self, state: GameState, command: str) -> tuple[int, int, int, int, str]:
        verb = command.split(" ", 1)[0]
        location = state.location or "Unknown"
        reversal_penalty = 0
        traversal_count = 0
        destination_visits = 0

        if verb == "unlock":
            rank = 0
        elif verb == "take" and self._take_target_is_relevant(command, state.objective):
            rank = 0
        elif verb in ("insert", "put"):
            rank = 0
        elif verb == "open":
            rank = 1
        elif verb == "go" and (location, command) not in self.routes:
            rank = 2
        elif verb == "go":
            rank = 3
            destination = self.routes[(location, command)]
            reversal_penalty = int(self._is_immediate_reversal(state, command))
            traversal_count = self.route_traversals[(location, command)]
            destination_visits = self.location_visits[destination]
        elif verb == "take":
            rank = 4
        elif verb == "examine":
            rank = 5
        else:
            rank = 6
        return (
            rank,
            reversal_penalty,
            traversal_count,
            destination_visits,
            command,
        )

    @staticmethod
    def _take_target_is_relevant(command: str, objective: str) -> bool:
        target = command[len("take "):].split(" from ", 1)[0].lower()
        objective = objective.lower()
        target_tokens = [token for token in target.split() if len(token) > 2]
        return any(token in objective for token in target_tokens)

    def _remember_route(
        self,
        before: GameState,
        command: str,
        after: GameState,
    ) -> None:
        origin = before.location
        destination = after.location
        if not origin or not destination:
            return

        self.routes[(origin, command)] = destination
        self.route_traversals[(origin, command)] += 1
        reverse = self._reverse_direction(command)
        if reverse and reverse in after.admissible_commands:
            self.routes[(destination, reverse)] = origin

    def _is_immediate_reversal(self, state: GameState, command: str) -> bool:
        if not self.history:
            return False

        previous = self.history[-1]
        if previous.state_before.location == previous.state_after.location:
            return False

        destination = self.routes.get((state.location or "Unknown", command))
        return destination == previous.state_before.location

    def _remember_two_action_cycle(self, record: ActionRecord) -> None:
        if len(self.history) < 2:
            return
        previous = self.history[-2]
        if (
            self.signature(previous.state_before) == self.signature(record.state_after)
            and self.signature(previous.state_after) == self.signature(record.state_before)
        ):
            self.cyclic_actions.add(
                (self.signature(previous.state_before), previous.command)
            )

    @staticmethod
    def _placement(command: str) -> tuple[str, str] | None:
        if command.startswith("insert ") and " into " in command:
            item, destination = command[len("insert "):].split(" into ", 1)
            return item, destination
        if command.startswith("put ") and " on " in command:
            item, destination = command[len("put "):].split(" on ", 1)
            return item, destination
        return None

    @staticmethod
    def _action_required_by_objective(command: str, objective: str) -> bool:
        verb, _, target = command.partition(" ")
        text = objective.lower()
        return verb in text and target in text

    @staticmethod
    def _reverse_direction(command: str) -> str | None:
        return {
            "go north": "go south",
            "go south": "go north",
            "go east": "go west",
            "go west": "go east",
            "go up": "go down",
            "go down": "go up",
        }.get(command)

    @staticmethod
    def signature(state: GameState) -> tuple:
        return (
            state.location,
            state.description,
            state.inventory,
            state.admissible_commands,
            state.score,
            state.won,
            state.lost,
        )
