import json
import re
from datetime import datetime, timezone
from pathlib import Path

from prompts import SYSTEM_PROMPT

class Agent:
    def __init__(
        self,
        brain,
        environment,
        max_steps=20,
        history_limit=8,
        trace_directory="logs",
    ):
        self.brain = brain
        self.environment = environment
        self.max_steps = max_steps
        self.history_limit = history_limit
        self.history = []
        self.failed_actions = {}
        self.current_subgoal = None
        self.current_success_condition = None
        self.visited_locations = set()
        self.known_routes = {}
        self.milestones = []
        self.world_facts = []
        self.trace_directory = Path(trace_directory)
        self.trace_path = None
        self.trace = None

    def start_trace(self):
        started_at = datetime.now(timezone.utc)
        run_id = started_at.strftime("%Y%m%dT%H%M%S.%fZ")

        self.trace_directory.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.trace_directory / f"prompt_trace_{run_id}.json"
        self.trace = {"prompts": []}
        self.write_trace()

    def write_trace(self):
        self.trace_path.write_text(
            json.dumps(self.trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def log_prompt(self, step_number, prompt):
        self.trace["prompts"].append({
            "step": step_number,
            "subgoal_before": self.current_subgoal,
            "success_condition_before": self.current_success_condition,
            "prompt_lines": prompt.split("\n"),
            "assessment": None,
            "subgoal_status": None,
            "subgoal_after": None,
            "success_condition_after": None,
            "memory_update": [],
            "action": None,
            "decision_accepted": None,
        })
        self.write_trace()
        return len(self.trace["prompts"]) - 1

    def log_decision(self, trace_index, decision, accepted):
        entry = self.trace["prompts"][trace_index]
        entry["assessment"] = decision["assessment"]
        entry["subgoal_status"] = decision["subgoal_status"]
        entry["subgoal_after"] = decision["subgoal"]
        entry["success_condition_after"] = decision["success_condition"]
        entry["memory_update"] = decision["memory_update"]
        entry["action"] = decision["action"]
        entry["decision_accepted"] = accepted
        self.write_trace()

    def format_history(self):
        if not self.history:
            return "No actions have been attempted yet."

        entries = []

        for item in self.history[-self.history_limit:]:
            entries.append(
                f"- In {item['location']}, tried: {item['action']}\n"
                f"  Result: {item['result']}\n"
                f"  Progress: {'yes' if item['made_progress'] else 'no'}"
            )

        return "\n".join(entries)

    def format_world_memory(self):
        visited = ", ".join(sorted(self.visited_locations)) or "None"

        if self.known_routes:
            routes = "\n".join(
                f"- {origin} --{action}--> {destination}"
                for (origin, action), destination in sorted(self.known_routes.items())
            )
        else:
            routes = "- No routes discovered yet."

        milestones = (
            "\n".join(f"- {item}" for item in self.milestones)
            or "- None yet."
        )
        facts = (
            "\n".join(f"- {fact}" for fact in self.world_facts)
            or "- None yet."
        )

        return f"""VISITED LOCATIONS:
{visited}

KNOWN ROUTES:
{routes}

COMPLETED MILESTONES:
{milestones}

OBSERVED FACTS:
{facts}"""

    def format_failed_actions(self, location):
        failures = [
            failure
            for (failure_location, _), failure in self.failed_actions.items()
            if failure_location == location
        ]

        if not failures:
            return "- None."

        return "\n".join(
            f"- Do not retry '{failure['action']}' yet. "
            f"It failed because: {failure['reason']} "
            f"Retry condition: {failure['retry_condition']}"
            for failure in failures
        )

    def get_action_target(self, action):
        for prefix in ("open ", "close ", "examine ", "drop "):
            if action.startswith(prefix):
                return action[len(prefix):]

        if action.startswith(("unlock ", "lock ")):
            return action.split(" with ", 1)[0].split(" ", 1)[1]

        if action.startswith("take "):
            return action[len("take "):].split(" from ", 1)[0]

        if action.startswith("insert "):
            return action.split(" into ", 1)[-1]

        if action.startswith("put "):
            return action.split(" on ", 1)[-1]

        return None

    def remember_failed_action(self, location, action, result):
        action_target = self.get_action_target(action)
        dependency_target = action_target
        retry_condition = "A directly relevant condition must change."

        locked_match = re.search(
            r"unlock the (.*?) with",
            result,
            flags=re.IGNORECASE,
        )
        closed_match = re.search(
            r"open the (.*?) first",
            result,
            flags=re.IGNORECASE,
        )

        if locked_match:
            dependency_target = locked_match.group(1).strip(" .")
            retry_condition = f"Successfully unlock {dependency_target}."
        elif closed_match:
            dependency_target = closed_match.group(1).strip(" .")
            retry_condition = f"Successfully open {dependency_target}."
        elif action_target:
            retry_condition = (
                f"A successful action must change {action_target}."
            )

        self.failed_actions[(location, action)] = {
            "action": action,
            "reason": result,
            "dependency_target": dependency_target,
            "retry_condition": retry_condition,
        }

    def resolve_failed_actions(self, action, made_progress):
        if not made_progress:
            return

        changed_target = self.get_action_target(action)

        if not changed_target:
            return

        resolved = [
            key
            for key, failure in self.failed_actions.items()
            if failure["dependency_target"] == changed_target
        ]

        for key in resolved:
            del self.failed_actions[key]

    def update_world_memory(
        self,
        previous_location,
        action,
        state,
        reward,
        made_progress,
        memory_update,
    ):
        current_location = self.environment.get_location(state)
        self.visited_locations.add(current_location)

        if current_location != previous_location:
            self.known_routes[(previous_location, action)] = current_location

        milestone_prefixes = ("take ", "open ", "unlock ")
        is_milestone = made_progress and action.startswith(milestone_prefixes)

        if reward > 0 or is_milestone:
            milestone = f"{action} -> {self.environment.get_action_result(state)}"
            if milestone not in self.milestones:
                self.milestones.append(milestone)

        for fact in memory_update:
            if fact not in self.world_facts:
                self.world_facts.append(fact)

    def get_candidate_commands(self, state):
        location = self.environment.get_location(state)

        return [
            command
            for command in state.admissible_commands
            if (location, command) not in self.failed_actions
        ]

    def build_prompt(self, state, candidate_commands=None):
        context = self.environment.get_context(state)
        candidate_commands = candidate_commands or self.get_candidate_commands(state)
        commands = "\n".join(
            f"- {command}"
            for command in candidate_commands
        )

        prompt = f"""
{SYSTEM_PROMPT}

OBJECTIVE:
{state.objective}

CURRENT LOCATION:
{context['location']}

ROOM DESCRIPTION:
{context['description']}

LAST ACTION RESULT:
{context['feedback']}

INVENTORY:
{context['inventory']}

SCORE: {context['score']}
MOVES: {context['moves']}

RECENT HISTORY:
{self.format_history()}

WORLD MEMORY:
{self.format_world_memory()}

FAILED ACTIONS IN THIS LOCATION:
{self.format_failed_actions(context['location'])}

CURRENT PLAN:
Subgoal:
{self.current_subgoal or "No subgoal yet. Create the first useful subgoal."}

Success condition:
{self.current_success_condition or "No success condition yet."}

AVAILABLE COMMANDS:
{commands}

DECISION:
"""

        return prompt

    def run(self):
        self.history = []
        self.failed_actions = {}
        self.current_subgoal = None
        self.current_success_condition = None
        self.visited_locations = set()
        self.known_routes = {}
        self.milestones = []
        self.world_facts = []
        state = self.environment.reset()
        self.visited_locations.add(self.environment.get_location(state))
        self.start_trace()

        print(f"Trace file: {self.trace_path}")

        print("\nOBJECTIVE:")
        print(state.objective)

        print("\nSTARTING OBSERVATION:")
        print(state.feedback)

        for step_number in range(1, self.max_steps + 1):

            print("\n" + "=" * 60)
            print(f"AGENT STEP {step_number}")
            print("=" * 60)

            print(f"Location: {self.environment.get_location()}")

            candidate_commands = self.get_candidate_commands(state)

            if not candidate_commands:
                print("No untried commands remain in the current state.")
                return

            prompt = self.build_prompt(state, candidate_commands)
            trace_index = self.log_prompt(step_number, prompt)

            decision = self.brain.generate(prompt)
            action = decision["action"]
            decision_accepted = action in candidate_commands
            self.log_decision(trace_index, decision, decision_accepted)

            print(f"Qwen subgoal: {decision['subgoal']}")
            print(f"Qwen assessment: {decision['assessment']}")
            print(f"Qwen chose: {action}")

            if not decision_accepted:
                print("Invalid or temporarily blocked model response.")
                print("Current candidate commands were:")

                for command in candidate_commands:
                    print(f"  - {command}")

                continue

            self.current_subgoal = decision["subgoal"]
            self.current_success_condition = decision["success_condition"]

            previous_location = self.environment.get_location(state)
            previous_signature = self.environment.state_signature(state)

            state, reward, done = self.environment.step(action)

            made_progress = (
                previous_signature != self.environment.state_signature(state)
                or reward > 0
            )

            self.history.append({
                "location": previous_location,
                "action": action,
                "result": self.environment.get_action_result(state),
                "reward": reward,
                "made_progress": made_progress,
            })

            self.update_world_memory(
                previous_location=previous_location,
                action=action,
                state=state,
                reward=reward,
                made_progress=made_progress,
                memory_update=decision["memory_update"],
            )

            action_result = self.environment.get_action_result(state)

            if made_progress:
                self.resolve_failed_actions(action, made_progress=True)
            else:
                self.remember_failed_action(
                    previous_location,
                    action,
                    action_result,
                )

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
