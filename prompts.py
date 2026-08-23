try:
    from .environment import GameState
    from .memory import WorldMemory
except ImportError:
    from environment import GameState
    from memory import WorldMemory


def build_decision_prompt(
    state: GameState,
    memory: WorldMemory,
    correction: str | None = None,
) -> str:
    candidates = memory.candidate_actions(state)
    candidate_text = "\n".join(
        f"{index}. {command}"
        for index, command in enumerate(candidates, start=1)
    ) or "None"

    visited = ", ".join(sorted(memory.visited_locations)) or "None"
    routes = "\n".join(
        f"- {origin} --{command}--> {destination} "
        f"(used {memory.route_traversals[(origin, command)]} times; "
        f"destination visited {memory.location_visits[destination]} times)"
        for (origin, command), destination in sorted(memory.routes.items())
    ) or "- None"
    findings = "\n".join(
        f"- In {location}, {command} => {result}"
        for (location, command), result in sorted(memory.findings.items())
    ) or "- None"
    recent = "\n".join(
        f"- {record.command} => {record.outcome.value}: {record.result}"
        for record in memory.history[-6:]
    ) or "- None"
    rejected = "\n".join(
        f"- {command}" for command in memory.rejected_actions(state)
    ) or "- None"
    correction_text = (
        f"\nPREVIOUS RESPONSE REJECTED:\n{correction}\n"
        if correction
        else ""
    )

    return f"""You control a TextWorld agent. Choose the smartest next action.

Use only verified evidence below. Never call an unreadable, empty, or
uninformative result a clue. Never claim an item was found unless the state or
feedback explicitly says so. Choose one exact candidate action. Prefer acquiring
visible useful items, unlocking/opening relevant objects, and unexplored routes.
Avoid undoing useful progress or storing carried items unless the objective
explicitly requires that placement.

OBJECTIVE:
{state.objective}

CURRENT STATE:
Location: {state.location}
Description: {state.description}
Inventory: {state.inventory}
Last result: {state.feedback}
Score: {state.score}
Moves: {state.moves}

VERIFIED MEMORY:
Visited: {visited}

Known routes:
{routes}

Observed findings (the text after => is the complete evidence):
{findings}

Recent verified outcomes:
{recent}

CANDIDATE ACTIONS (highest-priority actions for this move):
{candidate_text}

NOT OFFERED THIS MOVE (lower priority, exhausted, cyclic, or regressive):
{rejected}
{correction_text}
Return exactly one JSON object:
- "goal": one small outcome this action should accomplish
- "action": exactly one candidate action
- "reason": cite only current state or verified memory explaining why this is
  better than the other candidates
"""
