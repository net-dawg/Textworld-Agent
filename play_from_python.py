import textworld
import re

GAME_FILE = "games/tiny_game.z8"

def get_location(state):
    match = re.search(r"-=\s*(.*?)\s*=-", state.description)
    return match.group(1) if match else None

# Ask TextWorld to expose some useful information
request_infos = textworld.EnvInfos(
    description=True,
    inventory=True,
    location=True,
    objective=True,
    admissible_commands=True,
    score=True,
    moves=True,
    won=True,
    lost=True,
)


# Load the game
env = textworld.start(
    GAME_FILE,
    request_infos=request_infos,
)


# Start a new episode
state = env.reset()


print("=" * 60)
print("GAME START")
print("=" * 60)

print("\nOBJECTIVE:")
print(state.objective)

print("\nINITIAL OBSERVATION:")
print(state.feedback)


done = False

while not done:

    print("\n" + "-" * 60)

    print(f"Location: {get_location(state)}")
    print(f"Score: {state.score}")
    print(f"Moves: {state.moves}")

    print("\nInventory:")
    print(state.inventory)

    print("\nAdmissible commands:")
    for command in state.admissible_commands:
        print(f"  - {command}")

    print()

    command = input("> ").strip()

    if command.lower() in {"quit", "exit"}:
        break

    state, reward, done = env.step(command)

    print("\nTEXTWORLD RESPONSE:")
    print(state.feedback)

    print(f"\nReward from this action: {reward}")


print("\n" + "=" * 60)

if state.won:
    print("YOU WON")
elif state.lost:
    print("YOU LOST")
else:
    print("GAME ENDED")

print("=" * 60)

env.close()
