from textworld import GameMaker

# --------------------------------------------------
# 1. Create an empty TextWorld game
# --------------------------------------------------

M = GameMaker()


# --------------------------------------------------
# 2. Create two rooms
# --------------------------------------------------

kitchen = M.new_room(
    name="Kitchen",
    desc="A small kitchen with white cabinets and a wooden floor."
)

bedroom = M.new_room(
    name="Bedroom",
    desc="A quiet bedroom with a bed against the wall."
)


# --------------------------------------------------
# 3. Connect the rooms
#
# Kitchen --north--> Bedroom
# Bedroom --south--> Kitchen
# --------------------------------------------------

M.connect(
    kitchen.north,
    bedroom.south
)


# --------------------------------------------------
# 4. Place the player
# --------------------------------------------------

M.set_player(kitchen)


# --------------------------------------------------
# 5. Create an object
#
# "o" means ordinary portable object.
# --------------------------------------------------

key = M.new(
    type="o",
    name="brass key",
    desc="A small brass key."
)


# --------------------------------------------------
# 6. Place the key in the bedroom
# --------------------------------------------------

bedroom.add(key)


# --------------------------------------------------
# 7. Define the quest
#
# These commands describe a valid route to the
# winning condition.
# --------------------------------------------------

M.set_quest_from_commands([
    "go north",
    "take brass key",
])


# --------------------------------------------------
# 8. Print some internal information
# --------------------------------------------------

print("\nROOMS:")
for room in M.rooms:
    print(f"  - {room.name}")


print("\nWORLD FACTS:")
for fact in M.facts:
    print(f"  - {fact}")


# --------------------------------------------------
# 9. Compile the game
# --------------------------------------------------

game_file = M.compile("games/tiny_game.z8")

print(f"\nGame created: {game_file}")
