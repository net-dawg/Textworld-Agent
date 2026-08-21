import re
import textworld


class TextWorldEnvironment:
    def __init__(self, game_file):
        self.game_file = game_file

        self.request_infos = textworld.EnvInfos(
            description=True,
            inventory=True,
            objective=True,
            admissible_commands=True,
            score=True,
            moves=True,
            won=True,
            lost=True,
        )

        self.env = textworld.start(
            self.game_file,
            request_infos=self.request_infos,
        )

        self.state = None

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def step(self, command):
        self.state, reward, done = self.env.step(command)
        return self.state, reward, done

    def get_location(self, state=None):
        state = state or self.state

        if not state:
            return None

        match = re.search(r"-=\s*(.*?)\s*=-", state.description)

        if match:
            return match.group(1)

        return None

    def get_context(self, state=None):
        state = state or self.state

        return {
            "location": self.get_location(state),
            "description": state.description,
            "feedback": self.get_action_result(state),
            "inventory": state.inventory,
            "score": state.score,
            "moves": state.moves,
        }

    def get_action_result(self, state=None):
        state = state or self.state

        if state.moves == 0:
            return "No action has been taken yet."

        # TextWorld appends its parser prompt and status line to feedback.
        # They are terminal UI, not useful context for the model.
        feedback = re.sub(r"\n>\s*-=.*$", "", state.feedback, flags=re.DOTALL)
        return feedback.strip()

    def state_signature(self, state=None):
        state = state or self.state

        return (
            self.get_location(state),
            state.inventory,
            state.score,
            tuple(sorted(state.admissible_commands)),
        )

    def close(self):
        self.env.close()
