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

    def get_location(self):
        if not self.state:
            return None

        match = re.search(r"-=\s*(.*?)\s*=-", self.state.description)

        if match:
            return match.group(1)

        return None

    def close(self):
        self.env.close()
