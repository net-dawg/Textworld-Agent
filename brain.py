import requests


class OllamaBrain:
    def __init__(self, model="qwen3:8b"):
        self.model = model
        self.url = "http://127.0.0.1:11434/api/generate"

    def generate(self, prompt):
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()

        return data["response"].strip()
