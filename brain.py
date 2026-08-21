import json

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
                "format": {
                    "type": "object",
                    "properties": {
                        "assessment": {"type": "string"},
                        "subgoal_status": {
                            "type": "string",
                            "enum": ["continue", "complete", "blocked", "replace"],
                        },
                        "subgoal": {"type": "string"},
                        "success_condition": {"type": "string"},
                        "memory_update": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "action": {"type": "string"},
                    },
                    "required": [
                        "assessment",
                        "subgoal_status",
                        "subgoal",
                        "success_condition",
                        "memory_update",
                        "action",
                    ],
                    "additionalProperties": False,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()
        decision = json.loads(data["response"])

        text_fields = ["assessment", "subgoal", "success_condition", "action"]

        for field in text_fields:
            decision[field] = decision.get(field, "").strip()

        if any(not decision[field] for field in text_fields):
            raise ValueError("Model returned an incomplete ReAct decision.")

        return {
            "assessment": decision["assessment"],
            "subgoal_status": decision["subgoal_status"],
            "subgoal": decision["subgoal"],
            "success_condition": decision["success_condition"],
            "memory_update": [
                fact.strip()
                for fact in decision.get("memory_update", [])
                if fact.strip()
            ],
            "action": decision["action"],
        }
