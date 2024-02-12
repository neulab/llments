from typing import Any
from llments.lm.lm import LanguageModel
import random
import json
import pandas as pd


class EmpiricalDistribution(LanguageModel):
    def __init__(self, data: list[str], probs: list[float] | None = None):
        if probs is None:
            probs = [1 / len(data)] * len(data)
        self.data = pd.DataFrame({"text": data, "prob": probs})

    def generate(self, condition: str | None, **kwargs: Any) -> str:
        """Sample from the language model, possibly conditioned on a prefix."""
        if condition is None:
            return random.choices(self.data["text"], weights=self.data["probs"])[0]
        else:
            # Filter to only those that start with the condition
            filtered_df = self.data[self.data["text"].str.startswith(condition)]
            if filtered_df.empty:
                raise ValueError(
                    f"Condition {condition} does not match any strings in the "
                    "distribution."
                )
            # Normalize the probabilities
            filtered_df["prob"] = filtered_df["prob"] / filtered_df["prob"].sum()
            return random.choices(filtered_df["text"], weights=filtered_df["probs"])[0]

    def fit(self, target: LanguageModel, task_description: str | None = None):
        raise ValueError(
            "Cannot fit an empirical distribution to another distribution."
        )


def load_from_text_file(text_file: str):
    """Load the distribution from a text file."""
    with open(text_file, "r") as f:
        return EmpiricalDistribution(f.readlines())


def load_from_json_file(json_file: str):
    """Load the distribution from a text file."""
    with open(json_file, "r") as f:
        data = json.load(f)
        return EmpiricalDistribution(
            [x["text"] for x in data], [x["prob"] for x in data]
        )
