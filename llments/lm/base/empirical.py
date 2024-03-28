"""A module for empirical language models."""

from llments.lm.lm import LanguageModel
import random
import json
import pandas as pd


class EmpiricalDistribution(LanguageModel):
    """An empirical distribution of text data."""

    def __init__(self, data: list[str], probs: list[float] | None = None):
        """Initialize the empirical distribution.

        Args:
            data: The data to be used for the empirical distribution.
            probs: The probabilities of each data point in the distribution.
                If None, the probabilities are assumed to be uniform.
        """
        if probs is None:
            probs = [1 / len(data)] * len(data)
        self.data = pd.DataFrame({"text": data, "prob": probs})

    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """See base class."""
        filtered_df = self.data
        # Adjust distribution
        if condition:
            filtered_df = self.data[self.data["text"].str.startswith(condition)]
        if not do_sample:
            raise NotImplementedError("Greedy decoding is not implemented yet.")
        if max_length is not None:
            filtered_df = filtered_df[
                filtered_df["text"].str.split().len() <= max_length
            ]
        if temperature != 1.0:
            raise NotImplementedError("Temperature is not implemented yet.")
        if filtered_df.empty:
            raise ValueError(
                f"Condition {condition} does not match any strings in the "
                "distribution."
            )
        # Normalize the probabilities
        filtered_df["prob"] = filtered_df["prob"] / filtered_df["prob"].sum()
        rets: list[str] = random.choices(
            filtered_df["text"], weights=filtered_df["probs"], k=num_return_sequences
        )[0]
        return rets

    def calculate_probability(self, x: str) -> float:
        """Calculate the probability of an output given the language model.

        Args:
            x: The output sequence for which the probability is calculated.

        Returns:
            float: The probability of output x given the language model.
        """
        # Implementation logic
        raise NotImplementedError("This is not implemented yet.")

    def set_seed(self, seed: int) -> None:
        """Set the seed for the language model.

        Args:
            seed: The seed to set for the language model.
        """
        random.seed(seed)


def load_from_text_file(text_file: str) -> EmpiricalDistribution:
    """Load the distribution from a text file."""
    with open(text_file, "r") as f:
        return EmpiricalDistribution(f.readlines())


def load_from_json_file(json_file: str) -> EmpiricalDistribution:
    """Load the distribution from a text file."""
    with open(json_file, "r") as f:
        data = json.load(f)
        return EmpiricalDistribution(
            [x["text"] for x in data], [x["prob"] for x in data]
        )
