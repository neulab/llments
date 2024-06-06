"""A module for empirical language models."""

import json
import random

import pandas as pd

from llments.lm.lm import LanguageModel


class DatasetLM(LanguageModel):
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
        max_new_tokens: int | None = None,
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
        if max_new_tokens is not None:
            if max_length is not None:
                raise ValueError("Specify only one: max_length or max_new_tokens.")
            max_length = len(condition or "") + max_new_tokens
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
            filtered_df["text"], weights=filtered_df["prob"], k=num_return_sequences
        )
        return rets

    def chat_generate(
        self,
        messages: list[dict[str, str]],
        do_sample: bool = False,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> list[list[dict[str, str]]]:
        """Generate an output given a chat context.

        Args:
            messages: A list of dictionaries, each representing a message in the chat context. Each dictionary should contain the following keys:
            - "role": The role of the entity sending the message. This can be "system", "user", etc.
            - "content": The actual content of the message. Example:
            [
                {
                    "role": "system",
                    "content": "You are a friendly chatbot",
                },
                {
                    "role": "user",
                    "content": "How many helicopters can a human eat in one sitting?"
                },
            ]
            do_sample: Whether to use sampling or greedy decoding.
            max_length: The maximum length of the output sequence,
                (defaults to model max).
            max_new_tokens: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            temperature: The value used to module the next token probabilities.
            num_return_sequences: The number of independently computed returned
                sequences for each element in the batch.

        Returns:
            list[list[dict[str, str]]]: list of chat contexts with the generated responses.
        """
        raise NotImplementedError("This is not implemented yet.")

    def calculate_probability(self, condition: str | None, output: str) -> float:
        """Calculate the probability of a given text sequence in the empirical distribution.

        Args:
            condition: conditioning text, but in DatasetML it's defaulted as None.
            output: The text sequence for which to calculate the probability.

        Returns:
            float: The probability of the sequence if it exists in the dataset, otherwise 0.
        """
        result = self.data[self.data["text"] == output]["prob"]
        if not result.empty:
            return float(result.iloc[0])
        else:
            return 0.0

    def set_seed(self, seed: int) -> None:
        """Set the seed for the language model.

        Args:
            seed: The seed to set for the language model.
        """
        random.seed(seed)


def load_from_text_file(text_file: str) -> DatasetLM:
    """Load the distribution from a text file."""
    with open(text_file, "r") as f:
        return DatasetLM(f.readlines())


def load_from_json_file(json_file: str) -> DatasetLM:
    """Load the distribution from a text file."""
    with open(json_file, "r") as f:
        data = json.load(f)
        return DatasetLM([x["text"] for x in data], [x["prob"] for x in data])
