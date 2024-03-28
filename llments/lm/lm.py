"""Base class for language models."""

import abc


class LanguageModel:
    """Base class for language models."""

    @abc.abstractmethod
    def calculate_probability(self, output: str) -> float:
        """Calculate the probability of an output given the language model.

        Args:
            output: The output sequence for which the probability is calculated.

        Returns:
            float: The probability of output x given the language model.
        """
        ...

    @abc.abstractmethod
    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """Generate an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
                If None, the output is not conditioned.
            do_sample: Whether to use sampling or greedy decoding.
            max_length: The maximum length of the output sequence,
                (defaults to model max).
            temperature: The value used to module the next token probabilities.
            num_return_sequences: The number of independently computed returned
                sequences for each element in the batch.

        Returns:
            str: Sampled output sequences from the language model.
        """
        ...

    @abc.abstractmethod
    def set_seed(self, seed: int):
        """Set the seed for the language model.

        Args:
            seed: The seed to set for the language model.
        """
        ...
