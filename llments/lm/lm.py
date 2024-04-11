"""Base class for language models."""

import abc


class LanguageModel:
    """Base class for language models."""

    @abc.abstractmethod
    def calculate_probability(self, condition: str | None, output: str) -> float:
        """Calculate the probability of an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
                If None, the output is not conditioned.
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
    def chat_generate(
        self,
        messages: list[dict[str, str]],
        do_sample: bool = False,
        max_length: int | None = None,
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
            temperature: The value used to module the next token probabilities.
            num_return_sequences: The number of independently computed returned
                sequences for each element in the batch.

        Returns:
            list[list[dict[str, str]]]: list of chat contexts with the generated responses.
        """
        ...

    @abc.abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set the seed for the language model.

        Args:
            seed: The seed to set for the language model.
        """
        ...
