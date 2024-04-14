"""A module for retrieval-augmented generation based language models."""

from llments.datastore.datastore import Datastore
from llments.lm.lm import LanguageModel


class RAGLanguageModel(LanguageModel):
    """Retrieval-augmented generation based language models."""

    def __init__(self, base: LanguageModel, datastore: Datastore):
        """Apply retrieval-augmented generation over a datastore.

        Args:
            base: The language model to be modified.
            datastore: The datastore to be used to retrieve the relevant information.

        Returns:
            LanguageModel: The enhanced language model.
        """

    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """Generate an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
                If None, the output is not conditioned.
            do_sample: Whether to use sampling or greedy decoding.
            max_length: The maximum length of the output sequence,
                (defaults to model max).
            max_new_tokens: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            temperature: The value used to module the next token probabilities.
            num_return_sequences: The number of independently computed returned
                sequences for each element in the batch.

        Returns:
            str: Sampled output sequences from the language model.
        """
        raise NotImplementedError
