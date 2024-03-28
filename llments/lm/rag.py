from llments.datastore.datastore import Datastore
from llments.lm.lm import LanguageModel


class RAGLanguageModel(LanguageModel):
    def __init__(self, base: LanguageModel, datastore: Datastore):
        """Apply retrieval-augmented generation over a datastore.

        Args:
            base: The language model to be modified.
            datastore: The datastore object for document index

        Returns:
            LanguageModel: The enhanced language model.
        """

    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
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
            temperature: The value used to module the next token probabilities.
            num_return_sequences: The number of independently computed returned
                sequences for each element in the batch.

        Returns:
            str: Sampled output sequences from the language model.
        """
        raise NotImplementedError
