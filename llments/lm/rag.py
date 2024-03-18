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
        raise NotImplementedError("This is not implemented yet.")
