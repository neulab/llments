"""Module for a datastore containing data for retrieval."""
import abc
from typing import Any


class Datastore:
    """A datastore containing data for retrieval."""

    document_path: str

    @abc.abstractmethod
    def retrieve(
        self,
        query: str | None,
        max_results: int,
        query_encoder: str | None=None,
        device: str = 'cpu',
        pooling: str = 'cls',
        l2_norm: bool = False,
    ) -> Any:
        """Retrieve documents based on the specified parameters.

        Args:
            query (str): The query string to search for.
            max_results (int): Maximum number of results to retrieve.
            query_encoder (str, optional): The type of query encoder to be used.
            device (str, optional): Device to be used for encoding. Defaults to 'cpu'.
            pooling (str, optional): Type of pooling to be used for encoding. Defaults to 'cls'.
            l2_norm (bool, optional): Whether to apply L2 normalization to embeddings. Defaults to False.

        Returns:
            Any: Retrieved result object.
        """
        pass
