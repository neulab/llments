"""Module for a datastore containing data for retrieval."""
import abc
from typing import Any


class Datastore:
    """A datastore containing data for retrieval."""

    @abc.abstractmethod
    def retrieve(
        self,
        query: str,
        max_results: int,
        query_encoder: str | None=None,
        device: str | None=None,
        pooling: str | None=None,
        l2_norm: bool | None=None,
    ) -> object:
        """Retrieve documents based on the specified parameters.

        Args:
            query (str): The query string to search for.
            max_results (int): Maximum number of results to retrieve.
            query_encoder (str, optional): The type of query encoder to be used.

        Returns:
            object: Retrieved result object.
        """
        pass
