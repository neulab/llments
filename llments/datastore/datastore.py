"""Module for a datastore containing data for retrieval."""
import abc
from typing import TYPE_CHECKING
from pyserini.search.faiss import DenseSearchResult

class Datastore:
    """A datastore containing data for retrieval."""

    document_path: str
    index_path: str
    docid_field: str
    fields: list[str]

    @abc.abstractmethod
    def retrieve(
        self,
        query: str | None,
        max_results: int,
    ) -> list[DenseSearchResult]:
        """Retrieve documents based on the specified parameters.

        Args:
            query (str): The query string to search for.
            max_results (int): Maximum number of results to retrieve.

        Returns:
            list[DenseSearchResult]: Retrieved result objects.
        """
        pass
