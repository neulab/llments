"""Module for a datastore containing data for retrieval."""
import abc

try:
    from pyserini.search.faiss import DenseSearchResult
except ImportError:
    raise ImportError(
        "You need to install the `pyserini` package to use this class."
    )

try:
    from faiss import IndexPreTransform
except ImportError:
    raise ImportError(
        "You need to install the `faiss` package to use this class."
    )

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
        index: IndexPreTransform | None,
        docids: list[str] | None,
        max_results: int,
    ) -> list[DenseSearchResult]:
        """Retrieve documents based on the specified parameters.

        Args:
            query (str): The query string to search for.
            index (IndexPreTransform): The vector index from faiss 
            docids (list[str]): List of docids.
            max_results (int): Maximum number of results to retrieve.

        Returns:
            list[DenseSearchResult]: Retrieved result objects.
        """
        pass
