"""Module for a datastore containing data for retrieval."""
import abc
from typing import Any


class Datastore:
    """A datastore containing data for retrieval."""

    @abc.abstractmethod
    def __init__(
        self,
        index_path: str,
        document_path: str | None = None,
        index_encoder: Any | None = None,
        fields: list[str] | None = None,
        to_faiss: bool = False,
        device: str = "cpu",
        delimiter: str = "\n",
        docid_field: str | None = None,
        batch_size: int = 64,
        max_length: int = 256,
        dimension: int = 768,
        prefix: str | None = None,
        pooling: str | None = None,
        l2_norm: bool = False,
        use_openai: bool = False,
        rate_limit: int = 3500,
    ):
        """Initializes a PyseriniDatastore object.

        Args:
            index_path: The path to store the generated index.
            document_path: The path to the document file.
            index_encoder: The type of document encoder.
            fields: The document fields to be encoded.
            to_faiss: Store as a FAISS index.
            device: The device to be used for encoding.
            delimiter: Delimiter for document separation.
            docid_field: Field in the document containing document id.
            batch_size: Batch size for encoding.
            max_length: Maximum length of the input sequence.
            dimension: Dimensionality of the encoding.
            prefix: Prefix to add to each document.
            pooling: Pooling strategy for document encoding.
            l2_norm: Whether to apply L2 normalization.
            use_openai: Whether to use OpenAI's encoder.
            rate_limit: Rate limit for OpenAI API requests.
        """
        pass

    @abc.abstractmethod
    def encode(
        self,
        document_path: str | None = None,
        index_encoder: Any | None = None,
        fields: list[str] | None = None,
        to_faiss: bool = False,
        device: str = "cpu",
        delimiter: str = "\n",
        docid_field: str | None = None,
        batch_size: int = 64,
        max_length: int = 256,
        dimension: int = 768,
        prefix: str | None = None,
        pooling: str | None = None,
        l2_norm: bool = False,
        use_openai: bool = False,
        rate_limit: int = 3500,
    ) -> None:
        """Encodes documents using the specified parameters.

        Args:
            document_path: The path to the document file.
            index_encoder: The type of document encoder.
            fields: The document fields to be encoded.
            delimiter: Delimiter for document separation.
            docid_field: Field in the document containing document id.
            batch_size: Batch size for encoding.
            max_length: Maximum length of the input sequence.
            dimension: Dimensionality of the encoding.
            prefix: Prefix to add to each document.
            pooling: Pooling strategy for document encoding.
            l2_norm: Whether to apply L2 normalization.
            to_faiss: Whether to store as a FAISS index.
            device: The device to be used for encoding.
            use_openai: Whether to use OpenAI's encoder.
            rate_limit: Rate limit for OpenAI API requests.
        """
        pass
