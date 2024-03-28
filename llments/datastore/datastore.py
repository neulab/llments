import abc
from typing import Any


class Datastore:
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
        """
        Initializes a PyseriniDatastore object.

        Args:
            index_path: The path to store the generated index.
            document_path: The path to the document file. Defaults to None.
            index_encoder: The type of document encoder. Defaults to None.
            fields: The document fields to be encoded. Defaults to ['text'].
            to_faiss: Store as a FAISS index. Defaults to False.
            device: The device to be used for encoding. Defaults to 'cpu'.
            delimiter: Delimiter for document separation. Defaults to "\n".
            docid_field: Field in the document containing document id. Defaults to None.
            batch_size: Batch size for encoding. Defaults to 64.
            max_length: Maximum length of the input sequence. Defaults to 256.
            dimension: Dimensionality of the encoding. Defaults to 768.
            prefix: Prefix to add to each document. Defaults to None.
            pooling: Pooling strategy for document encoding. Defaults to None.
            l2_norm: Whether to apply L2 normalization. Defaults to None.
            use_openai: Whether to use OpenAI's encoder. Defaults to False.
            rate_limit: Rate limit for OpenAI API requests. Defaults to 3500.
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
        """
        Encodes documents using the specified parameters.

        Args:
            document_path: The path to the document file.
            index_encoder: The type of document encoder.
            fields: The document fields to be encoded. Defaults to ['text'].
            delimiter: Delimiter for document separation. Defaults to "\n".
            docid_field: Field in the document containing document id. Defaults to None.
            batch_size: Batch size for encoding. Defaults to 64.
            max_length: Maximum length of the input sequence. Defaults to 256.
            dimension: Dimensionality of the encoding. Defaults to 768.
            prefix: Prefix to add to each document. Defaults to None.
            pooling: Pooling strategy for document encoding. Defaults to None.
            l2_norm: Whether to apply L2 normalization. Defaults to None.
            to_faiss: Whether to store as a FAISS index. Defaults to False.
            device: The device to be used for encoding. Defaults to 'cpu'.
            use_openai: Whether to use OpenAI's encoder. Defaults to False.
            rate_limit: Rate limit for OpenAI API requests. Defaults to 3500.
        """
        pass
