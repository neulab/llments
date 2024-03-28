"""Module for a datastore containing data for retrieval."""
import abc


class Datastore:
    """A datastore containing data for retrieval."""

    @abc.abstractmethod
    def __init__(
        self,
        index_path: str,
        document_path=None,
        index_encoder=None,
        fields=None,
        to_faiss=False,
        device="cpu",
        delimiter="\n",
        docid_field=None,
        batch_size=64,
        max_length=256,
        dimension=768,
        prefix=None,
        pooling=None,
        l2_norm=None,
        use_openai=False,
        rate_limit=3500,
    ):
        """Initializes a PyseriniDatastore object.

        Args:
            index_path (str): The path to store the generated index.
            document_path (str, optional): The path to the document file.
            index_encoder (Any, optional): The type of document encoder.
            fields (List[str], optional): The document fields to be encoded.
            to_faiss (bool, optional): Store as a FAISS index.
            device (str, optional): The device to be used for encoding.
            delimiter (str, optional): Delimiter for document separation.
            docid_field (str, optional): Field in the document containing document id.
            batch_size (int, optional): Batch size for encoding.
            max_length (int, optional): Maximum length of the input sequence.
            dimension (int, optional): Dimensionality of the encoding.
            prefix (str, optional): Prefix to add to each document.
            pooling (str, optional): Pooling strategy for document encoding.
            l2_norm (bool, optional): Whether to apply L2 normalization.
            use_openai (bool, optional): Whether to use OpenAI's encoder.
            rate_limit (int, optional): Rate limit for OpenAI API requests.
        """
        pass

    @abc.abstractmethod
    def encode(
        self,
        document_path: str,
        index_encoder: str,
        fields: list,
        delimiter="\n",
        docid_field=None,
        batch_size=64,
        max_length=256,
        dimension=768,
        prefix=None,
        pooling=None,
        l2_norm=None,
        to_faiss=False,
        device="cpu",
        use_openai=False,
        rate_limit=3500,
    ):
        """Encodes documents using the specified parameters.

        Args:
            document_path (str): The path to the document file.
            index_encoder (str): The type of document encoder.
            fields (List[str], optional): The document fields to be encoded.
            delimiter (str, optional): Delimiter for document separation.
            docid_field (str, optional): Field in the document containing document id.
            batch_size (int, optional): Batch size for encoding.
            max_length (int, optional): Maximum length of the input sequence.
            dimension (int, optional): Dimensionality of the encoding.
            prefix (str, optional): Prefix to add to each document.
            pooling (str, optional): Pooling strategy for document encoding.
            l2_norm (bool, optional): Whether to apply L2 normalization.
            to_faiss (bool, optional): Whether to store as a FAISS index.
            device (str, optional): The device to be used for encoding.
            use_openai (bool, optional): Whether to use OpenAI's encoder.
            rate_limit (int, optional): Rate limit for OpenAI API requests.
        """
        pass
