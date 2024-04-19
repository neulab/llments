"""A module for PyseriniDatastore class."""

import os
from llments.datastore.datastore import Datastore

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pyserini.search.faiss import DenseSearchResult

class PyseriniDatastore(Datastore):
    """A PyseriniDatastore containing data for retrieval."""

    def __init__(
        self,
        index_path: str,
        document_path: str,
        index_encoder: str,
        fields: list[str],
        to_faiss: bool = False,
        device: str = "cpu",
        delimiter: str = "\n",
        docid_field: str = "id",
        batch_size: int = 64,
        max_length: int = 256,
        dimension: int = 768,
        prefix: str | None = None,
        pooling: str = "cls",
        l2_norm: bool = False,
        use_openai: bool = False,
        rate_limit: int = 3500,
        shard_id: int = 0,
        shard_num: int = 1,
    ):
        """Initializes a PyseriniDatastore object.

        Args:
            index_path (str): The path to store the generated index.
            document_path (str): The path to the document file.
            index_encoder (str): The type of document encoder.
            fields (list[str]): The document fields to be encoded.
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
            shard_id (int, optional): id of shards.
            shard_num (int, optional): number of shards.
        """
        self.index_path = index_path
        self.document_path = document_path
        self.device = device
        self.l2_norm = l2_norm
        self.pooling = pooling
        self.index_encoder = index_encoder

        if not os.path.exists(index_path):
            print("Creating the Datastore...")
            self.encode(
                index_encoder=index_encoder,
                fields=fields,
                delimiter=delimiter,
                docid_field=docid_field,
                batch_size=batch_size,
                max_length=max_length,
                dimension=dimension,
                prefix=prefix,
                pooling=pooling,
                l2_norm=l2_norm,
                to_faiss=to_faiss,
                device=device,
                use_openai=use_openai,
                rate_limit=rate_limit,
                shard_id=shard_id,
                shard_num=shard_num,
            )
        elif not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index path '{index_path}' does not exist. You have to create an index first."
            )

    def encode(
        self,
        index_encoder: str,
        fields: list[str],
        to_faiss: bool = False,
        device: str = "cpu",
        delimiter: str = "\n",
        docid_field: str = "id",
        batch_size: int = 64,
        max_length: int = 256,
        dimension: int = 768,
        prefix: str | None = None,
        pooling: str = "cls",
        l2_norm: bool = False,
        use_openai: bool = False,
        rate_limit: int = 3500,
        shard_id: int = 0,
        shard_num: int = 1,
    ) -> None:
        """Encodes documents using the specified parameters.

        Args:
            index_encoder (str): The type of document encoder.
            fields (List[str], optional): The document fields to be encoded.
            to_faiss (bool, optional): Whether to store as a FAISS index.
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
            shard_id (int, optional): id of shards.
            shard_num (int, optional): number of shards.
        """
        if index_encoder is None or fields is None:
            raise ValueError(
                "index_encoder and fields are required parameters."
            )

        try:
            from pyserini.encode import JsonlCollectionIterator
        except ImportError:
            raise ImportError(
                "You need to install the `pyserini` package to use this class."
            )

        encoder_class_map = {
            "dpr": "DprDocumentEncoder",
            "tct_colbert": "TctColBertDocumentEncoder",
            "aggretriever": "AggretrieverDocumentEncoder",
            "ance": "AnceDocumentEncoder",
            "sentence-transformers": "AutoDocumentEncoder",
            "unicoil": "UniCoilDocumentEncoder",
            "openai-api": "OpenAIDocumentEncoder",
            "cosdpr": "CosDprDocumentEncoder",
            "auto": "AutoDocumentEncoder",
        }

        _encoder_class = None
        encoder_class = None

        for class_keyword in encoder_class_map:
            if class_keyword in index_encoder.lower():
                try:
                    module = __import__(
                        "pyserini.encode", fromlist=[encoder_class_map[class_keyword]]
                    )
                    encoder_class = getattr(module, encoder_class_map[class_keyword])
                    _encoder_class = encoder_class_map[class_keyword]
                except ImportError:
                    raise ImportError(
                        "You need to install the `pyserini` package to use this class."
                    )
                break

        # if none of the class keyword was matched, use the AutoDocumentEncoder
        if encoder_class is None:
            try:
                from pyserini.encode import AutoDocumentEncoder
            except ImportError:
                raise ImportError(
                    "You need to install the `pyserini` package to use this class."
                )
            encoder_class = AutoDocumentEncoder
            _encoder_class = "AutoDocumentEncoder"

        if pooling is None:
            if "sentence-transformers" in index_encoder:
                pooling = "mean"
            elif "contriever" in index_encoder:
                pooling = "mean"
            else:
                pooling = "cls"

        if l2_norm is None:
            if "sentence-transformers" in index_encoder:
                l2_norm = True
            elif "contriever" in index_encoder:
                l2_norm = False
            else:
                l2_norm = False

        print("Initializing the document encoder ...")

        if _encoder_class == "AutoDocumentEncoder":
            encoder_instance = encoder_class(
                model_name=index_encoder,
                device=device,
                pooling=pooling,
                l2_norm=l2_norm,
                prefix=prefix,
            )
        else:
            encoder_instance = encoder_class(model_name=index_encoder, device=device)

        if to_faiss:
            try:
                from pyserini.encode import FaissRepresentationWriter
            except ImportError:
                raise ImportError(
                    "You need to install the `pyserini` package to use this class."
                )
            embedding_writer = FaissRepresentationWriter(
                self.index_path, dimension=dimension
            )
        else:
            try:
                from pyserini.encode import JsonlRepresentationWriter
            except ImportError:
                raise ImportError(
                    "You need to install the `pyserini` package to use this class."
                )
            embedding_writer = JsonlRepresentationWriter(self.index_path)

        collection_iterator = JsonlCollectionIterator(
            self.document_path, fields, docid_field, delimiter
        )

        if use_openai:
            try:
                from pyserini.encode import OPENAI_API_RETRY_DELAY
            except ImportError:
                raise ImportError(
                    "You need to install the `pyserini` package to use this class."
                )
            batch_size = int(rate_limit / (60 / OPENAI_API_RETRY_DELAY))

        print("Building the index ...")

        with embedding_writer:
            for batch_info in collection_iterator(batch_size, shard_id, shard_num):
                texts = batch_info[fields[0]]
                titles = batch_info["title"] if "title" in fields else None
                expands = batch_info["expand"] if "expand" in fields else None
                fp16 = False
                max_length = max_length
                add_sep = False

                embeddings = encoder_instance.encode(
                    texts=texts,
                    titles=titles,
                    expands=expands,
                    fp16=fp16,
                    max_length=max_length,
                    add_sep=add_sep,
                )
                batch_info["vector"] = embeddings
                embedding_writer.write(batch_info, fields)

        print("\nIndex creation completed sucessfully!")

    def retrieve(
        self,
        query: str | None,
        max_results: int,
    ) -> list[DenseSearchResult]:
        """Retrieve documents based on the specified searcher name.

        Args:
            query (str): Query string to search for.
            max_results (int): Maximum number of results to retrieve.

        Returns:
            list[DenseSearchResult]: Retrieved result objects.
        """
        try:
            from pyserini.search import FaissSearcher
            from pyserini.search.faiss import AutoQueryEncoder
        except ImportError:
            raise ImportError(
                "You need to install the `pyserini` package to use this class."
            )
        encoder = AutoQueryEncoder(encoder_dir=self.index_encoder, device=self.device, pooling=self.pooling, l2_norm=self.l2_norm)
        searcher = FaissSearcher(self.index_path, encoder)
        hits = searcher.search(query, k=max_results)
        return hits
    