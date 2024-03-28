import os
from llments.datastore.datastore import Datastore

class PyseriniDatastore(Datastore):
    def __init__(self, index_path: str, document_path: str, index_encoder=None, fields=None, 
                 to_faiss=False, device='cpu', delimiter="\n", docid_field=None, batch_size=64, max_length=256, 
                 dimension=768, prefix=None, pooling=None, l2_norm=None, use_openai=False, rate_limit=3500):
        """
        Initializes a PyseriniDatastore object.

        Args:
            index_path (str): The path to store the generated index.
            document_path (str): The path to the document file
            index_encoder (Any, optional): The type of document encoder. Defaults to None.
            fields (List[str], optional): The document fields to be encoded. Defaults to ['text'].
            to_faiss (bool, optional): Store as a FAISS index. Defaults to False.
            device (str, optional): The device to be used for encoding. Defaults to 'cpu'.
            delimiter (str, optional): Delimiter for document separation. Defaults to "\n".
            docid_field (str, optional): Field in the document containing document id. Defaults to None.
            batch_size (int, optional): Batch size for encoding. Defaults to 64.
            max_length (int, optional): Maximum length of the input sequence. Defaults to 256.
            dimension (int, optional): Dimensionality of the encoding. Defaults to 768.
            prefix (str, optional): Prefix to add to each document. Defaults to None.
            pooling (str, optional): Pooling strategy for document encoding. Defaults to None.
            l2_norm (bool, optional): Whether to apply L2 normalization. Defaults to None.
            use_openai (bool, optional): Whether to use OpenAI's encoder. Defaults to False.
            rate_limit (int, optional): Rate limit for OpenAI API requests. Defaults to 3500.
        """  
        self.index_path = index_path
        self.document_path = document_path

        if not os.path.exists(index_path):
            print("Creating the Datastore...")
            self.encode(index_encoder=index_encoder, fields=fields, delimiter=delimiter,
                        docid_field=docid_field, batch_size=batch_size, max_length=max_length, dimension=dimension, prefix=prefix,
                        pooling=pooling, l2_norm=l2_norm, to_faiss=to_faiss, device=device, use_openai=use_openai, rate_limit=rate_limit)

    def encode(self, index_encoder: str, fields: list, delimiter="\n", docid_field=None,
                batch_size=64, max_length=256, dimension=768, prefix=None, pooling=None, l2_norm=None, to_faiss=False,
                device='cpu', use_openai=False, rate_limit=3500):
        """
        Encodes documents using the specified parameters.

        Args:
            index_encoder (str): The type of document encoder.
            fields (List[str], optional): The document fields to be encoded. Defaults to ['text'].
            delimiter (str, optional): Delimiter for document separation. Defaults to "\n".
            docid_field (str, optional): Field in the document containing document id. Defaults to None.
            batch_size (int, optional): Batch size for encoding. Defaults to 64.
            max_length (int, optional): Maximum length of the input sequence. Defaults to 256.
            dimension (int, optional): Dimensionality of the encoding. Defaults to 768.
            prefix (str, optional): Prefix to add to each document. Defaults to None.
            pooling (str, optional): Pooling strategy for document encoding. Defaults to None.
            l2_norm (bool, optional): Whether to apply L2 normalization. Defaults to None.
            to_faiss (bool, optional): Whether to store as a FAISS index. Defaults to False.
            device (str, optional): The device to be used for encoding. Defaults to 'cpu'.
            use_openai (bool, optional): Whether to use OpenAI's encoder. Defaults to False.
            rate_limit (int, optional): Rate limit for OpenAI API requests. Defaults to 3500.
        """

        if index_encoder is None or fields is None:
            raise ValueError("index_encoder and fields are required parameters.")

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
                    module = __import__('pyserini.encode', fromlist=[encoder_class_map[class_keyword]])
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
                pooling = 'mean'
            elif "contriever" in index_encoder:
                pooling = 'mean'
            else:
                pooling = 'cls'
        
        if l2_norm is None:
            if "sentence-transformers" in index_encoder:
                l2_norm = True
            elif "contriever" in index_encoder:
                l2_norm = False
            else:
                l2_norm = False

        print("Initializing the document encoder ...")

        if _encoder_class == "AutoDocumentEncoder":
            encoder_instance = encoder_class(model_name=index_encoder, device=device, pooling=pooling, l2_norm=l2_norm, prefix=prefix)
        else:
            encoder_instance = encoder_class(model_name = index_encoder, device = device)

        if to_faiss:
            try:
                from pyserini.encode import FaissRepresentationWriter
            except ImportError:
                raise ImportError(
                    "You need to install the `pyserini` package to use this class."
                )
            embedding_writer = FaissRepresentationWriter(self.index_path, dimension=dimension)
        else:
            try:
                from pyserini.encode import JsonlRepresentationWriter
            except ImportError:
                raise ImportError(
                    "You need to install the `pyserini` package to use this class."
                )
            embedding_writer = JsonlRepresentationWriter(self.index_path)

        collection_iterator = JsonlCollectionIterator(self.document_path, fields, docid_field, delimiter)

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
            for batch_info in collection_iterator(batch_size):
                texts = batch_info['text']
                titles = batch_info['title'] if 'title' in fields else None
                expands = batch_info['expand'] if 'expand' in fields else None
                fp16 = False
                max_length = max_length
                add_sep = False
                
                embeddings = encoder_instance.encode(texts=texts, titles=titles, expands=expands, fp16=fp16, max_length=max_length, add_sep=add_sep)
                batch_info['vector'] = embeddings
                embedding_writer.write(batch_info, fields)

        print("\nIndex creation completed sucessfully!")
