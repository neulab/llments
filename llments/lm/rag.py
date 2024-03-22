from llments.datastore.pyserini_datastore import PyseriniDatastore
from llments.lm.lm import LanguageModel


class RAGLanguageModel(LanguageModel):
    def __init__(self, base: LanguageModel, document_path: str, index_path: str, index_encoder: str, fields: list, 
                 to_faiss: bool, device: str, delimiter="\n", docid_field=None, batch_size=64, max_length=256, 
                 dimension=768, prefix=None, pooling=None, l2_norm=None, use_openai=False, rate_limit=3500):
        """Apply retrieval-augmented generation over a datastore.

        Args:
            base: The language model to be modified.
            document_path: The path to the document file
            index_path: The path to store the generated index
            index_encoder: The type of document encoder
            fields: The document fields to be encoded
            to_faiss: Store as a FAISS index
            device: The device to be used for encoding
            delimiter: Delimiter for document separation
            docid_field: Field in the document containing document id
            batch_size: Batch size for encoding
            max_length: Maximum length of the input sequence
            dimension: Dimensionality of the encoding
            prefix: Prefix to add to each document
            pooling: Pooling strategy for document encoding
            l2_norm: Whether to apply L2 normalization

        Returns:
            LanguageModel: The enhanced language model.
        """
        print("Creating Datastore...")
        pyserini_encoder = PyseriniDatastore(document_path, index_path, index_encoder, fields, device)
        pyserini_encoder.encode(delimiter=delimiter, docid_field=docid_field, batch_size=batch_size, max_length=max_length, 
                                dimension=dimension, prefix=prefix, pooling=pooling, l2_norm=l2_norm, to_faiss=to_faiss, 
                                use_openai=use_openai, rate_limit=rate_limit)
        print("Datastore creation completed successfully!")

