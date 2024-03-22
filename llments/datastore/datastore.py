import abc

class Datastore:
    @abc.abstractmethod
    def __init__(self, input_jsonl, output_dir, encoder, fields, device):
        """Initialize a Datastore object.

        Args:
            input_jsonl (str): Path to the JSONL document file.
            output_dir (str): Directory to store the encoded representations.
            encoder (str): The type of document encoder to use.
            fields (list): The document fields to be encoded.
            device (str): The device to be used for encoding.
        """
        pass

    @abc.abstractmethod
    def encode(self, delimiter="\n", docid_field=None, batch_size=64, max_length=256, dimension=768, 
               prefix=None, pooling=None, l2_norm=None, to_faiss=False, use_openai=False, rate_limit=3500):
        """Encode documents using the specified parameters.

        Args:
            delimiter (str, optional): Delimiter for document separation. Defaults to "\n".
            docid_field (str, optional): Field in the document containing document id. Defaults to None.
            batch_size (int, optional): Batch size for encoding. Defaults to 64.
            max_length (int, optional): Maximum length of the input sequence. Defaults to 256.
            dimension (int, optional): Dimensionality of the encoding. Defaults to 768.
            prefix (str, optional): Prefix to add to each document. Defaults to None.
            pooling (str, optional): Pooling strategy for document encoding. Defaults to 'cls'.
            l2_norm (bool, optional): Whether to apply L2 normalization. Defaults to False.
            to_faiss (bool, optional): Whether to store as a FAISS index. Defaults to False.
            use_openai (bool, optional): Whether to use OpenAI API. Defaults to False.
            rate_limit (int, optional): Rate limit for OpenAI API. Defaults to 3500.
        """
        pass
