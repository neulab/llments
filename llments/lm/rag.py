import json
from llments.datastore.datastore import Datastore
from llments.lm.lm import LanguageModel

class RAGLanguageModel(LanguageModel):
    def __init__(self, base: LanguageModel, datastore: Datastore, searcher_name: str, encoder_name=None):
        """Apply retrieval-augmented generation over a datastore.

        Args:
            base (LanguageModel): The base language model to be modified.
            datastore (Datastore): The datastore object for document index.
            searcher_name (str): Name of the searcher to be used (e.g., "bm25" or "faiss").
            encoder_name (str, optional): Name of the encoder to be used if searcher is "faiss". Defaults to None.

        Attributes:
            datastore (Datastore): The datastore object for document index.
            searcher_name (str): Name of the searcher to be used.
            encoder: The encoder object used for encoding queries (specifically for "faiss" searcher).
            searcher: The initialized searcher object.
            doc_dict (dict): Dictionary containing document contents with document ID as keys.
        """
        self.datastore = datastore
        self.searcher_name = searcher_name
        self.encoder = None
        self.searcher = self.initialize_retriever(searcher_name, encoder_name)
        self.doc_dict = RAGLanguageModel.read_jsonl_to_dict(datastore.document_path)

    @staticmethod
    def read_jsonl_to_dict(file_path):
        """
        Read JSONL file and convert it into a dictionary with document ID as keys and contents as values.

        Args:
            file_path (str): Path to the JSONL file.

        Returns:
            dict: Dictionary containing document contents with document ID as keys.

        """
        data_dict = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                data_dict[json_data['id']] = json_data['contents']
        return data_dict

    def initialize_retriever(self, encoder_name=None, device='cpu', pooling='cls', l2_norm=False):
        """
        Initialize the document retriever based on the specified searcher name.

        Args:
            encoder_name (str, optional): Name of the encoder to be used if searcher is "faiss". Defaults to None.
            device (str, optional): Device to be used for encoding. Defaults to 'cpu'.
            pooling (str, optional): Type of pooling to be used for encoding. Defaults to 'cls'.
            l2_norm (bool, optional): Whether to apply L2 normalization to embeddings. Defaults to False.

        Returns:
            object: Initialized searcher object.
        """
        if self.searcher_name == "bm25":
            try:
                from pyserini.search.lucene import LuceneSearcher
            except ImportError:
                raise ImportError(
                    "You need to install the `pyserini` package to use this class."
                )
            searcher = LuceneSearcher(self.datastore.index_path)
        
        elif self.searcher_name == "faiss":
            if encoder_name is None:
                raise ValueError("Please enter an encoder name.")
            try:
                from pyserini.search import FaissSearcher
                from pyserini.search.faiss import AutoQueryEncoder
            except ImportError:
                raise ImportError(
                    "You need to install the `pyserini` package to use this class."
                )
            self.encoder = AutoQueryEncoder(encoder_dir=encoder_name, device=device, pooling=pooling, l2_norm=l2_norm)
            searcher = FaissSearcher(self.datastore.index_path, self.encoder)

        else:
            raise ValueError("Please enter a valid searcher name.")
        
        return searcher

    def generate(self, condition: str | None, do_sample: bool = False, max_length: int | None = None, temperature: float = 1, num_return_sequences: int = 1) -> list[str]:
        """Generate an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
                If None, the output is not conditioned.
            do_sample: Whether to use sampling or greedy decoding.
            max_length: The maximum length of the output sequence,
                (defaults to model max).
            temperature: The value used to module the next token probabilities.
            num_return_sequences: The number of independently computed returned
                sequences for each element in the batch.

        Returns:
            str: Sampled output sequences from the language model.
        """
        if self.searcher_name == "faiss":
            query = self.encoder.encode(query)
        hits = self.searcher.search(query, k = 1)
        top_docid = hits[0].docid
        context = self.doc_dict[top_docid]
        return [context]

        

        

