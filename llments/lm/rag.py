"""Module for RAG language model."""

import json
from llments.datastore.datastore import Datastore
from llments.lm.lm import LanguageModel

class RAGLanguageModel(LanguageModel):
    """RAGLanguageModel class for performing Retrieval Augmented Generation."""

    def __init__(
        self,
        base: LanguageModel,
        datastore: Datastore,
        max_results: int = 1,
        query_encoder: str | None=None,
        device: str = "cpu",
        pooling: str = "cls",
        l2_norm: bool = False,
    ) -> None:
        """Apply retrieval-augmented generation over a datastore.

        Args:
            base (LanguageModel): The base language model to be modified.
            datastore (Datastore): The datastore object for document index.
            max_results (int, optional): Maximum number of results to retrieve. Defaults to 1.
            query_encoder (str, optional): Name of the encoder to be used if searcher is "faiss". Defaults to None.
            device (str, optional): Device to be used for encoding. Defaults to cpu.
            pooling (str, optional): Type of pooling to be used for encoding. Defaults to cls.
            l2_norm (bool, optional): Whether to apply L2 normalization to embeddings. Defaults to False.
        """
        self.base = base
        self.datastore = datastore
        self.doc_dict = RAGLanguageModel.read_jsonl_to_dict(datastore.document_path)
        self.max_results = max_results
        self.query_encoder = query_encoder
        self.device = device
        self.pooling = pooling
        self.l2_norm = l2_norm

    @staticmethod
    def read_jsonl_to_dict(
        file_path: str
    ) -> dict[str, str]:
        """Read JSONL file and convert it into a dictionary with document ID as keys and contents as values.

        Args:
            file_path (str): Path to the JSONL file.

        Returns:
            dict: Dictionary containing document contents with document ID as keys.
        """
        data_dict = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_data = json.loads(line)
                data_dict[str(json_data['id'])] = json_data['contents']
        return data_dict
    
    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
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
            str: output sequence from the language model.
        """
        top_docs = self.datastore.retrieve(
            condition,
            max_results=self.max_results,
            query_encoder=self.query_encoder,
            device=self.device,
            pooling=self.pooling,
            l2_norm=self.l2_norm,
        )

        context = ' '.join([self.doc_dict[str(key.docid)] for key in top_docs])
        prompt = None
        if condition is not None:
            prompt = "Please answer the following question, given its context.\nQuestion: " + condition + "\nContext: " + context + "\nAnswer: "
        
        lm_response = self.base.generate(
            condition=prompt,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )

        processed_responses = [x.split("Answer: ")[1].strip() for x in lm_response]
        return processed_responses
    

        

        

