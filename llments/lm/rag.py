"""Module for RAG language model."""

import json
import os
from typing import Callable

import torch
from llments.datastore.datastore import Datastore
from llments.lm.lm import LanguageModel


class RAGLanguageModel(LanguageModel):
    """RAGLanguageModel class for performing Retrieval Augmented Generation."""

    def __init__(
        self,
        base: LanguageModel,
        datastore: Datastore,
        max_results: int = 1,
    ) -> None:
        """Apply retrieval-augmented generation over a datastore.

        Args:
            base (LanguageModel): The base language model to be modified.
            datastore (Datastore): The datastore object for document index.
            max_results (int, optional): Maximum number of results to retrieve. Defaults to 1.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "You need to install the `faiss` package to use this class."
            )
        self.base = base
        self.datastore = datastore
        print("Loading the index...")
        self.index = faiss.read_index(
            os.path.join(datastore.index_path, "index"), faiss.IO_FLAG_MMAP
        )
        self.docids = RAGLanguageModel.load_docids(
            os.path.join(datastore.index_path, "docid")
        )
        print("Index loaded successfully!")
        print("Loading the document file...")
        self.doc_dict = self.read_jsonl_to_dict(datastore.document_path)
        print("Documents loaded successfully!")
        self.max_results = max_results

    def set_max_results(self, max_results: int) -> None:
        """Set the max retrieval results for RAG.

        Args:
            max_results (int): The maximum retrieved results for RAG.
        """
        self.max_results = max_results

    def read_jsonl_to_dict(self, file_path: str) -> dict[str, str]:
        """Read JSONL file and convert it into a dictionary with document ID as keys and contents as values.

        Args:
            file_path (str): Path to the JSONL file.

        Returns:
            dict: Dictionary containing document contents with document ID as keys.
        """
        data_dict = {}
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                json_data = json.loads(line)
                data_dict[str(json_data[self.datastore.docid_field])] = json_data[
                    self.datastore.fields[0]
                ]
        return data_dict

    @staticmethod
    def load_docids(file_path: str) -> list[str]:
        """Read docids and convert it into a list.

        Args:
            file_path (str): Path to the docids file.

        Returns:
            dict: List containing document IDs.
        """
        with open(file_path, "r") as file:
            docids = [line.rstrip() for line in file.readlines()]
        return docids

    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1,
        num_return_sequences: int = 1,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]]
        | None = None,
    ) -> list[str]:
        """Generate an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
                If None, the output is not conditioned.
            do_sample: Whether to use sampling or greedy decoding.
            max_length: The maximum length of the output sequence,
                (defaults to model max).
            max_new_tokens: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            temperature: The value used to module the next token probabilities.
            num_return_sequences: The number of independently computed returned
                sequences for each element in the batch.
            prefix_allowed_tokens_fn: This argument is not supported for RAGLanguageModel.

        Returns:
            str: output sequence from the language model.
        """
        if prefix_allowed_tokens_fn is not None:
            raise NotImplementedError(
                "The 'prefix_allowed_tokens_fn' argument is not supported for RAGLanguageModel."
            )

        top_docs = self.datastore.retrieve(
            condition,
            index=self.index,
            docids=self.docids,
            max_results=self.max_results,
        )

        context = "\n".join([self.doc_dict[str(key.docid)] for key in top_docs])
        prompt = None
        if condition is not None:
            prompt = (
                "\nContext: "
                + context
                + "\nPlease answer the following question.\nQuestion: "
                + condition
                + "\nAnswer: "
            )

        lm_response = self.base.generate(
            condition=prompt,
            do_sample=do_sample,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )

        return lm_response
