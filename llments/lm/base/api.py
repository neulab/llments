"""Base class for API-Based Language Models."""

import abc
import os
from typing import Callable
import warnings

from litellm import ModelResponse, batch_completion, completion
import torch

from llments.lm.lm import LanguageModel


class APIBasedLM(LanguageModel):
    """Base class for API-Based Language Models.

    Represents a language model that interacts with an API for generating responses.

    Usage:
    - Instantiate this class with the model name and the api endpoint
    - Set the API key of the language model as an environment
      variable for secure access.

    Attributes:
        model_name (str): The name of the language model.
        api_base (str): The API endpoint to call the model.
    """

    @abc.abstractmethod
    def calculate_probability(self, condition: str | None, output: str) -> float:
        """Calculate the probability of an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
                If None, the output is not conditioned.
            output: The output sequence for which the probability is calculated.

        Returns:
            float: The probability of output x given the language model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __init__(self, model_name: str, api_base: str) -> None:
        """Initialize the APIBasedLM instance.

        Args:
            model_name (str): The name of the language model.
            api_base (str): The API endpoint to call the model.
        """
        self.model_name = model_name
        self.api_base = api_base

    @abc.abstractmethod
    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]]
        | None = None,
    ) -> list[str]:
        """Generate a response based on the given prompt.

        This method sends a prompt to the language model API and retrieves
        the generated response.

        Args:
            condition (str): The conditioning sequence for the output.
                If None, the output is not conditioned.
            do_sample (bool): Whether to use sampling or greedy decoding.
            max_length (int): The maximum length of the output sequence,
                (defaults to model max).
            max_new_tokens (float): The maximum number of tokens to generate in the chat completion.
            temperature (float): The sampling temperature to be used, between 0 and 2.
            num_return_sequences (int): The number of chat completion choices to generate for each input message.
            prefix_allowed_tokens_fn: This argument is not supported for API-based language models.

        Returns:
            str: Sampled output sequences from the language model.
        """
        if condition is not None:
            warnings.warn(
                "A non-default value for 'condition' was provided.", UserWarning
            )
        if do_sample:
            warnings.warn(
                "A non-default value for 'do_sample' was provided.", UserWarning
            )
        if max_length is not None:
            warnings.warn(
                "A non-default value for 'max_length' was provided.", UserWarning
            )
        if prefix_allowed_tokens_fn is not None:
            raise NotImplementedError(
                "The 'prefix_allowed_tokens_fn' argument is not supported for API-based language models."
            )

        responses = []
        response = completion(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
            api_base=self.api_base,
            messages=[{"content": condition, "role": "user"}],
        )
        for choice in response["choices"]:
            responses.append(choice["message"]["content"])
        return responses

    @abc.abstractmethod
    def chat_generate(
        self,
        messages: list[dict[str, str]],
        do_sample: bool = False,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> list[list[dict[str, str]]]:
        """Generate responses to multiple prompts using the batch_completion function.

        This method sends multiple prompts to the language model API and retrieves
        the generated response for each of the prompts.

        Args:
            messages: A list of dictionaries, each representing a message in the chat context. Each dictionary should contain the following keys:
            - "role": The role of the entity sending the message. This can be "system", "user", etc.
            - "content": The actual content of the message. Example:
            [
                {
                    "role": "system",
                    "content": "You are a friendly chatbot",
                },
                {
                    "role": "user",
                    "content": "How many helicopters can a human eat in one sitting?"
                },
            ]
            do_sample (bool): Whether to use sampling or greedy decoding.
            max_length (int): The maximum length of the output sequence,
                (defaults to model max).
            max_new_tokens (float): The maximum number of tokens to generate in the chat completion.
            temperature (float): The sampling temperature to be used, between 0 and 2.
            num_return_sequences (int): The number of chat completion choices to generate for each input message.

        Returns:
            list[list[dict[str, str]]]: list of chat contexts with the generated responses.
        """
        if do_sample:
            warnings.warn(
                "A non-default value for 'do_sample' was provided.", UserWarning
            )
        if max_length is not None:
            warnings.warn(
                "A non-default value for 'max_length' was provided.", UserWarning
            )

        responses = batch_completion(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=num_return_sequences,
            api_base=self.api_base,
            messages=[messages],
        )
        responses = [r["message"]["content"] for r in responses[0]["choices"]]
        return [messages + [{"role": "assistant", "content": r}] for r in responses]

    @abc.abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set the seed for the language model.

        Args:
            seed: The seed to set for the language model.
        """
        raise NotImplementedError
