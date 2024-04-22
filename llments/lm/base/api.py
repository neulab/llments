"""Base class for API-Based Language Models."""

import os
import abc
from llments.lm.lm import LanguageModel
from litellm import acompletion, batch_completion, ModelResponse

class APIBasedLM(LanguageModel):
    """Base class for API-Based Language Models.
    
    Represents a language model that interacts with an API for generating responses.
    
    Usage:
    - Instantiate this class with the model name.
    - Set the API key of the language model as an environment 
      variable for secure access.
      
    Attributes:
        model_name (str): The name of the language model.
    """
    
    @abc.abstractmethod
    def __init__(self, model_name: str) -> None:
        """Initialize the APIBasedLM instance.
        
        Args:
            model_name (str): The name of the language model.
        """
        self.model_name = model_name

    @abc.abstractmethod
    async def generate(
        self, 
        temperature: float | None,
        max_tokens: float | None,
        n: int | None,
        message: str | None
        ) -> ModelResponse:
        """Generate a response based on the given prompt.
        
        This method sends a prompt to the language model API and retrieves
        the generated response.
        To handle potential rate limitations, this method uses the asynchronous
        version of the completion function, which allows for concurrent function calls.
        
        Args:
            temperature (float): The sampling temperature to be used, between 0 and 2. 
            max_tokens (float): The maximum number of tokens to generate in the chat completion.
            n (int): The number of chat completion choices to generate for each input message.
            message (str): The prompt for generating a response.
            
        Returns:
            ModelResponse: The generated response object from the language model.
        """
        response = await acompletion(
            model = self.model_name,
            temperature = temperature,
            max_tokens = max_tokens,
            n = n,
            messages=[{"content": message, "role": "user"}]
        )
        return response

    @abc.abstractmethod
    def generate_batch(
        self, 
        temperature: float | None,
        max_tokens: float | None,
        n: int | None,
        messages: list[str]
        ) -> list:
        """Generate responses to multiple prompts using the batch_completion function.
        
        This method sends multiple prompts to the language model API and retrieves
        the generated response for each of the prompts.
        
        Args:
            temperature (float): The sampling temperature to be used, between 0 and 2. 
            max_tokens (float): The maximum number of tokens to generate in the chat completion.
            n (int): The number of chat completion choices to generate for each input message.
            messages (list): List of multiple prompts for generating the responses.
            
        Returns:
            list: List of responses generated by the language model for all the prompts.
        """
        responses = batch_completion(
            model = self.model_name,
            temperature = temperature,
            max_tokens = max_tokens,
            n = n,
            messages=[[{"content": content, "role": "user"}] for content in messages]
        )
        return responses
