"""Base class for API-Based Language Models."""

import os
from litellm import acompletion

class APIBasedLM:
    """Base class for API-Based Language Models.

    Represents a language model that interacts with an API for generating responses.

    Usage:
    - Instantiate this class with the model name.
    - Set the API key of the language model as an environment 
      variable for secure access.

    Attributes:
        model_name (str): The name of the language model.
    """
    
    def __init__(self, model_name: str) -> None:
        """Initialize the APIBasedLM instance.

        Args:
            model_name (str): The name of the language model.
        """
        self.model_name = model_name
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response based on the given prompt.

        This method sends a prompt to the language model API and retrieves
        the generated response.

        To handle potential rate limitations, this method uses the asynchronous
        version of the completion function, which allows for concurrent function calls.

        Args:
            prompt (str): The prompt for generating a response.

        Returns:
            str: The generated response from the language model.
        """
        response = await acompletion(
            model=self.model_name,
            messages=[{"content": prompt, "role": "user"}]
        )
        return str(response['choices'][0]['message']['content'])
