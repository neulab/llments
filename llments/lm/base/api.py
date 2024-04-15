import os
from litellm import completion

class APIBasedLM:
    """Base class for API-Based Language Models.
    This class represents an API-based language model that generates responses
    using a specified API key and model name.

    Attributes:
        api_key (str): The API key used for accessing the language model.
        model_name (str): The name of the language model.
    """
    def __init__(self, api_key, model_name) -> None:
        """Initialize the APIBasedLM instance.

        Args:
            api_key (str): The API key used for accessing the language model.
            model_name (str): The name of the language model.
        """
        self.api_key = api_key
        self.model_name = model_name
    
    def generate_response(self, prompt) -> str:
        """Generate a response based on the given prompt.

        This method sends a prompt to the language model API and retrieves
        the generated response.

        Args:
            prompt (str): The prompt for generating a response.

        Returns:
            str: The generated response from the language model.
        """
        response = completion(
            model=self.model_name,
            messages=[{"content": prompt, "role": "user"}]
        )
        return response['choices'][0]['message']['content']
