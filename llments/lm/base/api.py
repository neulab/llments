"""Base class for API-Based Language Models."""

import os
from litellm import completion

"""Set ENV variables.

The language model's API key must be specified as an environment variable.
The name of the environment variable depends on the model in use.
For example: For OpenAI models the API key is set as OPENAI_API_KEY
"""

os.environ["API_KEY"] = "your-api-key"

class APIBasedLM:
    """Base class for API-Based Language Models.
    
    This class represents an API-based language model that generates responses
    using a specified API key and model name.

    Attributes:
        model_name (str): The name of the language model.
    """
    
    def __init__(self, model_name: str) -> None:
        """Initialize the APIBasedLM instance.

        Args:
            model_name (str): The name of the language model.
        """
        self.model_name = model_name
    
    def generate_response(self, prompt: str) -> str:
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
        return str(response['choices'][0]['message']['content'])
