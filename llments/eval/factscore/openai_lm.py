"""OpenAI Model Module."""
from llments.eval.factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict, Any, cast

class OpenAIModel(LM):
    """OpenAI Language Model Class.

    This class extends the `LM` base class to interface with OpenAI's language models, including ChatGPT
    and InstructGPT. It handles API key management, text generation via the OpenAI API, and caching of
    generated outputs to optimize performance.
    
    Attributes:
        model_name (str): Name of the OpenAI model to use (e.g., "ChatGPT", "InstructGPT").
        key_path (str): Path to the file containing the OpenAI API key.
        temp (float): Temperature parameter for text generation, controlling randomness.
        save_interval (int): Interval at which the cache is saved to disk.
    """
    def __init__(
        self,
        model_name: str,
        cache_file: str,
        key_path: str = "api.key"
    ) -> None:
        """Initialize the OpenAIModel instance.

        Args:
            model_name (str): Name of the OpenAI model to use (e.g., "ChatGPT", "InstructGPT").
            cache_file (str): Path to the cache file for storing generated outputs.
            key_path (str, optional): Path to the file containing the OpenAI API key. Defaults to "api.key".
        """
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self) -> None:
        """Load the OpenAI API key and set the model name.

        This method reads the API key from the specified file and configures the OpenAI API client.
        It also sets the `model` attribute to the specified `model_name`.
        
        Raises:
            AssertionError: If the API key file does not exist.
        """
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        openai.api_key = api_key.strip()
        self.model = cast(str, self.model_name)

    def _generate(
        self,
        prompt: str,
        sample_idx: int = 0,
        max_sequence_length: int = 2048,
        max_output_length: int = 128,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text using the OpenAI API based on the input prompt.

        This method handles caching of generated outputs and interacts with the OpenAI API to produce
        text completions. It supports different models like ChatGPT and InstructGPT.

        Args:
            prompt (str): The input prompt for text generation.
            sample_idx (int, optional): Index to differentiate between samples. Defaults to 0.
            max_sequence_length (int, optional): Maximum length of the input sequence. Defaults to 2048.
            max_output_length (int, optional): Maximum length of the generated output. Defaults to 128.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple containing the generated text and the raw API response.

        Raises:
            NotImplementedError: If the specified `model_name` is not supported.
        """
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(message, temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            assert response is not None, "API response is None"
            output = cast(str,response["choices"][0]["message"]["content"])
            return output, response
        elif self.model_name == "InstructGPT":
            # Call API
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(message, temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            assert response is not None, "API response is None"
            output = response["choices"][0]["message"]["content"]
            return output, response
        else:
            raise NotImplementedError()

def call_ChatGPT(
    message: List[Dict[str, str]],
    model_name: str = "gpt-3.5-turbo",
    max_len: int = 1024,
    temp: float = 0.7,
    verbose: bool = False
) -> Dict[str, Any] | None:
    """Call the OpenAI ChatCompletion API to generate a response based on the input message.

    Args:
        message (List[Dict[str, str]]): The input message(s) to send to the ChatCompletion API.
        model_name (str, optional): The OpenAI model to use for text generation. Defaults to "gpt-3.5-turbo".
        max_len (int, optional): Maximum number of tokens to generate. Defaults to 1024.
        temp (float, optional): Temperature parameter for text generation, controlling randomness. Defaults to 0.7.
        verbose (bool, optional): If True, print detailed error information. Defaults to False.

    Returns:
        Dict[str, Any]: The raw response from the OpenAI ChatCompletion API.

    Raises:
        AssertionError: If an InvalidRequestError occurs, such as when the prompt is too long.
    """
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.ChatCompletion.create(model=model_name,
                                                    messages=message,
                                                    max_tokens=max_len,
                                                    temperature=temp)
            received = True
        except:
            # print(message)
            num_rate_errors += 1
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                assert False
            
            logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(
    prompt: str,
    model_name: str = "text-davinci-003",
    max_len: int = 512,
    temp: float = 0.7,
    num_log_probs: int = 0,
    echo: bool = False,
    verbose: bool = False
) -> Any | None:
    """Call the OpenAI GPT-3 API to generate a response based on the input prompt.

    This function handles API rate limits by implementing an exponential backoff retry mechanism.
    It continues to retry until a successful response is received or a critical error occurs.

    Args:
        prompt (str): The input prompt for text generation.
        model_name (str, optional): The OpenAI model to use for text generation. Defaults to "text-davinci-003".
        max_len (int, optional): Maximum number of tokens to generate. Defaults to 512.
        temp (float, optional): Temperature parameter for text generation, controlling randomness. Defaults to 0.7.
        num_log_probs (int, optional): Number of log probabilities to return. Defaults to 0.
        echo (bool, optional): If True, the prompt is echoed in the generated output. Defaults to False.
        verbose (bool, optional): If True, print detailed error information. Defaults to False.

    Returns:
        Any: The raw response from the OpenAI GPT-3 API.

    Raises:
        AssertionError: If an InvalidRequestError occurs, such as when the prompt is too long.
    """
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.Completion.create(model=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo)
            received = True
        except:
            error = sys.exc_info()[0]
            num_rate_errors += 1
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API error: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response
