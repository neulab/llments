"""LM (Language Model) Base Class Module."""
import pickle
import os
import time
from typing import Dict, Any

class LM(object):
    """LM (Language Model) Base Class.

    This class serves as a base for language models, managing caching of generated outputs
    and defining the interface for loading models and generating text. It handles the storage
    and retrieval of cached responses to optimize performance.

    Attributes:
        cache_file (str): Path to the cache file for storing generated outputs.
        cache_dict (Dict[str, Any]): Dictionary storing cached responses.
        model (Optional[Any]): The language model instance.
        add_n (int): Counter for the number of new cache entries added.
    """
    def __init__(self, cache_file: str) -> None:
        """Initialize the LM (Language Model) instance.

        Args:
            cache_file (str): Path to the cache file for storing generated outputs.
        """
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model = None
        self.add_n = 0

    def load_model(self) -> None:
        """Load the language model and put it as self.model.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError()

    def _generate(
        self,
        prompt: str,
        sample_idx: int = 0,
        max_sequence_length: int = 2048,
        max_output_length: int = 128,
    ) -> Any:
        """Generate text based on the input prompt.

        Args:
            prompt (str): The input prompt to generate text from.
            sample_idx (int, optional): Index to differentiate between samples. Defaults to 0.
            max_sequence_length (int, optional): Maximum length of the input sequence. Defaults to 2048.
            max_output_length (int, optional): Maximum length of the generated output. Defaults to 128.

        Returns:
            Any: The generated text or model output.
        """
        prompt = prompt.strip() # it's important not to end with a whitespace
        cache_key = f"{prompt}_{sample_idx}"

        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        if self.model is None:
            self.load_model()

        if prompt.endswith(" True or False?\nAnswer:"):
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
        else:
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        self.cache_dict[cache_key] = generated
        self.add_n += 1
        return generated

    def save_cache(self) -> None:
        """Save the current cache to the cache file."""
        if self.add_n == 0:
            return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry: bool = True) -> Any:
        """Load the cache from the cache file.

        Args:
            allow_retry (bool, optional): Whether to retry loading the cache in case of errors.
                Defaults to True.

        Returns:
            Any: The loaded cache dictionary.

        Raises:
            Exception: Propagates the exception if `allow_retry` is False and loading fails.
        """
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache
