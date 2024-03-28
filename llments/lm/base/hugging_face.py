"""Module for HuggingFace language models."""

from llments.lm.lm import LanguageModel
import json


class HuggingFaceLM(LanguageModel):
    """A language model that uses the HuggingFace library."""

    def __init__(
        self,
        model: str,
        device: str | None = None,
    ):
        """Initialize a HuggingFaceLM.

        Args:
            model: The name of the model.
            device: The device to run the model on.
        """
        try:
            from transformers import pipeline, TextGenerationPipeline
        except ImportError:
            raise ImportError(
                "You need to install the `transformers` package to use this class."
            )
        self.text_generator: TextGenerationPipeline = pipeline(
            "text-generation", model=model, device=device
        )

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
            str: A sampled output sequence from the language model.
        """
        results = self.text_generator(
            condition,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            clean_up_tokenization_spaces=True,
            truncation=max_length is not None,
        )
        return [res["generated_text"] for res in results]

    def set_seed(self, seed: int):
        """Set the seed for the language model.

        Args:
            seed: The seed to set for the language model.
        """
        try:
            from transformers import set_seed
        except ImportError:
            raise ImportError(
                "You need to install the `transformers` package to use this class."
            )
        set_seed(seed)

    def calculate_probability(self, output: str) -> float:
        """Calculate the probability of an output given the language model.

        Args:
            output: The output sequence for which the probability is calculated.

        Returns:
            float: The probability of output x given the language model.
        """
        raise NotImplementedError


def load_from_spec(spec_file: str) -> HuggingFaceLM:
    """Load a language model from a specification file.

    Args:
        spec_file: The path to the specification file.
        The file should specifies the model identifier "model" and any other relevant parameters such as "device".

    Returns:
        A HuggingFaceLM instance.
    """
    with open(spec_file, "r") as file:
        spec = json.load(file)

    model_name = spec.get("model")
    device = spec.get("device", None)

    return HuggingFaceLM(model=model_name, device=device)
