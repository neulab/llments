from llments.lm.lm import LanguageModel
from transformers import pipeline, set_seed


class HuggingFaceLM(LanguageModel):
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
        self.text_generator = pipeline("text-generation", model=model, device=device)

    def fit(
        self, target: LanguageModel, task_description: str | None = None
    ) -> LanguageModel:
        """Fit the language model to a target language model's distribution.

        Args:
            target: The language model that should be fitted to.
            task_description: A task description that explains more about
              what the language model that should be fit is doing (a prompt).

        Returns:
            The fitted language model.
        """
        raise NotImplementedError("This is not implemented yet.")

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
        )
        return [res["generated_text"] for res in results]

    def set_seed(self, seed: int):
        """Set the seed for the language model.

        Args:
            seed: The seed to set for the language model.
        """
        set_seed(seed)


def load_from_spec(spec_file: str) -> HuggingFaceLM:
    """Load a language model from a specification file.

    Args:
        spec_file: The path to the specification file.

    Returns:
        A language model.
    """
    raise NotImplementedError("This is not implemented yet.")
