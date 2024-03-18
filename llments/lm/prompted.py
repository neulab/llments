"""An LM that modifies the distribution of a base model using a prompt."""

from llments.lm.lm import LanguageModel


class PromptedLanguageModel(LanguageModel):
    """An LM that modifies the distribution of a base model using a prompt."""

    def __init__(self, base: LanguageModel, prompt: str):
        """Modify the base model's distribution using a textual prompt.

        Args:
            base: The language model to be modified.
            prompt: The prompt to be used to modify the base model.
        """
        raise NotImplementedError("This is not implemented yet.")
