"""Module for fitting language models to other language models."""

from llments.lm.lm import LanguageModel


class FitLanguageModel(LanguageModel):
    """A language model that is fitted to match another language model."""

    def __init__(self, base: LanguageModel):
        """Fit a language model to match another language model.

        Args:
            base: The language model to be modified.

        Returns:
            LanguageModel: The fitted language model.
        """
        raise NotImplementedError("This is not implemented yet.")
