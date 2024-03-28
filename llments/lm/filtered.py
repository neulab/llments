"""Module for filtered language models."""

from llments.filter.filter import Filter
from llments.lm.lm import LanguageModel


class FilteredLanguageModel(LanguageModel):
    """A language model that is filtered down according to a filtering rule."""

    def __init__(self, base: LanguageModel, filter: Filter):
        """Filter down the model's space according to a filtering rule.

        Args:
            base: The language model to be modified.
            filter: The filter to be used to filter down the model's space.

        Returns:
            The filtered language model.
        """
        raise NotImplementedError("This is not implemented yet.")
