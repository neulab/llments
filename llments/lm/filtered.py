from llments.filter.filter import Filter
from llments.lm.lm import LanguageModel


class FilteredLanguageModel(LanguageModel):
    def __init__(self, base: LanguageModel, filter: Filter):
        """Filter down the model's space according to a filtering rule.

        Args:
            base: The language model to be modified.

        Returns:
            The filtered language model.
        """
        raise NotImplementedError("This is not implemented yet.")
