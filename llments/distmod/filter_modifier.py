from .distmod import DistributionModifier
from llments.lm.lm import LanguageModel


class FilterModifier(DistributionModifier):
    def __call__(self, base: LanguageModel, **kwargs) -> LanguageModel:
        """Filter down the model's space according to a filtering rule.

        Args:
            base (LanguageModel): The language model to be modified.

        Returns:
            LanguageModel: The filtered language model.
        """
        # filtering_rule = kwargs.get("filtering_rule", None)
        # Filtering implementation
        raise NotImplementedError("This is not implemented yet.")
