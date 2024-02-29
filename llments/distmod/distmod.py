from llments.lm.lm import LanguageModel


class DistributionModifier:
    def __call__(self, base: LanguageModel, **kwargs) -> LanguageModel:
        """Modify the base model's distribution based on the specific modifier operator.

        Args:
            base (LanguageModel): The language model to be modified.

        Returns:
            LanguageModel: The modified language model.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
