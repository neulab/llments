from .distmod import DistributionModifier
from llments.lm.lm import LanguageModel


class EnsembleModifier(DistributionModifier):
    def __call__(self, base: LanguageModel, **kwargs) -> LanguageModel:
        """Combine several models into one by ensembling their outputs based on specified weights.

        Args:
            base (LanguageModel): The language model to be modified.

        Returns:
            LanguageModel: A new ensembled language model.
        """
        # models = kwargs.get("models", None)
        # weights = kwargs.get("weights", None)
        # Ensembling implementation
        raise NotImplementedError("This is not implemented yet.")
