from .distmod import DistributionModifier
from llments.lm.lm import LanguageModel


class RetrievalAugmentedGenerationModifier(DistributionModifier):
    def __call__(self, base: LanguageModel, **kwargs) -> LanguageModel:
        """Apply retrieval-augmented generation over a dataset to enhance the model's generation.

        Args:
            base (LanguageModel): The language model to be modified.

        Returns:
            LanguageModel: The enhanced language model.
        """
        # RAG implementation
        raise NotImplementedError("This is not implemented yet.")
