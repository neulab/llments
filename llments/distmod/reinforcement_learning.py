from .distmod import DistributionModifier
from llments.lm.lm import LanguageModel


class ReinforcementLearningModifier(DistributionModifier):
    def __call__(self, base: LanguageModel, **kwargs) -> LanguageModel:
        """Apply reinforcement learning to modify a model based on the provided reward function.

        Args:
            base (LanguageModel): The language model to be modified.

        Returns:
            LanguageModel: The modified language model.
        """
        # reward_function = kwargs.get("reward_function", None)
        # Reinforcement Learning implementation
        raise NotImplementedError("This is not implemented yet.")
