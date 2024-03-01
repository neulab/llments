from llments.eval.eval import Evaluator
from llments.lm.lm import LanguageModel


class RLLanguageModel(LanguageModel):
    def __init__(self, base: LanguageModel, evaluator: Evaluator):
        """Apply reinforcement learning to modify a model based on the reward function.

        Args:
            base: The language model to be modified.
            evaluator: The evaluator to be used to calculate the reward.

        Returns:
            LanguageModel: The modified language model.
        """
        raise NotImplementedError("This is not implemented yet.")
