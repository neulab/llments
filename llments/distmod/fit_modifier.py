from .distmod import DistributionModifier
from llments.lm.lm import LanguageModel


class Fitter(DistributionModifier):
    def __call__(self, base: LanguageModel, **kwargs) -> LanguageModel:
        """Fit a language model to match another language model's probability distribution.

        Args:
            base (LanguageModel): The language model to be modified.

        Returns:
            LanguageModel: The fitted language model.
        """
        target = kwargs.get("target", None)
        task_description = kwargs.get("task_description", None)
        base.fit(target, task_description)
        return base
