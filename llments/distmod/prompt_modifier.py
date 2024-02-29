from .distmod import DistributionModifier
from llments.lm.lm import LanguageModel


class PromptModifier(DistributionModifier):
    def __call__(self, base: LanguageModel, **kwargs) -> LanguageModel:
        """Modify the base model's distribution using a textual prompt.

        Args:
            base (LanguageModel): The language model to be modified.

        Returns:
            LanguageModel: The modified language model.
        """
        # prompt_text = kwargs.get("prompt_text", None)
        # Prompting implementation
        # Integrate prompt_text into base_model generation process
        raise NotImplementedError("This is not implemented yet.")
