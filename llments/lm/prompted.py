from llments.lm.lm import LanguageModel


class PromptedLanguageModel(LanguageModel):
    def __init__(self, base: LanguageModel, prompt: str):
        """Modify the base model's distribution using a textual prompt.

        Args:
            base: The language model to be modified.
        """
        raise NotImplementedError("This is not implemented yet.")
