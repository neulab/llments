from llments.lm.lm import LanguageModel


class FitLanguageModel(LanguageModel):
    def __init__(self, base: LanguageModel):
        """Fit a language model to match another language model.

        Args:
            base: The language model to be modified.

        Returns:
            LanguageModel: The fitted language model.
        """
        raise NotImplementedError("This is not implemented yet.")
