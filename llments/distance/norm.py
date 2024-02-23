from llments.lm.lm import LanguageModel


class L1Distance(object):
    """L1 distance between two language models."""

    def distance(self, lm1: LanguageModel, lm2: LanguageModel) -> float:
        """Returns a distance between two language models."""
        raise NotImplementedError("This is not implemented yet.")
