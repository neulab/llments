import abc

from llments.lm.lm import LanguageModel


class Distance:
    @abc.abstractmethod
    def distance(self, lm1: LanguageModel, lm2: LanguageModel) -> float:
        """Returns a distance between two language models."""
        ...
