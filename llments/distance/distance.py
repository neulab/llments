"""A distance module for comparing language models."""

import abc

from llments.lm.lm import LanguageModel


class Distance:
    """A distance between two language models."""

    @abc.abstractmethod
    def distance(self, lm1: LanguageModel, lm2: LanguageModel) -> float:
        """Returns a distance between two language models."""
        ...
