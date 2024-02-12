import abc
import dataclasses


class PairwiseEvaluator:
    """A class that defines an evaluation function, assessing a hypothesized string."""

    @abc.abstractmethod
    def evaluate(self, hyp: str, ref: str) -> float:
        """Returns an evaluation score between 0 and 1 for two strings.

        Args:
            hyp: The hypothesized string (e.g. a system output).
            ref: The reference string (e.g. a gold-standard output).
        """
        ...


@dataclasses.dataclass
class EvaluatorMetadata:
    ...


class GeneralEvaluator:
    """A class that defines an evaluation function, assessing a hypothesized string."""

    @abc.abstractmethod
    def evaluate(self, hyp: str, ref: EvaluatorMetadata) -> float:
        """Returns an evaluation score between 0 and 1 for two strings.

        Args:
            hyp: The hypothesized string (e.g. a system output).
            ref: The reference string (e.g. a gold-standard output).
        """
        ...
