import abc


class LanguageModel:
    @abc.abstractmethod
    def calculate_probability(self, output: str) -> float:
        """Calculate the probability of an output given the language model.

        Args:
            output: The output sequence for which the probability is calculated.

        Returns:
            float: The probability of output x given the language model.
        """
        ...

    @abc.abstractmethod
    def generate(self, condition: str | None) -> str:
        """Generate an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
                If None, the output is not conditioned.

        Returns:
            str: A sampled output sequence from the language model.
        """
        ...
