import abc


class LanguageModel:
    @abc.abstractmethod
    def generate(
        self,
        condition: str | None,
        **kwargs,
    ) -> str:
        """Generate from the language model, possibly conditioned on a prefix."""
        ...

    @abc.abstractmethod
    def fit(
        self, target: "LanguageModel", task_description: str | None = None
    ) -> "LanguageModel":
        """Fit the language model to a target language model's distribution.

        Args:
            target: The language model that should be fitted to.
            task_description: A task description that explains more about
              what the language model that should be fit is doing (a prompt).

        Returns:
            The fitted language model.
        """
        ...

    @abc.abstractmethod
    def calculcate_probability(self, output: str) -> float:
        """Calculate the probability of an output given the language model.

        Args:
            output (str): The output sequence for which the probability is calculated.

        Returns:
            float: The probability of output x given the language model.
        """
        ...

    @abc.abstractmethod
    def sample(self) -> str:
        """Sample an output given the language model.

        Returns:
            str: A sampled output sequence from the language model.
        """
        ...
