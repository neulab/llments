import abc


class LanguageModel:

    @abc.abstractclassmethod
    def sample(self, condition: str | None) -> str:
        """Sample from the language model, possibly conditioned on a prefix."""
        ...

    @abc.abstractclassmethod
    def fit(self, target: "LanguageModel") -> "LanguageModel":
        """Fit the language model to a target language model's distribution."""
        ...
