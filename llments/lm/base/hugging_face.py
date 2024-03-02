from llments.lm.lm import LanguageModel


class HuggingFaceLM(LanguageModel):
    def sample(
        self,
        condition: str | None,
        **kwargs,
    ) -> str:
        """Generate from the language model, possibly conditioned on a prefix."""
        raise NotImplementedError("This is not implemented yet.")

    def fit(
        self, target: LanguageModel, task_description: str | None = None
    ) -> LanguageModel:
        """Fit the language model to a target language model's distribution.

        Args:
            target: The language model that should be fitted to.
            task_description: A task description that explains more about
              what the language model that should be fit is doing (a prompt).

        Returns:
            The fitted language model.
        """
        raise NotImplementedError("This is not implemented yet.")


def load_from_spec(spec_file: str) -> HuggingFaceLM:
    """Load a language model from a specification file.

    Args:
        spec_file: The path to the specification file.

    Returns:
        A language model.
    """
    raise NotImplementedError("This is not implemented yet.")
