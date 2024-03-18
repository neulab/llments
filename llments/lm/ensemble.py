"""Module for ensembling language models together."""

from typing import Literal
from llments.lm.lm import LanguageModel


class EnsembleLanguageModel(LanguageModel):
    """Ensemble several language models together."""

    def __init__(
        self,
        bases: list[LanguageModel],
        coeffs: list[float],
        method: Literal["linear", "log_linear"] = "linear",
    ):
        """Ensemble several language models together.

        Args:
            bases: The base language models to be modified.
            coeffs: The coefficients to be used for each base language model.
            method: The method to be used for ensembling the language models.
                - "linear": The language models are ensembled linearly.
                - "log_linear": The language models are ensembled log-linearly.

        Returns:
            A new ensembled language model.
        """
        raise NotImplementedError("This is not implemented yet.")
