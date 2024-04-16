"""A module defining the superclass for evaluators."""

import abc
import dataclasses
import tqdm


@dataclasses.dataclass
class EvalContext:
    """A context for evaluating a hypothesized string."""

    ...


class Evaluator:
    """A class that defines an evaluation function, assessing a hypothesized string."""

    @abc.abstractmethod
    def evaluate(self, hyp: str, context: EvalContext | None = None) -> float:
        """Returns an evaluation score (usually between 0-1) conditioned on data.

        Args:
            hyp: The hypothesized string (e.g. a system output).
            context: The reference context to condition on.

        Returns:
            The evaluation score, usually between 0 and 1 inclusive.
        """
        ...

    def evaluate_batch(
        self,
        hyps: list[str],
        contexts: list[EvalContext] | None = None,
        minibatch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[float]:
        """Evaluate many hypotheses at once.

        Args:
            hyps: A list of hypothesized strings (e.g. system outputs).
            contexts: The reference context to condition on.
            minibatch_size: The size of the minibatch to use,
                None guesses a good size automatically.
            show_progress: Whether to show a progress bar.

        Returns:
            A list of evaluation scores, usually between 0 and 1 inclusive.
        """
        if show_progress:
            hyps = tqdm.tqdm(hyps, desc="Evaluating")
        if contexts is not None:
            if len(hyps) != len(contexts):
                raise ValueError(
                    "The number of contexts must match the number of hypotheses."
                )
            return [self.evaluate(hyp, context) for hyp, context in zip(hyps, contexts)]
        else:
            return [self.evaluate(hyp) for hyp in hyps]
