import abc
import warnings

import tqdm
from llments.eval.eval import Evaluator, EvalContext


class SentimentEvaluator(Evaluator):
    """An evaluator that evaluates the sentiment of an output."""

    @abc.abstractmethod
    def evaluate(self, hyp: str, context: EvalContext | None = None) -> float:
        """Returns a sentiment score (usually between 0-1) conditioned on data.

        Args:
            hyp: The hypothesized string (e.g. a system output).
            context: Any additional context about the evaluation.

        Returns:
            The evaluation score, usually between 0 and 1 inclusive.
        """
        ...


class HuggingFaceSentimentEvaluator(SentimentEvaluator):
    """An evaluator that uses HuggingFace to evaluate the sentiment of an output."""

    def __init__(self, model: str | None = None, device: str | None = None):
        """Initialize a HuggingFaceSentimentEvaluator.

        Args:
            model: The name of the model.
            device: The device to run the model on.
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "HuggingFaceSentimentEvaluator requires the `transformers` library."
            )
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=model,
            device=device,
        )
        self.sentiment_dict = {"negative": 0, "positive": 1, "neutral": 0.5}

    def evaluate(self, hyp: str, context: EvalContext | None = None) -> float:
        """Returns a sentiment score (usually between 0-1) conditioned on data.

        Args:
            hyp: The hypothesized string (e.g. a system output).
            context: Not used.

        Returns:
            The evaluation score, usually between 0 and 1 inclusive.
        """
        if context is not None:
            warnings.warn(
                "HuggingFaceSentimentEvaluator does not use the context argument.",
            )
        pred = self.sentiment_pipeline(hyp)
        return self.sentiment_dict[pred["label"].lower()]

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
            context: Not used.
            show_progress: Whether to show a progress bar.

        Returns:
            A list of evaluation scores, usually between 0 and 1 inclusive.
        """
        if contexts is not None:
            warnings.warn(
                "HuggingFaceSentimentEvaluator does not use the context argument.",
            )
        # TODO: we could have more intelligent guessing here
        if minibatch_size is None:
            minibatch_size = 128
        minibatch = []
        all_scores = []
        starts = range(0, len(hyps), minibatch_size)
        if show_progress:
            starts = tqdm.tqdm(
                starts, desc=f"Analyzing batches of size {minibatch_size}"
            )
        for i in starts:
            minibatch = hyps[i : i + minibatch_size]
            minibatch_scores = [
                self.sentiment_dict[x["label"]]
                for x in self.sentiment_pipeline(minibatch)
            ]
            all_scores.extend(minibatch_scores)
        return all_scores
