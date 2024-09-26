"""Evaluators for toxicity analysis."""

import os
import requests  # type: ignore
import json
import time
import tqdm
from typing import Dict
from llments.eval.eval import Evaluator, EvalContext


class ToxicityEvaluator(Evaluator):
    """A class that defines an evaluation function for toxicity analysis."""

    def __init__(self, api_key: str):
        """Initializes the ToxicityEvaluator class.

        Args:
            api_key (str): API key for accessing the Perspective API.
        """
        self.api_key = api_key
        self.url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
        self.categories = [
            "toxicity",
            "severe_toxicity",
            "insult",
            "flirtation",
            "profanity",
            "sexually_explicit",
            "threat",
            "identity_attack",
        ]
        self.languages = ["en"]

    def _get_toxicity_scores(self, text: str) -> Dict[str, float]:
        """Helper method to get toxicity scores for a single text input from the API.

        Args:
            text (str): The text to evaluate.

        Returns:
            Dict[str, float]: A dictionary containing toxicity scores for various categories.
        """
        if len(text.strip()) == 0:
            # Return zeros for empty text
            return {category: 0.0 for category in self.categories}

        data_dict = {
            "comment": {"text": text},
            "languages": self.languages,
            "requestedAttributes": {cat.upper(): {} for cat in self.categories},
        }

        # Make the API request
        response = requests.post(url=self.url, data=json.dumps(data_dict))
        response_dict = json.loads(response.content)

        # Return scores or zeros if no valid response
        if "attributeScores" in response_dict:
            scores = {}
            for category in self.categories:
                score = response_dict["attributeScores"][category.upper()][
                    "spanScores"
                ][0]["score"]["value"]
                scores[category] = score
            return scores
        else:
            # Return zeros for invalid responses
            return {category: 0.0 for category in self.categories}

    # override the evaluate method from the Evaluator class
    # for better design logic, the evaluate() and evaluate_batch() methods
    # only support the "toxicity" context
    def evaluate(self, hyp: str, context: EvalContext | None = None) -> float:
        """Returns the toxicity score for a given hypothesis.

        Args:
            hyp (str): The hypothesized string (e.g. a system output).
            context (EvalContext | None): The reference context to condition on.

        Returns:
            float: The toxicity score, usually between 0 and 1 inclusive.
        """
        if context is not None:
            raise Warning(
                "This method only supports the 'toxicity' metric. \
                            Please use evaluate_multiple() for other metrics."
            )

        return self._get_toxicity_scores(hyp)["toxicity"]

    # override the evaluate_batch method from the Evaluator class
    def evaluate_batch(
        self,
        hyps: list[str],
        contexts: list[EvalContext] | None = None,
        minibatch_size: int | None = None,
        show_progress: bool = False,
    ) -> list[float]:
        """Evaluate the toxicity of many hypotheses at once.

        Args:
            hyps (list[str]): A list of hypothesized strings (e.g. system outputs).
            contexts (list[EvalContext] | None): The reference context to condition on.
            minibatch_size (int | None): The size of the minibatch to use,
                None guesses a good size automatically.
            show_progress (bool): Whether to show a progress bar.

        Returns:
            list[float]: A list of toxicity scores, usually between 0 and 1 inclusive.
        """
        if contexts is not None:
            # raise a warning that this method only supports the "toxicity" context
            raise Warning(
                "This method only supports the 'toxicity' metric. \
                            Please use evaluate_batch_multiple() for other metrics."
            )

        if show_progress:
            hyps = tqdm.tqdm(hyps, desc="Evaluating")

        res = []
        for hyp in hyps:
            # to avoid rate limiting
            time.sleep(1.2)
            res.append(self.evaluate(hyp, contexts))
        return res

    # override the evaluate_multiple method from the Evaluator class
    # set the default contexts to ["toxicity"]

    def evaluate_multiple(
        self,
        hyp: str,
        metrics: list[str] = ["toxicity"],
        show_progress: bool = False,
    ) -> list[float]:
        """Evaluate multiple metrics at once.

        Args:
            hyp (str): The hypothesized string.
            metrics (list[str]): A list of metrics to evaluate.
            show_progress (bool): Whether to show a progress bar.

        Returns:
            Dict[str, float]: A dictionary of evaluation scores, usually between 0 and 1 inclusive.
        """
        for metric in metrics:
            if metric not in self.categories:
                raise ValueError(f"Invalid metric: {metric}")

        # get the toxicity scores based on the contexts
        scores = self._get_toxicity_scores(hyp)

        # return the scores for the specified contexts
        return [scores[metric] for metric in metrics]

    def evaluate_batch_multiple(
        self,
        hyps: list[str],
        metrics: list[str] = ["toxicity"],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Evaluate multiple metrics for many hypotheses at once.

        Args:
            hyps (list[str]): A list of hypothesized strings.
            metrics (list[str]): A list of metrics to evaluate.
            show_progress (bool): Whether to show a progress bar.

        Returns:
            list[list[float]]: A list of lists of evaluation scores, usually between 0 and 1 inclusive.
        """
        if show_progress:
            hyps = tqdm.tqdm(hyps, desc="Evaluating")

        res = []
        for hyp in hyps:
            # to avoid rate limiting
            time.sleep(1.2)
            res.append(self.evaluate_multiple(hyp, metrics))

        return res
