from llments.eval.eval import Evaluator, EvalContext

class RAGEvalContext(EvalContext):
    """A context for evaluating a hypothesized string."""
    def __init__(self, data: list[str]):
        """
        Initialize the EvalContext.

        Args:
            data (list[str]): List of strings for evaluation context.
        """
        self.data = data

class RAGEvaluator(Evaluator):
    """An evaluator to evaluate the sentiment of an output."""

    @staticmethod
    def normalize_answer(s: str) -> str:
        """Lower text and remove punctuation, articles, and extra whitespace."""
        import re
        import string

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @staticmethod
    def is_potential_number(word: str) -> bool:
        """
        Check if a word is a potential part of a number in textual form.

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word is a potential number part, False otherwise.
        """
        number_parts = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", 
                        "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", 
                        "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion"]
        return word.lower() in number_parts
    
    @staticmethod
    def _metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
        """
        Compute the maximum score over multiple ground truths.

        Args:
            metric_fn: The metric function to apply.
            prediction (str): The predicted answer.
            ground_truths (list[str]): List of ground truth answers.

        Returns:
            float: The maximum score over ground truths.
        """
        scores_for_ground_truths = []
        
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)

        return max(scores_for_ground_truths)
    
    @staticmethod
    def convert_textual_numbers_to_numeric(sentence: str) -> str:
        """
        Convert textual numbers within a sentence to numeric form.

        Args:
            sentence (str): The input sentence.

        Returns:
            str: The sentence with numbers converted to numeric form.
        """
        try:
            from word2number import w2n
        except ImportError:
            raise ImportError(
                "RAGEvaluator requires the `word2number` library."
            )
        words = sentence.split()
        converted_words = []
        current_number_phrase = []

        for word in words:
            if RAGEvaluator.is_potential_number(word):
                current_number_phrase.append(word)
            else:
                if current_number_phrase:
                    # Convert the current number phrase to a number
                    number_string = " ".join(current_number_phrase)
                    try:
                        numeric_value = w2n.word_to_num(number_string)
                        converted_words.append(str(numeric_value))
                    except ValueError:
                        # If conversion fails, keep the original phrase
                        converted_words.extend(current_number_phrase)
                    current_number_phrase = []
                
                converted_words.append(word)

        # Handle any remaining number phrase at the end
        if current_number_phrase:
            try:
                number_string = " ".join(current_number_phrase)
                numeric_value = w2n.word_to_num(number_string)
                converted_words.append(str(numeric_value))
            except ValueError:
                converted_words.extend(current_number_phrase)

        return ' '.join(converted_words)

    # EM score definition
    def _exact_match_score(self, prediction: str, ground_truth: str) -> bool:
        """
        Compute exact match score.

        Args:
            prediction (str): The predicted answer.
            ground_truth (str): The ground truth answer.

        Returns:
            bool: True if prediction exactly matches the ground truth, False otherwise.
        """
        return RAGEvaluator.normalize_answer(prediction) == RAGEvaluator.normalize_answer(ground_truth)
    
    # F1 score definition
    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1 score.

        Args:
            prediction (str): The predicted answer.
            ground_truth (str): The ground truth answer.

        Returns:
            float: The F1 score.
        """
        from collections import Counter

        prediction_tokens = RAGEvaluator.normalize_answer(prediction).split()
        ground_truth_tokens = RAGEvaluator.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    # ROUGEL score definition
    @staticmethod
    def _rougel_score(prediction: str, ground_truth: str) -> float:
        """
        Compute ROUGEL score.

        Args:
            prediction (str): The predicted answer.
            ground_truth (str): The ground truth answer.

        Returns:
            float: The ROUGEL score.
        """
        try:
            from rouge import Rouge
        except ImportError:
            raise ImportError(
                "RAGEvaluator requires the `rouge` library."
            )
        rouge = Rouge()
        # no normalization
        try:
            scores = rouge.get_scores(prediction, ground_truth, avg=True)
        except ValueError:  # "Hypothesis is empty."
            return 0.0
        return scores["rouge-l"]["f"]

    def evaluate(self, hyp: str, context: EvalContext, metric: str) -> float:
        """
        Returns a sentiment score (usually between 0-1) conditioned on data.

        Args:
            hyp (str): The hypothesized string (e.g., a system output).
            context (EvalContext): Any additional context about the evaluation.
            metric (str): The metric to evaluate ("accuracy", "exact_match", "f1", "rougel").

        Returns:
            float: The evaluation score, usually between 0 and 1 inclusive.
        """
        # Convert textual numbers in the hypothesis and context to numeric form
        guess_answer = self.convert_textual_numbers_to_numeric(hyp)
        gold_candidate_answers = [self.convert_textual_numbers_to_numeric(ans) for ans in context.data]

        # Evaluate based on the specified metric
        if metric == "accuracy":
            local_accuracy = 0
            if guess_answer in gold_candidate_answers:
                local_accuracy = 1
            return local_accuracy

        if metric == "exact_match":
            local_em = self._metric_max_over_ground_truths(
                self._exact_match_score, guess_answer, gold_candidate_answers
            )
            return local_em

        if metric == "f1":
            local_f1 = self._metric_max_over_ground_truths(
                self._f1_score, guess_answer, gold_candidate_answers
            )
            return local_f1

        if metric == "rougel":
            local_rougel = self._metric_max_over_ground_truths(
                self._rougel_score, guess_answer, gold_candidate_answers
            )
            return local_rougel
        
    def evaluate_batch(
        self,
        hyps: list[str],
        contexts: list[EvalContext],
        metric: str,
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
            metric (str): The metric to evaluate ("accuracy", "exact_match", "f1", "rougel").

        Returns:
            A list of evaluation scores, usually between 0 and 1 inclusive.
        """
        if show_progress:
            import tqdm
            hyps = tqdm.tqdm(hyps, desc="Evaluating")
        if contexts is not None:
            if len(hyps) != len(contexts):
                raise ValueError(
                    "The number of contexts must match the number of hypotheses."
                )
            return [self.evaluate(hyp, context, metric) for hyp, context in zip(hyps, contexts)]
        else:
            return [self.evaluate(hyp) for hyp in hyps]
