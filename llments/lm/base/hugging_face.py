"""Module for HuggingFace language models."""

import json
from typing import Any

from llments.lm.lm import LanguageModel


class HuggingFaceLM(LanguageModel):
    """A language model that uses the HuggingFace library."""

    def __init__(
        self,
        model: str,
        device: str | None = None,
        token: str | None = None,
    ):
        """Initialize a HuggingFaceLM.

        Args:
            model: The name of the model.
            device: The device to run the model on.
            token: Auth token for certain models from HF
        """
        try:
            from transformers import TextGenerationPipeline, pipeline
        except ImportError:
            raise ImportError(
                "You need to install the `transformers` package to use this class."
            )
        self.text_generator: TextGenerationPipeline = pipeline(
            "text-generation", model=model, device=device, token=token
        )

    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """Generate an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
                If None, the output is not conditioned.
            do_sample: Whether to use sampling or greedy decoding.
            max_length: The maximum length of the output sequence,
                (defaults to model max).
            max_new_tokens: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            temperature: The value used to module the next token probabilities.
            num_return_sequences: The number of independently computed returned
                sequences for each element in the batch.

        Returns:
            str: A sampled output sequence from the language model.
        """
        results = self.text_generator(
            condition,
            do_sample=do_sample,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            clean_up_tokenization_spaces=True,
            truncation=max_length is not None,
        )
        return [res["generated_text"] for res in results]

    def chat_generate(
        self,
        messages: list[dict[str, str]],
        do_sample: bool = False,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
    ) -> list[list[dict[str, str]]]:
        """Generate an output given a chat context.

        Args:
            messages: A list of dictionaries, each representing a message in the chat context. Each dictionary should contain the following keys:
            - "role": The role of the entity sending the message. This can be "system", "user", etc.
            - "content": The actual content of the message. Example:
            [
                {
                    "role": "system",
                    "content": "You are a friendly chatbot",
                },
                {
                    "role": "user",
                    "content": "How many helicopters can a human eat in one sitting?"
                },
            ]
            do_sample: Whether to use sampling or greedy decoding.
            max_length: The maximum length of the output sequence,
                (defaults to model max).
            max_new_tokens: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            temperature: The value used to module the next token probabilities.
            num_return_sequences: The number of independently computed returned
                sequences for each element in the batch.

        Returns:
            list[list[dict[str, str]]]: list of chat contexts with the generated responses.
        """
        results = self.text_generator(
            messages,
            do_sample=do_sample,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            clean_up_tokenization_spaces=True,
            truncation=max_length is not None,
        )
        return [res["generated_text"] for res in results]

    def set_seed(self, seed: int) -> None:
        """Set the seed for the language model.

        Args:
            seed: The seed to set for the language model.
        """
        try:
            from transformers import set_seed
        except ImportError:
            raise ImportError(
                "You need to install the `transformers` package to use this class."
            )
        set_seed(seed)

    def calculate_probability(self, condition: str | None, output: str) -> float:
        """Calculate the probability of an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
            output: The output sequence for which the probability is calculated.

        Returns:
            float: The probability of output x given the language model.
        """
        raise NotImplementedError


class HuggingFaceLMFitter:
    """A class responsible for fitting one Hugging Face language model to match another.

    This class provides the interface for adapting a base language model to more
    closely resemble the target language model.
    """

    @classmethod
    def fit(
        cls,
        base: HuggingFaceLM,
        target: LanguageModel,
        batch_size: int = 32,
        training_steps: int = 200,
        output_dir: str = "./training_results",
        logging_dir: str = "./logs",
    ) -> LanguageModel:
        """Fit the language model to a target language model's distribution.

        Args:
            base: The HF language model to fine-tune.
            target: The language model that should be fitted to.
            batch_size: Batch size for training.
            training_steps: Number of training steps.
            output_dir: Directory to save training results.
            logging_dir: Directory to save logs.

        Returns:
            The fitted language model.
        """
        try:
            import torch
            from torch.utils.data import Dataset
            from transformers import Trainer, TrainingArguments
        except ImportError:
            raise ImportError(
                "You need to install 'transformers' and 'torch' packages to use this "
                "function."
            )

        # Generate data and prepare training dataset
        inputs, labels = cls._prepare_training_data(
            base, target, batch_size, training_steps
        )

        class TrainingDataset(Dataset):  # type: ignore
            def __init__(
                self, encodings: dict[str, torch.Tensor], labels: torch.Tensor
            ):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                item = {
                    key: torch.tensor(val[idx]) for key, val in self.encodings.items()
                }
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self) -> int:
                return len(self.labels)

        dataset = TrainingDataset(inputs["input_ids"], labels)

        num_train_epochs = training_steps / (len(dataset) / batch_size)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            logging_dir=logging_dir,
            logging_steps=10,
        )

        trainer = Trainer(
            model=base.text_generator.model,
            args=training_args,
            train_dataset=dataset,
        )

        trainer.train()

        return base

    @classmethod
    def _prepare_training_data(
        cls,
        base: HuggingFaceLM,
        target: LanguageModel,
        batch_size: int,
        training_steps: int,
    ) -> tuple[dict[str, Any], Any]:
        """Generate data from the target language model, using generate() function.

        Helper function of fit().

        Args:
            base: model to fit.
            target: target language model.
            batch_size: Number of examples processed in one step.
            training_steps: Number of steps to train.

        Returns:
            inputs: Generated data (type: HF BatchEncoding): result from calling HF
                tokenizer.
            labels: "Up shift" each token to create the labels.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "You need to install/import 'torch' package to use this function."
            )

        samples = target.generate(
            condition=None,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=batch_size * training_steps,
        )

        tokenizer = base.text_generator.tokenizer
        inputs = tokenizer(
            samples, padding=True, truncation=True, return_tensors="pt"
        )  # return pytorch tensor

        labels = inputs.input_ids[:, 1:].clone()
        labels = torch.nn.functional.pad(
            labels, (0, 1), value=-100
        )  # Pad with -100 on the right

        # Adjust input_ids by removing the last token to match labels' size
        inputs.input_ids = inputs.input_ids[:, :-1]

        return inputs, labels


def load_from_spec(spec_file: str) -> HuggingFaceLM:
    """Load a language model from a specification file.

    Args:
        spec_file: The path to the specification file.
            The file should specifies the model identifier "model" and any other
            relevant parameters such as "device".

    Returns:
        A HuggingFaceLM instance.
    """
    with open(spec_file, "r") as file:
        spec = json.load(file)

    model_name = spec.get("model")
    device = spec.get("device", None)

    return HuggingFaceLM(model=model_name, device=device)
