"""Module for HuggingFace language models."""

import json
from typing import Any

from llments.lm.lm import LanguageModel


class HuggingFaceLM(LanguageModel):
    """A language model that uses the HuggingFace library."""

    def __init__(
        self,
        model: str,
        tokenizer_path: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
        token: str | None = None,
    ):
        """Initialize a HuggingFaceLM.

        Args:
            model: The name of the model.
            tokenizer_path: path to find tokenizer, used with creating the model from a checkpoint.
            device: The device to run the model on.
            cache_dir: Path to a directory in which a downloaded pretrained model configuration should be cached
                        if the standard cache should not be used.
            token: Auth token for certain models from HF
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "You need to install the `transformers` package to use this class."
            )
        if not ".ckpt" in model:  # use the same tokenizer as the model
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model, do_sample=True, use_cache=True, cache_dir=cache_dir
            )
            self.device = device or "cpu"
            self.model.to(self.device)
        elif ".ckpt" in model and tokenizer_path is not None:  # load from checkpoint
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer = tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                model, from_tf=bool(".ckpt" in model)
            )
        else:
            raise ValueError(
                "You must create model from one of the following ways: \n"
                + "1. Input HF model name.\n"
                + "2. Load model from a checkpoint file, include tokenizer path as well."
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
        inputs = self.tokenizer(
            condition, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            clean_up_tokenization_spaces=True,
            truncation=max_length is not None,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )
        return [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]  # decode output tokens to strings

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
        chat_context = ""
        for message in messages:
            chat_context += f'{message["role"]}: {message["content"]}\n'

        inputs = self.tokenizer.encode(
            chat_context, return_tensors="pt", truncation=True, max_length=max_length
        )
        inputs = inputs.to(self.device)

        generated_tokens = self.model.generate(
            input_ids=inputs,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
        )

        return [
            self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_tokens
        ]

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
        batch_size: int = 8,  # batch size per device
        training_steps: int = 200,  # ie. max_steps
        training_epochs: float = 3.0,
        output_dir: str = "./training_results",  # ie. checkpoint_dir
        logging_dir: str = "./logs",
        do_train: bool = False,
        do_eval: bool = False,
        learning_rate: float = 5e-05,
        warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        evalution_strategy: str = "no",
        eval_steps: int = 500,
        prediction_loss_only: bool = False,
        optim: str = "adamw_torch",
        logging_steps: int = 500,
    ) -> LanguageModel:
        """Fit the language model to a target language model's distribution.

        Args:
            base: The HF language model to fine-tune.
            target: The language model that should be fitted to.
            batch_size: The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training and evaluation.
            training_steps: Number of training steps.
            training_epochs: Number of iterations to go through the entire dataset.
            output_dir: Directory to save training results.
            logging_dir: Directory to save logs.
            do_train: Whether to run training or not.
            do_eval: Whether to run evaluation on the validation set or not.
                        Will be set to True if evaluation_strategy is different from "no".
            learning_rate: The initial learning rate for AdamW optimizer.
            warmup_steps: Number of steps used for a linear warmup from 0 to learning_rate.
            max_grad_norm: Maximum gradient norm (for gradient clipping).
            evalution_strategy: The evaluation strategy to adopt during training.
            eval_steps: Number of update steps between two evaluations if evaluation_strategy="steps".
            prediction_loss_only: When performing evaluation and generating predictions, only returns the loss.
            optim: The optimizer to use. Can only choose from a list of names.
            logging_steps: Number of update steps between two logs if logging_strategy="steps".

        Returns:
            The fitted language model.
        """
        try:
            from transformers import (
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
            )
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

        # train_epochs = training_steps / (len(dataset) / batch_size)

        if training_epochs != 3.0:  # train using (training_steps, batch_size)
            training_args = TrainingArguments(
                output_dir=output_dir,
                do_train=do_train,
                do_eval=do_eval,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                max_grad_norm=max_grad_norm,
                max_steps=training_steps,  # use steps here
                optim=optim,
                evaluation_strategy=evalution_strategy,
                eval_steps=eval_steps,
                prediction_loss_only=prediction_loss_only,
                logging_dir=logging_dir,
                logging_steps=logging_steps,
            )
        else:  # train using (epochs, batch_size)
            training_args = TrainingArguments(
                output_dir=output_dir,
                do_train=do_train,
                do_eval=do_eval,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                max_grad_norm=max_grad_norm,
                num_train_epochs=training_epochs,  # use epochs here
                optim=optim,
                evaluation_strategy=evalution_strategy,
                eval_steps=eval_steps,
                prediction_loss_only=prediction_loss_only,
                logging_dir=logging_dir,
                logging_steps=logging_steps,
            )

        trainer = Trainer(
            model=base.model,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=base.tokenizer, mlm=False
            ),
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

        tokenizer = base.tokenizer
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


def load_from_checkpoint(model_path: str, tokenizer_path: str) -> HuggingFaceLM:
    """Load a language model from a checkpoint file.

    Args:
        model_path: model checkpoint path, has the suffix ".ckpt"
        tokenizer_path: the path to find tokenizer.

    Returns:
        A HuggingFaceLM instance.
    """
    return HuggingFaceLM(model=model_path, tokenizer_path=tokenizer_path)
