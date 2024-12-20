"""Module for HuggingFace language models."""

import json
import os
from typing import Any, Callable, List, Optional
import torch

from llments.lm.lm import LanguageModel


class HuggingFaceLM(LanguageModel):
    """A language model that uses the HuggingFace library."""

    def __init__(
        self,
        model: str,
        tokenizer_path: str | None = None,
        device: str | None = None,
        cache_dir: str | None = None,
    ):
        """Initialize a HuggingFaceLM.

        Args:
            model: The name of the model.
            tokenizer_path: path to find tokenizer, used with creating the model from a checkpoint.
            device: The device to run the model on.
            cache_dir: Path to a directory in which a downloaded pretrained model configuration should be cached
                        if the standard cache should not be used.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "You need to install the `transformers` package to use this class."
            )

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            do_sample=True,
            use_cache=True,
            cache_dir=cache_dir,
            from_tf=bool(".ckpt" in model),
        )

        # load tokenizer
        if tokenizer_path is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except:
                raise ValueError(
                    "You must create model from one of the following ways: \n"
                    + "1. Input a pretrained HF model name, and optionally the compatible tokenizer path. \n"
                    + "2. Load model from a checkpoint file, include tokenizer path as well. \n"
                )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True
            )

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # set device
        self.device = device or "cpu"
        self.model.to(self.device)

    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]]
        | None = None,
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
            prefix_allowed_tokens_fn: this function constraints the beam search to allowed tokens only at each step.
                This function takes 2 arguments: the batch ID and input_ids and returns a list with the allowed tokens for the next generation.

        Returns:
            str: A sampled output sequence from the language model.
        """
        inputs = self.tokenizer(
            condition,
            return_tensors="pt",
            truncation=max_length is not None,
            max_length=max_length,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
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
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=max_length is not None,
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

        prompt_length = len(self.tokenizer.decode(inputs[0], skip_special_tokens=True))
        respones = [
            self.tokenizer.decode(g, skip_special_tokens=True)[prompt_length:]
            for g in generated_tokens
        ]
        return [messages + [{"role": "assistant", "content": r}] for r in respones]

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
        raise NotImplementedError("This is not implemented yet.")

    def calculate_log_probability(self, condition: str | None, output: str) -> float:
        """Calculate the log probability of an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
            output: The output sequence for which the probability is calculated.

        Returns:
            float: The probability of output x given the language model.
        """
        import numpy as np

        if condition:
            full_input = condition + output
        else:
            full_input = output

        # Tokenize the full input (condition + output or just output)
        inputs = self.tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            padding=False,  # Avoid padding unless needed
        )

        # Get model outputs (logits)
        full_outputs = self.model(**inputs, return_dict=True)
        logits = (
            full_outputs.logits.detach().cpu().numpy()
        )  # Convert logits to NumPy array
        full_input_ids = inputs["input_ids"][0].cpu().numpy()

        # define a softmax function
        def softmax(logits: np.ndarray) -> np.ndarray:
            exps = np.exp(
                logits - np.max(logits, axis=-1, keepdims=True)
            )  # Stabilize softmax
            return exps / np.sum(exps, axis=-1, keepdims=True)

        # Calculate the probability of the output
        probs = softmax(logits[0])  # Only one sequence in the batch
        probs = probs[:-1, :]

        # calculate the num of tokens corresponding to the output
        output_ids = self.tokenizer(output)["input_ids"]
        output_ids = output_ids[1:]
        full_input_ids = full_input_ids[1:]
        start_idx = len(full_input_ids) - len(output_ids)

        # take the last # of output_tokens from the log_probs
        log_probs = np.log(probs[np.arange(start_idx, len(full_input_ids)), output_ids])

        # convert the log_probs to a float
        return float(np.sum(log_probs))

    def calculate_perplexity(self, condition: str | None, output: str) -> float:
        """Calculate the perplexity of an output given the language model.

        Args:
            condition: The conditioning sequence for the output.
            output: The output sequence for which the probability is calculated.

        Returns:
            float: The perplexity of output x given the language model.
        """
        import numpy as np

        log_prob = self.calculate_log_probability(condition, output)
        num_tokens = len(self.tokenizer(output)["input_ids"]) - 1

        return float(np.exp(-log_prob / num_tokens))

    def calculate_perplexity_batch(
        self, condition: list[str] | None, outputs: list[str]
    ) -> float:
        """Calculate the perplexity of multiple outputs given the language model.

        Args:
            condition: The conditioning sequence for the output.
            outputs: The output sequences for which the probability is calculated.

        Returns:
            list[float]: The perplexity of outputs given the language model.
        """
        if condition:
            full_inputs = [c + o for c, o in zip(condition, outputs)]
        else:
            full_inputs = outputs

        # check if the user have import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        try:
            from datasets import Dataset
            from transformers import (
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )
            import numpy as np
        except ImportError:
            print(
                "Naive implementation is used. This may harm the efficiency of the calculation."
            )
            try:
                import numpy as np
            except ImportError:
                raise ImportError(
                    "You need to install 'numpy' package to use this function."
                )
            return float(
                np.mean(self.calculate_perplexity(None, o) for o in full_inputs)
            )

        # prepare the dataset
        inputs = self.tokenizer(
            full_inputs,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        dataset = Dataset.from_dict(inputs)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_arguments = TrainingArguments(
            output_dir="trash",
            per_device_eval_batch_size=1,
            do_train=False,
            do_eval=True,
            fp16=False,
            prediction_loss_only=True,
        )

        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=training_arguments,
            eval_dataset=dataset,
        )

        eval_result = trainer.evaluate()

        return float(np.exp(eval_result["eval_loss"]))


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
        eval_target: LanguageModel | None = None,
        batch_size: int = 8,  # batch size per device
        training_steps: int = 200,
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
        save_steps: int = 500,
        lora_r: int | None = None,
        lora_alpha: int | None = None,
    ) -> LanguageModel:
        """Fit the language model to a target language model's distribution.

        Args:
            base: The HF language model to fine-tune.
            target: The language model that should be fitted to.
            eval_target: The language model used to evaluate the training process.
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
            save_steps: Number of updates steps between two checkpoints.
            lora_r: Lora attention dimension (the “rank”).
            lora_alpha: The alpha parameter for Lora scaling.

        Returns:
            The fitted language model.
        """
        try:
            from datasets import Dataset
            from transformers import (
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )
        except ImportError:
            raise ImportError(
                "You need to install 'transformers' and 'torch' packages to use this "
                "function."
            )

        # Generate data and prepare training dataset
        samples = target.generate(
            condition=None,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=batch_size * training_steps,
        )

        if not base.tokenizer.pad_token:
            base.tokenizer.pad_token = base.tokenizer.eos_token
        inputs = base.tokenizer(
            samples, padding=True, truncation=True, return_tensors="pt"
        )

        # convert tokenized text into a Dataset object
        dataset = Dataset.from_dict(inputs)
        print("Dataset LM for training prepared!")

        if eval_target:
            eval_samples = eval_target.generate(
                condition=None,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=batch_size * training_steps,
            )

            eval_inputs = base.tokenizer(
                eval_samples, padding=True, truncation=True, return_tensors="pt"
            )
            eval_dataset = Dataset.from_dict(eval_inputs)
            print("Dataset LM for evaluation prepared!")

        # wrap the base model with peft
        if lora_r and lora_alpha:
            print("Using LORA attention for fitting.")
            try:
                from peft import (
                    LoraConfig,
                    get_peft_model,
                    prepare_model_for_kbit_training,
                )
            except ImportError:
                raise ImportError(
                    "You need to install 'peft' package to use this LORA functionality."
                )
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                # trainable layers: all linear layers between multihead attention
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = prepare_model_for_kbit_training(base.model)
            base.model = get_peft_model(model, lora_config)

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
            save_steps=save_steps,
        )

        # Make output_dir and logging_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)

        print("Start fitting...")
        if not do_eval:
            trainer = Trainer(
                model=base.model,
                args=training_args,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=base.tokenizer, mlm=False
                ),
                train_dataset=dataset,
            )
        else:
            trainer = Trainer(
                model=base.model,
                args=training_args,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=base.tokenizer, mlm=False
                ),
                train_dataset=dataset,
                eval_dataset=eval_dataset,
            )

        trainer.train()
        base.tokenizer.save_pretrained(output_dir)
        trainer.save_model(output_dir)
        print("fitted modes saved to", output_dir)

        return base


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
