from llments.lm.lm import LanguageModel
from llments.lm.base.hugging_face import HuggingFaceLM


class LMFitter:
    @classmethod
    def fit(cls, base, target: LanguageModel, **kwargs):
        """Fit a language model to match another language model.

        Args:
            base: The language model to be modified.
            target: The targetting language model to fit on.

        Returns:
            LanguageModel: The fitted language model.
        """
        if isinstance(base, HuggingFaceLM):
            return HuggingFaceLMFitter.fit(base, target, **kwargs)
        else:
            raise NotImplementedError(
                f"Cannot fit language models of type {type(base)}"
            )


class HuggingFaceLMFitter(LMFitter):
    @classmethod
    def fit(cls, base, target, **kwargs) -> LanguageModel:
        """Fit the language model to a target language model's distribution.

        Args:
            base: The HF language model to fine-tune. (delete the type identifier to pass mypy type checker)
            target: The language model that should be fitted to.
            batch_size: Number of examples processed in one step.
            training_steps: Number of steps to train.

        Returns:
            The fitted language model.
        """
        try:
            from transformers import TrainingArguments, Trainer
        except ImportError:
            raise ImportError(
                "You need to install 'transformers' package to use this function."
            )

        batch_size = kwargs.get("batch_size", 32)
        training_steps = kwargs.get("training_steps", 200)

        # Generate data and prepare training dataset
        inputs, labels = cls._prepare_training_data(
            base, target, batch_size, training_steps
        )
        dataset = cls._prepare_training_dataset(inputs, labels)

        num_train_epochs = training_steps / (len(dataset) / batch_size)

        training_args = TrainingArguments(
            output_dir="./training_results",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            logging_dir="./logs",
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
        base,
        target: LanguageModel,
        batch_size: int,
        training_steps: int,
    ):
        """Generate data from the target language model, using generate() function.

        Helper function of fit().
        Args:
            target: target language model.
            batch: Number of examples processed in one step.
            steps: Number of steps to train.
        Returns:
            inputs: Generated data (type: HF BatchEncoding): result from calling HF tokenizer.
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

    @classmethod
    def _prepare_training_dataset(cls, inputs, labels):
        """Return customized Dataset object, to be used in HF Trainer class.

        Helper function of fit()
        Args:
            inputs: generate inputs
            labels: labels from generate inputs
        Returns:
            Dataset object
        """
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError:
            raise ImportError(
                "You need both 'torch' and 'torch.utils.data' packages to use this function."
            )

        class TrainingDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {
                    key: torch.tensor(val[idx]) for key, val in self.encodings.items()
                }
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return TrainingDataset(inputs["input_ids"], labels)
