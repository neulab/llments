from llments.lm.lm import LanguageModel


class HuggingFaceLM(LanguageModel):
    def __init__(
        self,
        model: str,
        device: str | None = None,
    ):
        """Initialize a HuggingFaceLM.

        Args:
            model: The name of the model.
            device: The device to run the model on.
        """
        try:
            from transformers import pipeline, TextGenerationPipeline
        except ImportError:
            raise ImportError(
                "You need to install the `transformers` package to use this class."
            )
        self.text_generator: TextGenerationPipeline = pipeline(
            "text-generation", model=model, device=device
        )
        self.model_name = model 
        self.device = device 

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
        inputs, labels = self._prepare_training_data(target)
        dataset = GeneratedDataset(inputs, labels)

        # TODO: use HF Trainer class to train the model
        

    def _prepare_training_data(self, target: LanguageModel):
        """Generate data from the target language model, using generate() function.
        
        Helper function of fit().
        Args:
            target: target language model.
        Returns:
            inputs: Generated data (type: HF BatchEncoding): result from calling HF tokenizer.
            labels: "Up shift" each token to create the labels.
        """
        # Generate samples from the target model, consider this as one batch.
        samples = target.generate(condition=None, do_sample=True, max_length=50, temperature=1.0, num_return_sequences=1000)
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("You need to install the `transformers` package to use this method.")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = tokenizer(samples, padding=True, truncation=True, return_tensors="pt") # return pytorch tensor

        # Prepare labels by shifting
        labels = inputs.input_ids[:, 1:].clone()
        try:
            import torch
        except:
            raise ImportError("You need to install/import 'torch' package to use this function.")
        labels = torch.nn.functional.pad(labels, (0, 1), value=-100)  # Pad with -100 on the right

        # Adjust input_ids by removing the last token to match labels' size
        inputs.input_ids = inputs.input_ids[:, :-1]
        return inputs, labels
    
    def _prepare_training_dataset(self, inputs, labels):
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
        except:
            raise ImportError("You need both 'torch' and 'torch.utils.data' packages to use this function.")
        return GeneratedDataset(inputs, labels)
        

    def generate(
        self,
        condition: str | None,
        do_sample: bool = False,
        max_length: int | None = None,
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
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            clean_up_tokenization_spaces=True,
            truncation=max_length is not None,
        )
        return [res["generated_text"] for res in results]

    def set_seed(self, seed: int):
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

import Dataset
class GeneratedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_from_spec(spec_file: str) -> HuggingFaceLM:
    """Load a language model from a specification file.

    Args:
        spec_file: The path to the specification file.
        The file should specifies the model identifier "model" and any other relevant parameters such as "device".

    Returns:
        A HuggingFaceLM instance.
    """
    try:
        import json
    except ImportError:
        raise ImportError("You need to import/install json to use this function.")
    with open(spec_file, 'r') as file:
        spec = json.load(file)

    model_name = spec.get('model')
    device = spec.get('device', None)

    return HuggingFaceLM(model=model_name, device=device)
    
