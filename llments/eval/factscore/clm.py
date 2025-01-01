# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""CLM (Causal Language Model) Module."""
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional, Union, List, Tuple

from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

from llments.eval.factscore.utils import convert_model_to_int8_on_gpu
from llments.eval.factscore.lm import LM

class CLM(LM):
    """CLM (Causal Language Model) Class.

    This class extends the `LM` base class to provide functionalities specific to causal language modeling.
    It leverages pre-trained models from Hugging Face's Transformers library, enabling text generation
    based on input prompts. The class includes methods for loading models, generating text, and managing
    caches to optimize performance.

    Attributes:
        model_name (str): Name of the pre-trained language model.
        model_dir (str): Directory path where the pre-trained model is stored.
        model (Optional[AutoModelForCausalLM]): Loaded causal language model.
        tokenizer (LlamaTokenizer): Tokenizer corresponding to the pre-trained model.
    """
    def __init__(
        self,
        model_name: str,
        model_dir: str,
        cache_file: Optional[str] = None,
    ) -> None:
        """Initialize the CLM (Causal Language Model) instance.

        Args:
            model_name (str): Name of the pre-trained language model.
            model_dir (str): Directory path where the pre-trained model is stored.
            cache_file (Optional[str], optional): Path to the cache file for storing generated outputs.
                Defaults to None.
        """
        self.model_name = model_name
        self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self) -> None:
        """Load the pre-trained causal language model and its tokenizer.

        This method loads the model from the specified directory, converts it to int8 precision
        for efficient GPU utilization, and initializes the tokenizer.

        Raises:
            EnvironmentError: If the model or tokenizer cannot be loaded.
        """
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)
        self.model = convert_model_to_int8_on_gpu(self.model, device='cuda')
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_dir)

    def _generate(
        self,
        prompt: str,
        sample_idx: int = 0,
        max_sequence_length: int = 2048,
        max_output_length: int = 128,
        prompts: Union[str, List[str]] = None,
        end_if_newline: bool = False,
        end_if_second_newline: bool = False,
        verbose: bool = False,
    ) -> Union[Tuple[str, np.ndarray], Tuple[List[str], List[np.ndarray]]]:
        """Generate text based on input prompts using the causal language model.

        Args:
            prompt (str): The input prompt to generate text from.
            sample_idx (int, optional): Index to differentiate between samples. Defaults to 0.
            max_sequence_length (int, optional): Maximum length of the input sequence.
                Defaults to 2048.
            max_output_length (int, optional): Maximum length of the generated output.
                Defaults to 128.
            prompts (Union[str, List[str]]): Single prompt string or a list of prompt strings.
            end_if_newline (bool, optional): If True, truncate the generation at the first newline.
                Defaults to False.
            end_if_second_newline (bool, optional): If True, truncate the generation at the second newline.
                Defaults to False.
            verbose (bool, optional): If True, print detailed generation information.
                Defaults to False.

        Returns:
            Union[Tuple[str, np.ndarray], Tuple[List[str], List[np.ndarray]]]:
                - If a single prompt is provided, returns a tuple containing the generated text and its scores.
                - If multiple prompts are provided, returns a tuple of lists containing generated texts and their corresponding scores.

        Raises:
            AssertionError: If the lengths of generations, prompts, and scores do not match.
        """
        assert self.model is not None, "Model has not been loaded. Call load_model() before generating."
        
        is_single = type(prompts)==str
        if isinstance(prompts, str):
            prompts = [prompts]

        input_ids = self.tokenizer(prompts).input_ids
        if verbose:
            input_ids = tqdm(input_ids)

        generations: List[str] = []
        scores = []
        for curr_input_ids in input_ids:
            if len(curr_input_ids) > max_sequence_length - max_output_length:
                curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
            curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
            gen_outputs = self.model.generate(
                curr_input_ids,
                max_length=curr_input_ids.shape[1]+max_output_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            gen_tokens = gen_outputs["sequences"]
            # saving the logits for the very first token
            gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
            gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

            if end_if_newline:
                gen = gen.split("\n")[0].strip()
            elif end_if_second_newline:
                gen = "\n".join(gen.split("\n")[:2]).strip()

            if verbose and len(generations)==0:
                print ("Input:", prompts[0])
                print ("Prediction:", gen)

            if self.model_name.startswith("llama-sni"):
                gen = gen.split("</s>")[0]
                
            generations.append(gen)
            scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            return generations[0], scores[0]
        
        return generations, scores

