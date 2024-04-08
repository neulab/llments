"""This script uses LLMs to generate responses to survey questions."""

import os
import tqdm
from pathlib import Path
import pandas as pd
import numpy as np

from llments.lm.lm import LanguageModel

import re
import string


def is_valid_prediction(answer: str, num_options: int) -> bool:
    """Check if the response is a valid prediction."""
    answer = answer.strip()
    valid_options = [*string.ascii_uppercase[:num_options]]
    # for word in re.split("[^a-zA-Z]", answer):
    #     if word in valid_options:
    #         return True
    if answer in valid_options:
        return True
    return False


def get_col_names(bias_type: str) -> list[str]:
    """Get the column names for the bias type."""
    bias_cols = {
        "acquiescence": ["orig alpha", "pos alpha"],
        "allow_forbid": ["orig alpha", "forbid alpha"],
        "odd_even": ["middle alpha", "no middle alpha"],
        "response_order": ["orig alpha", "reversed alpha"],
        "question_order": ["question 0", "question 1"],
        "opinion_float": ["orig alpha", "float alpha"],
    }
    col_names = bias_cols[bias_type]
    return col_names


def generate_survey_responses(
    model: LanguageModel,
    prompts_file: str,
    bias_type: str,
    output_path: str,
    is_chat_model: bool = True,
    seed: int | None = None,
    num_samples: int = 50,
    overwrite: bool = False,
    prompt_template: str = "Please answer the following question with one of the alphabetical options provided.\nQuestion: ",
) -> None:
    """Generate responses to survey questions in prompts_file.

    Args:
        model: The language model.
        prompts_file: The csv file containing the prompts.
        bias_type: one of ["acquiescence", "allow_forbid", "odd_even", "response_order", "question_order"]
        output_path: output path (pickle file).
        is_chat_model: Whether the model is a chat model.
        seed: The seed for the language model.
        num_samples: The number of valid (i.e. one of the letter options) responses to sample.
        overwrite: Whether to overwrite the output file if it exists.
        prompt_template: The template for the prompt.
    """
    print(
        f"generate_survey_responses: model={model}, prompts_file={prompts_file}, bias_type={bias_type}, output_path={output_path}, seed={seed}, num_samples={num_samples}, overwrite={overwrite}, prompt_template={prompt_template}"
    )

    if seed is not None:
        model.set_seed(seed)

    if os.path.exists(output_path) and not overwrite:
        print(f"Output file {output_path} already exists. Skipping.")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    prompts = pd.read_csv(prompts_file)
    orig_col_name, modified_col_name = get_col_names(bias_type)
    results = []
    for index, row in prompts.iterrows():
        for prompt_col_name in [orig_col_name, modified_col_name]:
            question = row[prompt_col_name]
            key, num_options = row["key"], row["num options"]
            prompt: str | list[dict[str, str]] = ""
            if is_chat_model:
                prompt = [
                    {
                        "role": "system",
                        "content": "Respond only with the letter corresponding to your answer.",
                    },
                    {
                        "role": "user",
                        "content": prompt_template + question,
                    },
                ]
            else:
                # TODO: few-shot prompting?
                prompt = prompt_template + question
            print(f"Generating responses for prompt3:\n{prompt}")
            all_answers: list[str] = []
            while len(all_answers) < num_samples:
                responses: list[str] | list[list[dict[str, str]]] = model.generate(
                    prompt,
                    do_sample=True,
                    max_new_tokens=2,
                    temperature=1.0,
                    num_return_sequences=10,
                )
                # answers = [r[-1].get("content", "") for r in responses]
                # if responses and isinstance(responses[0], list):
                #     answers = [r[-1].get("content", "") for r in responses]
                # else:
                #     # TODO: handle non-chat models
                #     answers = [
                #         r[len(prompt) :] for r in responses if r.startswith(prompt)
                #     ]
                for response in responses:
                    answer: str = ""
                    if isinstance(response, list):
                        answer = response[-1].get("content", "")
                    elif isinstance(response, str) and isinstance(prompt, str):
                        answer = (
                            response[len(prompt) :]
                            if response.startswith(prompt)
                            else response
                        )
                    if is_valid_prediction(answer, num_options):
                        all_answers.append(answer.strip().lower())
            results.append(
                {
                    "key": key,
                    "num_options": num_options,
                    "question": question,
                    "type": prompt_col_name,
                    "responses": ",".join(all_answers[:num_samples]),
                }
            )
        break  # TODO: delete me, for testing purposes, only generate responses for the first row
    results_df = pd.DataFrame(results)
    results_df.to_pickle(output_path)
