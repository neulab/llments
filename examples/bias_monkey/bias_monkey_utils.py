"""This script uses LLMs to generate responses to survey questions."""

import os
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from tqdm import tqdm

from llments.lm.lm import LanguageModel


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


def reverse_label(num_options: int, responses: str) -> str:
    """Reverse the labels of the responses."""
    response_list = list(responses.split(","))
    alpha_labels = list(string.ascii_lowercase[:num_options])
    reverse_labels = alpha_labels[::-1]
    label_map = dict(zip(alpha_labels, reverse_labels))
    reversed_labels = [label_map[char.lower()] for char in response_list]
    return ",".join(reversed_labels)


def shift_label(num_options: int, responses: str) -> str:
    """Shift the labels of the responses."""
    if num_options % 2 == 1:
        return responses
    # if even, shift responses by 1 after midpoint
    # e.g. a,b,c,d -> a,b,d,e
    response_list = list(responses.split(","))
    midpoint = num_options // 2
    alpha_labels = list(string.ascii_lowercase[:num_options])
    new_alpha_labels = list(string.ascii_lowercase[: num_options + 1])
    shifted_labels = []
    for char in response_list:
        index = alpha_labels.index(char)
        if index >= midpoint:
            index += 1
            char = new_alpha_labels[index]
        shifted_labels.append(char)
    return ",".join(shifted_labels)


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


def format_df(
    all_data_df: pd.DataFrame,
    bias_type: str,
    perturbation: str | None = None,
    num_responses: int = 50,
) -> pd.DataFrame:
    """Format the dataframe for the bias type."""
    if bias_type == "acquiescence_reword":
        bias_type = "acquiescence"
    if perturbation not in ["key_typo", "middle_random", "letter_swap"]:
        perturbation = None

    # bias specific reformatting
    # does not apply to perturbations
    if bias_type == "response_order" and perturbation is None:
        all_data_df["responses"] = all_data_df.apply(
            lambda row: reverse_label(row.num_options, row.responses)
            if row.type != "orig alpha"
            else row.responses,
            axis=1,
        )
    if bias_type == "odd_even" and perturbation is None:
        all_data_df["responses"] = all_data_df.apply(
            lambda row: shift_label(row.num_options, row.responses), axis=1
        )

    keys = all_data_df["key"].unique()
    cols = get_col_names(bias_type)
    groups = cols

    all_responses = []
    key_col = []
    groups_col = []
    num_options_col = []
    df = pd.DataFrame(columns=["key", "response", "group", "num_options"])
    for key in keys:
        q_variations_df = all_data_df[all_data_df["key"] == key]
        for col, group in zip(cols, groups):
            q_row = q_variations_df[q_variations_df["type"] == col].squeeze()
            try:
                responses = list(q_row.responses.split(","))
                num_model_responses = len(responses)
                if num_model_responses < num_responses:
                    num_responses = num_model_responses
            except:
                print(q_row)
            key_col += [key] * num_responses
            groups_col += [group] * num_responses
            num_options_col += [q_row.num_options] * num_responses
            # if there are > 50 responses, only take the first 50
            all_responses += responses[:num_responses]

    upper_responses = all_responses.copy()
    all_responses = []
    for response in upper_responses:
        all_responses.append(response.lower().strip())

    df["key"] = pd.Series(key_col)
    df["response"] = pd.Series(all_responses)
    df["group"] = pd.Series(groups_col)
    df["num_options"] = pd.Series(num_options_col)
    return df


def generate_survey_responses(
    model: LanguageModel,
    prompts_file: str,
    bias_type: str,
    output_path: str,
    is_chat_model: bool = True,
    seed: int | None = None,
    num_samples: int = 50,
    batch_size: int = 10,
    max_attempts: int | None = None,
    overwrite: bool = False,
    prompt_template: str = "Please answer the following question with one of the alphabetical options provided.\nQuestion: ",
) -> pd.DataFrame:
    """Generate responses to survey questions in prompts_file.

    Args:
        model: The language model.
        prompts_file: The csv file containing the prompts.
        bias_type: one of ["acquiescence", "allow_forbid", "odd_even", "response_order", "question_order"]
        output_path: output path (pickle file).
        is_chat_model: Whether the model is a chat model.
        seed: The seed for the language model.
        num_samples: The number of valid (i.e. one of the letter options) responses to sample.
        batch_size: batch size for generation.
        max_attempts: The maximum number of attempts to generate valid responses.
        overwrite: Whether to overwrite the output file if it exists.
        prompt_template: The template for the prompt.
    """
    if seed is not None:
        model.set_seed(seed)

    if os.path.exists(output_path) and not overwrite:
        print(f"Output file {output_path} already exists. Skipping.")
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    prompts = pd.read_csv(prompts_file)
    orig_col_name, modified_col_name = get_col_names(bias_type)
    results = []
    for index, row in tqdm(
        prompts.iterrows(), total=prompts.shape[0], desc="Generating responses"
    ):
        for prompt_col_name in [orig_col_name, modified_col_name]:
            question = row[prompt_col_name]
            num_options = (
                row["num options new"]
                if prompt_col_name == modified_col_name
                and "num options new" in prompts.columns
                else row["num options"]
            )
            print(f"prompt_col_name {prompt_col_name}, num_options {num_options}")
            print(f"question {question}")
            all_answers: list[str] = []
            max_attempts = max_attempts or num_samples * 10
            num_attempts = 0
            while len(all_answers) < num_samples and num_attempts < max_attempts:
                if is_chat_model:
                    chat_prompt = [
                        {
                            "role": "system",
                            "content": "Respond only with the letter corresponding to your answer.",
                        },
                        {
                            "role": "user",
                            "content": prompt_template + question,
                        },
                    ]
                    chat_responses = model.chat_generate(
                        chat_prompt,
                        do_sample=True,
                        max_new_tokens=2,
                        temperature=1.0,
                        num_return_sequences=batch_size,
                    )
                    answers = [r[-1].get("content", "") for r in chat_responses]
                else:
                    prompt = prompt_template + question + "\nAnswer: "
                    responses = model.generate(
                        prompt,
                        do_sample=True,
                        max_new_tokens=2,
                        temperature=1.0,
                        num_return_sequences=batch_size,
                    )
                    answers = [
                        r[len(prompt) :] if r.startswith(prompt) else r
                        for r in responses
                    ]
                num_attempts += len(answers)
                all_answers += [
                    a.strip().lower()
                    for a in answers
                    if is_valid_prediction(a, num_options)
                ]
            results.append(
                {
                    "key": row["key"],
                    "num_options": num_options,
                    "question": question,
                    "type": prompt_col_name,
                    "responses": ",".join(all_answers[:num_samples]),
                }
            )
        break  # TODO: delete this
    results_df = pd.DataFrame(results)
    results_df.to_pickle(output_path)
    return results_df
