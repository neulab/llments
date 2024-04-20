"""This script uses LLMs to generate responses to survey questions."""

import csv
import os
import random
import re
import string
from collections import Counter
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import SubplotSpec
from scipy.stats import entropy, ttest_1samp, wasserstein_distance
from tqdm import tqdm
from tqdm.contrib import itertools as tqdm_itertools

from llments.lm.lm import LanguageModel

bias_types = [
    "acquiescence",
    "response_order",
    "odd_even",
    "opinion_float",
    "allow_forbid",
]
clean_bias_labels = [
    "Acquiescence",
    "Response Order",
    "Odd/even",
    "Opinion Float",
    "Allow/forbid",
]
perturbations = ["-key_typo", "-middle_random", "-letter_swap"]
exp_settings = ["modified", "key typo", "middle random", "letter swap"]
clean_labels = ["bias", "key typo", "middle random", "letter swap"]


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


def get_groups(bias_type: str) -> tuple[str, str, list[str], list[str]]:
    """Get the groups for the bias type."""
    if "acquiescence" in bias_type:
        first_group = "pos alpha"
        second_group = "orig alpha"
        first_options = ["a"]
        second_options = first_options
    elif "response_order" in bias_type:
        first_group = "orig alpha"
        second_group = "reversed alpha"
        first_options = ["a"]
        second_options = first_options
    elif "odd_even" in bias_type:
        first_group = "no middle alpha"
        second_group = "middle alpha"
        first_options = ["b", "d"]
        second_options = first_options
    elif "opinion_float" in bias_type:
        first_group = "orig alpha"
        second_group = "float alpha"
        first_options = ["c"]
        second_options = first_options
    elif "allow_forbid" in bias_type:
        first_group = "orig alpha"
        second_group = "forbid alpha"
        first_options = ["b"]

        if (
            "key_typo" in bias_type
            or "middle_random" in bias_type
            or "letter_swap" in bias_type
        ):
            second_options = ["b"]
        else:
            second_options = ["a"]
    else:
        raise ValueError(f"Invalid bias type: {bias_type}")

    assert len(first_options) == len(second_options)

    return first_group, second_group, first_options, second_options


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
        return pd.DataFrame()
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


def run_stat_test(bias_type: str, csv_file: str) -> tuple[list[float], Any, list[str]]:
    """Run a statistical test."""
    scores = {}

    exclude_list = [
        "GAP21Q46_W82",
        "RACESURV15b_W43",
        "DRONE4D_W27",
        "ABORTIONALLOW_W32",
        "INEQ10_W54",
        "INEQ11_W54",
        "POLICY1_W42",
        "GOVT_ROLE_W32",
    ]

    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        first_group, second_group, first_options, second_options = get_groups(bias_type)
        for row in reader:
            if ("allow_forbid" in bias_type and row["key"] not in exclude_list) or (
                "allow_forbid" not in bias_type
            ):
                if row["key"] not in scores:
                    scores[row["key"]] = 0

                if row["group"] == first_group and row["response"] in first_options:
                    scores[row["key"]] += 1
                if row["group"] == second_group and row["response"] in second_options:
                    scores[row["key"]] += -1

    values = [value / 50 * 100 for value in list(scores.values())]

    p_value = ttest_1samp(values, 0)[1]

    return values, p_value, list(scores.keys())


def plot_heatmap(models: list[str], results_dir: str) -> pd.DataFrame:
    """Plot heatmap comparing LLMs’ behavior on bias types with their respective behavior on the set of perturbations.

    Blue indicates a positive effect, orange indicates a negative effect, hatched cells indicate non-significant change.
    """
    clean_model_labels = list(models)

    all_results = []

    # for model in models:
    #     for i in range(len(bias_types)):
    for model, i in tqdm_itertools.product(models, range(len(bias_types))):
        bias_type = bias_types[i]
        values, p_value, keys = run_stat_test(
            bias_type,
            f"{results_dir}/{model}/csv/{bias_type}.csv",  # TODO: fix this path
        )
        lst = [model, clean_bias_labels[i], mean(values), p_value]

        for perturbation in perturbations:
            if bias_types[i] == "opinion_float":  # qustions are the same
                bias_type = "odd_even" + perturbation
            else:
                bias_type = bias_types[i] + perturbation

            values, p_value, keys = run_stat_test(
                bias_type,
                f"{results_dir}/{model}/csv/{bias_type}.csv",  # TODO: fix this path
            )
            lst += [mean(values), p_value]

        all_results.append(lst)

    df = pd.DataFrame(
        all_results,
        columns=[
            "model",
            "bias type",
            "modified",
            "bias p value",
            "key typo",
            "key typo p value",
            "middle random",
            "middle random p value",
            "letter swap",
            "letter swap p value",
        ],
    )
    df = df.round(4)

    # plot heatmap
    models = list(models) + ["ideal"]
    clean_model_labels += ["Most Human-like"]

    fig, axs = plt.subplots(2, len(models) // 2, figsize=(15, 6))

    cmap_name = "tab20c"

    for i in range(len(models)):
        model = models[i]

        effect_data = np.zeros((len(bias_types), len(exp_settings)))
        effect_values = [
            [" ", "", "", ""],
            ["", "", "", ""],
            ["", "", "", ""],
            ["", "", "", ""],
            ["", "", "", ""],
        ]
        p_values = np.zeros((len(bias_types), len(exp_settings)))

        for k in range(len(exp_settings)):
            for j in range(len(bias_types)):
                exp_setting = exp_settings[k]

                if exp_setting == "modified":
                    p_val_col = "bias p value"

                elif exp_setting == "key typo":
                    p_val_col = "key typo p value"

                elif exp_setting == "middle random":
                    p_val_col = "middle random p value"

                elif exp_setting == "letter swap":
                    p_val_col = "letter swap p value"

                if model == "ideal":
                    p_value = 0.01 if k == 0 else 1

                    if p_value < 0.05:
                        effect_data[j][k] = -0.7
                    else:
                        effect_data[j][k] = np.nan

                    p_values[j][k] = p_value

                else:
                    effect_size = df[
                        (df["bias type"] == clean_bias_labels[j])
                        & (df["model"] == model)
                    ][exp_setting]
                    p_value = df[
                        (df["bias type"] == clean_bias_labels[j])
                        & (df["model"] == model)
                    ][p_val_col]

                    p_values[j][k] = p_value.item()
                    if p_value.item() < 0.05:
                        if effect_size.item() > 0:
                            effect_data[j][k] = -0.7
                            effect_values[j][k] = round(effect_size.item(), 1)
                        else:
                            effect_data[j][k] = -0.3
                            effect_values[j][k] = round(effect_size.item(), 1)
                    else:
                        effect_data[j][k] = np.nan

        r = i // 5
        c = i % 5

        if r == 0 and c == 0:
            sns.heatmap(
                effect_data,
                annot=effect_values,
                fmt="",
                xticklabels=False,
                cbar=False,
                ax=axs[r, c],
                cmap=cmap_name,
                vmin=-1,
                vmax=1,
                linewidths=1,
                linecolor="gray",
            )
            tickvalues = [num + 0.5 for num in range(0, len(exp_settings))]
            axs[r, c].set_title(clean_model_labels[i])

            tickvalues1 = [num + 0.5 for num in range(0, len(bias_types))]
            axs[r, c].set_yticks(tickvalues1)
            axs[r, c].set_yticklabels(clean_bias_labels, rotation=0)

        if r == 0 and c != 0:
            sns.heatmap(
                effect_data,
                annot=effect_values,
                fmt="",
                xticklabels=False,
                yticklabels=False,
                cbar=False,
                ax=axs[r, c],
                cmap=cmap_name,
                vmin=-1,
                vmax=1,
                linewidths=1,
                linecolor="gray",
            )
            axs[r, c].set_title(clean_model_labels[i])

        if r == 1 and c == 0:
            sns.heatmap(
                effect_data,
                annot=effect_values,
                fmt="",
                cbar=False,
                ax=axs[r, c],
                cmap=cmap_name,
                vmin=-1,
                vmax=1,
                linewidths=1,
                linecolor="gray",
            )
            tickvalues = [num + 0.5 for num in range(0, len(exp_settings))]
            axs[r, c].set_xticks(tickvalues)
            axs[r, c].set_xticklabels(clean_labels, rotation=90)
            axs[r, c].set_title(clean_model_labels[i])

            tickvalues1 = [num + 0.5 for num in range(0, len(bias_types))]
            axs[r, c].set_yticks(tickvalues1)
            axs[r, c].set_yticklabels(clean_bias_labels, rotation=0)

        if r == 1 and c != 0:
            sns.heatmap(
                effect_data,
                annot=effect_values,
                fmt="",
                yticklabels=False,
                cbar=False,
                ax=axs[r, c],
                cmap=cmap_name,
                vmin=-1,
                vmax=1,
                linewidths=1,
                linecolor="gray",
            )
            tickvalues = [num + 0.5 for num in range(0, len(exp_settings))]
            axs[r, c].set_xticks(tickvalues)
            axs[r, c].set_xticklabels(clean_labels, rotation=90)
            axs[r, c].set_title(clean_model_labels[i])

        zm = np.ma.masked_less(p_values, 0.05)

        x = np.arange(effect_data.shape[1] + 1)
        y = np.arange(effect_data.shape[0] + 1)

        axs[r, c].pcolor(x, y, zm, hatch="//", alpha=0.0)

    plt.savefig("perturbation.pdf", format="pdf", bbox_inches="tight")
    return df


def get_entropies(bias_type: str, pkl_file: str) -> tuple[float, float, float, float]:
    """Get entropies."""
    first_group, second_group, _, _ = get_groups(bias_type)

    df = pd.read_pickle(pkl_file)

    entropies = []
    norm_counts = []

    for index, row in df.iterrows():
        num_options = row["num_options"]
        if "odd_even" == bias_type and row["type"] == "no middle alpha":
            num_options = 4
        elif "opinion_float" == bias_type and row["type"] == "float alpha":
            num_options = 6

        temp = row["responses"].replace(" ", "").split(",")
        cnts = sorted(Counter(temp).items(), key=itemgetter(0))
        final_counts = [itm_count for _, itm_count in cnts]
        norm_final_counts = [
            itm_count / sum(final_counts) for itm_count in final_counts
        ]
        entropies.append(entropy(norm_final_counts) / np.log(num_options))

        norm_counts.append(norm_final_counts)

    df["entropy"] = entropies
    df["norm counts"] = norm_counts

    orig_entropy = []
    new_entropy = []
    entropy_diffs = []
    for key in df["key"].unique():
        subset_df = df[df["key"] == key][["key", "type", "entropy", "norm counts"]]
        entropy_diff = (
            subset_df.loc[subset_df["type"] == first_group, "entropy"].item()
            - subset_df.loc[subset_df["type"] == second_group, "entropy"].item()
        )
        entropy_diffs.append(entropy_diff)
        orig_entropy.append(
            subset_df.loc[subset_df["type"] == second_group, "entropy"].item()
        )
        new_entropy.append(
            subset_df.loc[subset_df["type"] == first_group, "entropy"].item()
        )

    return (
        round(np.mean(orig_entropy), 2),
        round(np.var(orig_entropy), 2),
        round(np.mean(new_entropy), 2),
        round(np.var(new_entropy), 2),
    )


def plot_uncertainity(models: list[str], results_dir: str) -> pd.DataFrame:
    """Plot uncertainity."""
    clean_model_labels = list(models)
    all_results = []
    # for model in models:
    #     for i in range(len(bias_types)):
    for model, i in tqdm_itertools.product(models, range(len(bias_types))):
        bias_type = bias_types[i]
        orig_mean, orig_std, new_mean, new_std = get_entropies(
            bias_type,
            f"{results_dir}/{model}/{bias_type}.pickle",  # TODO: fix this path
        )
        lst = [model, bias_type, orig_mean, orig_std, new_mean, new_std]
        for perturbation in perturbations:
            if bias_types[i] == "opinion_float":  # qustions are the same
                bias_type = "odd_even" + perturbation
            else:
                bias_type = bias_types[i] + perturbation
            orig_mean, orig_std, new_mean, new_std = get_entropies(
                bias_type,
                f"{results_dir}/{model}/{bias_type}.pickle",  # TODO: fix this path
            )
            lst += [new_mean, new_std]
        all_results.append(lst)
    df = pd.DataFrame(
        all_results,
        columns=[
            "model",
            "bias type",
            "original mean",
            "original std",
            "bias mean",
            "bias std",
            "key typo mean",
            "key typo std",
            "middle random mean",
            "middle random std",
            "letter swap mean",
            "letter swap std",
        ],
    )
    df = df.round(4)
    fig, axs = plt.subplots(len(models), len(bias_types), figsize=(10, 14))

    for i in range(len(models)):
        for j in range(len(bias_types)):
            row = df.loc[
                (df["model"] == models[i]) & (df["bias type"] == bias_types[j])
            ]

            x = ["bias", "key typo", "middle random", "letter swap"]
            y = [
                row["bias mean"].item(),
                row["key typo mean"].item(),
                row["middle random mean"].item(),
                row["letter swap mean"].mean(),
            ]
            e = [
                row["bias std"].item(),
                row["key typo std"].item(),
                row["middle random std"].item(),
                row["letter swap std"].mean(),
            ]
            axs[i, j].errorbar(x, y, yerr=e, fmt="o")
            axs[i, j].axhline(y=row["original mean"].item(), color="r", linestyle="-")
            axs[i, j].set_ylim(0, 1)

            axs[i, j].set_xticks(range(len(clean_labels)))
            axs[i, j].set_xticklabels(clean_labels, rotation=90)

            axs[i, j].set_title(clean_bias_labels[j])
            if i != len(models) - 1:
                axs[i, j].get_xaxis().set_visible(False)

    grid = plt.GridSpec(len(models), len(bias_types))
    for k in range(len(clean_model_labels)):
        axes = fig.add_subplot(grid[k, ::])
        axes.set_title(f"{clean_model_labels[k]}\n", fontweight="semibold")
        axes.set_frame_on(False)
        axes.axis("off")
    fig.tight_layout()
    plt.savefig("uncertainty.pdf", format="pdf", bbox_inches="tight")
    return df
