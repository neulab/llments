"""This script uses community GPT models to generate opinions given prompts."""

import os
import tqdm
import csv
from pathlib import Path
from community_lm_constants import anes_df
import pandas as pd
import numpy as np

from llments.lm.lm import LanguageModel
from llments.lm.rag import RAGLanguageModel
from llments.eval.sentiment import SentimentEvaluator

def generate_community_opinion_rag(
    model: RAGLanguageModel,
    prompt_option: str,
    output_path: str,
    seed: int,
    preceding_prompt: str | None = None,
    overwrite: bool = False,
) -> None:
    """Generate opinions for a given prompt with RAG.

    Args:
        model: The RAG language model.
        prompt_option: The prompt option.
        output_path: The output path.
        seed: The seed for the language model.
        preceding_prompt: The preceding prompt.
        overwrite: Whether to overwrite the output file if it exists.
    """
    model.set_seed(seed)

    questions = anes_df.pid.values.tolist()
    prompts = anes_df[prompt_option].values.tolist()

    output_folder = os.path.join(output_path, prompt_option)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for question, prompt in tqdm.tqdm(
        zip(questions, prompts), total=len(questions), desc="Generating opinions"
    ):
        output_path = os.path.join(output_folder, question + ".txt")
        if os.path.exists(output_path) and not overwrite:
            continue
        total_prompt = (
            " ".join([preceding_prompt, prompt]) if preceding_prompt else prompt
        )

        responses = model.generate(
            condition=total_prompt,
            do_sample=True,
            max_new_tokens=100,
            temperature=1.0,
            num_return_sequences=100,
        )

        responses = [x.split("\n")[0] for x in responses]

        with open(output_path, "w") as out:
            for line in responses:
                line = line.replace("\n", " ")
                if preceding_prompt:
                    line = line.replace(preceding_prompt + " ", "")
                out.write(line)
                out.write("\n")


def generate_community_opinion(
    model: LanguageModel,
    prompt_option: str,
    output_path: str,
    seed: int,
    preceding_prompt: str | None = None,
    overwrite: bool = False,
) -> None:
    """Generate opinions for a given prompt.

    Args:
        model: The language model.
        prompt_option: The prompt option.
        output_path: The output path.
        seed: The seed for the language model.
        preceding_prompt: The preceding prompt.
        overwrite: Whether to overwrite the output file if it exists.
    """
    model.set_seed(seed)

    questions = anes_df.pid.values.tolist()
    prompts = anes_df[prompt_option].values.tolist()

    output_folder = os.path.join(output_path, prompt_option)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for question, prompt in tqdm.tqdm(
        zip(questions, prompts), total=len(questions), desc="Generating opinions"
    ):
        output_path = os.path.join(output_folder, question + ".txt")
        if os.path.exists(output_path) and not overwrite:
            continue
        total_prompt = (
            " ".join([preceding_prompt, prompt]) if preceding_prompt else prompt
        )
        responses = model.generate(
            total_prompt,
            do_sample=True,
            max_length=50,
            temperature=1.0,
            num_return_sequences=1000,
        )
        responses = [x.split("\n")[0] for x in responses]

        with open(output_path, "w") as out:
            for line in responses:
                line = line.replace("\n", " ")
                if preceding_prompt:
                    line = line.replace(preceding_prompt + " ", "")
                out.write(line)
                out.write("\n")


def compute_group_stance(
    evaluator: SentimentEvaluator,
    data_folder: str,
    output_filename: str,
    overwrite: bool = False,
) -> None:
    """Compute the group sentiment for a set of generated opinions.

    Args:
        evaluator: The sentiment evaluator.
        data_folder: The folder containing the generated opinions.
        output_filename: The output filename.
        overwrite: Whether to overwrite the output file if it exists.
    """
    if not overwrite and os.path.exists(output_filename):
        return

    questions = anes_df.pid.values.tolist()
    model_name = data_folder.strip("/").split("/")[-1]

    columns = ["model_name", "run", "prompt_format", "question", "group_sentiment"]
    rows = []
    for run_id in range(1, 6):
        run_format = f"run_{run_id}"
        print(f"Processing {run_format} ...")
        for prompt_id in range(1, 5):
            prompt_format = f"Prompt{prompt_id}"
            for question in tqdm.tqdm(questions, "Processing questions"):
                file_name = os.path.join(
                    data_folder, run_format, prompt_format, question + ".txt"
                )
                with open(file_name) as f:
                    file_lines = f.readlines()
                sentiment_vals = evaluator.evaluate_batch(
                    file_lines, minibatch_size=len(file_lines)
                )
                group_sentiment = np.mean(sentiment_vals) * 100
                rows.append(
                    [model_name, run_format, prompt_format, question, group_sentiment]
                )

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_filename)

def compute_group_stance_factscore(
    evaluator: SentimentEvaluator,
    input_filename: str,
) -> dict[str, float]:
    """
    Calculates group sentiment for the democratic and republican parties.

    Args:
        input_filename (str): The input filename.
        evaluator: The sentiment evaluator.

    Returns:
        dict: A dictionary with keys 'democratic' and 'republican' containing their respective sentiments.
    """
    democratic_responses = []
    republican_responses = []
    
    try:
        with open(input_filename, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                party = row['Party'].strip().lower()
                response = row['Response'].strip()
                if not response:
                    continue  # Skip empty responses
                if party == 'democrats':
                    democratic_responses.append(response)
                elif party == 'republicans':
                    republican_responses.append(response)
                else:
                    print(f"Warning: Unknown party '{party}' in row: {row}")

        # Function to evaluate sentiments
        def evaluate_sentiments(responses, party_name):
            if not responses:
                print(f"No responses found for party: {party_name.capitalize()}")
                return None
            sentiment_vals = evaluator.evaluate_batch(responses, minibatch_size=len(responses))
            group_sentiment = np.mean(sentiment_vals) * 100
            return group_sentiment

        # Calculate sentiments for each group
        sentiments = {}
        sentiments['democratic'] = evaluate_sentiments(democratic_responses, 'democratic')
        sentiments['republican'] = evaluate_sentiments(republican_responses, 'republican')
        return sentiments

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' does not exist.")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}
