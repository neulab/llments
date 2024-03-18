"""

This script uses community GPT models to generate opinions given prompts.

"""

import os
import tqdm
from pathlib import Path
from community_lm_constants import anes_df
import pandas as pd
import numpy as np

from llments.lm.lm import LanguageModel
from llments.eval.sentiment import SentimentEvaluator


def generate_community_opinion(
    model: LanguageModel,
    prompt_option: str,
    output_path: str,
    seed: int,
    preceding_prompt: str | None = None,
    overwrite: bool = False,
) -> None:
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
