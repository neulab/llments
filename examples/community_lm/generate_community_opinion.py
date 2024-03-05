"""

This script uses community GPT models to generate opinions given prompts.

"""

from llments.lm.lm import LanguageModel
import os
import tqdm
from pathlib import Path
from constants import anes_df


def generate_community_opinion(
    model: LanguageModel,
    prompt_option: str,
    output_path: str,
    seed: int,
    preceding_prompt: str | None = None,
    overwrite: bool = False,
):
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
