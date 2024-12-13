# FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation

This is a replication of the experiments from
[FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form
Text Generation](https://aclanthology.org/2023.emnlp-main.741) (Min et al., EMNLP
2023).

## Dependencies

To better align with the original implementation in the paper,
we recommend using the version of the dependencies in the `requirements.txt` file.
You can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Configuration

### OpenAI API Key

To access and use OpenAI's services (such as GPT models),
you must obtain an API key from OpenAI.
After acquiring your API key, store it in a txt file
such as `key.txt` amd pass it when creating an instance
of the `FactScorer` class.

### Data Preparation

Before running the code to generate the CommunityLM responses,
make sure you have created the following directory to store the data:

```bash
mkdir -p factscore_data
```

This will be the data directory for the FactScore analysis
and all CSV files must be inside this folder.

## Reference

Some of this code and data was derived from the
[FActScore repo](https://github.com/shmsw25/FActScore).

If you use this example, we would appreciate if you acknowledge
[LLMents](https://github.com/neulab/llments) and the original paper.

```bibtex
@misc{
    title = "{LLMents}: A Toolkit for Language Model Experiments",
    author = "
      Graham Neubig and
      Aakriti Kinra and
      Mihir Bansal and
      Qingyang Liu and
      Rohan Modi and
      Xinran Wan
    ",
    year = "2024",
    howpublished = "https://github.com/neulab/llments",
}
```

```bibtex
@inproceedings{min-etal-2023-factscore,
    title = "{FA}ct{S}core: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation",
    author = "Min, Sewon  and
      Krishna, Kalpesh  and
      Lyu, Xinxi  and
      Lewis, Mike  and
      Yih, Wen-tau  and
      Koh, Pang  and
      Iyyer, Mohit  and
      Zettlemoyer, Luke  and
      Hajishirzi, Hannaneh",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/2023.emnlp-main.741",
    }
```
