# FairEval

This is a replication of the experiments from
[FairEval](https://arxiv.org/abs/2305.17926) (Wang et al. 2023), which
critically examines the LLMs-as-evaluator paradigm.

## Dependencies

To better align with the original implementation in the paper,
we recommend using the version of the dependencies in the `requirements.txt` file.
You can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Configuration

### OpenAI API Key

To access and use OpenAI's services (such as GPT models), you must obtain an API key from OpenAI.
After acquiring your API key, enter it in the notebook code.
Typically, this involves setting a variable or configuring an environment variable.
For example, in your notebook:

```python
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
openai.api_key = os.environ["OPENAI_API_KEY"]
```

Alternatively, you can set the API key in your environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Reference

Some of this code and data was derived from the
[FairEval repo](https://github.com/i-eval/faireval).

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
@article{Wang2023LargeLM,
  title={Large Language Models are not Fair Evaluators},
  author={Peiyi Wang and Lei Li and Liang Chen and Dawei Zhu and Binghuai Lin and Yunbo Cao and Qi Liu and Tianyu Liu and Zhifang Sui},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.17926},
}
```
