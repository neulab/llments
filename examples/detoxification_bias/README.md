# Detoxifying Language Models Risks Marginalizing Minority Voices

This is a replication of the experiments from
[The 'Detoxifying' paper](https://aclanthology.org/2021.naacl-main.190)
(Xu et al., NAACL 2021), which investigates the impact of
detoxification on the performance of language models on language patterns
concerning minority groups.

## Dependencies

To better align with the original implementation in the paper,
we recommend using the version of the dependencies in the`requirements.txt`
file. You can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Data Preparation

Before runnning code to preprocess the data, make sure you have create
the following directories to store the data:

```bash
mkdir -p data/raw/civilcomments data/raw/translation_pairs
mkdir -p data/train/ft
mkdir -p data/eval/translation_pairs/scored/ 
mkdir -p data/eval/translation_pairs/filtered/
```

Then, download the Jigsaw Civil Comments dataset from Kaggle,
save and unzip the target file under the `data/raw/civilcomments` directory.

The evaluation data is the AAVE/SAE Paired Dataset, which can be find in this
[link](https://aclanthology.org/2020.emnlp-main.473/) of its ACL main page.
After downloading the data, save and unzip the target file under the
`data/raw/translation_pairs` directory.

## Reference

Some of this code and data was derived from the
[Detoxification repo](https://github.com/albertkx/detoxifying-lms).

If you use this example, we would appreciate if you acknowledge LLMents
and the original paper and datasets.

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
@inproceedings{Xu2021Detoxifying,
    Title = {Detoxifying Language Models Risks Marginalizing Minority Voices},
    Author = {
        Albert Xu and 
        Eshaan Pathak and 
        Eric Wallace and 
        Suchin Gururangan and 
        Maarten Sap and 
        Dan Klein},
    Booktitle = {
        North American Chapter of the Association 
        for Computational Linguistics}
    year={2021}
}
```

```bibtex
@inproceedings{groenwold-etal-2020-investigating,
    title = "Investigating {A}frican-{A}merican 
        {V}ernacular {E}nglish in Transformer-Based Text Generation",
    author = "
        Groenwold, Sophie and 
        Ou, Lily and 
        Parekh, Aesha and 
        Honnavalli, Samhita and 
        Levy, Sharon and 
        Mirza, Diba and 
        Wang, William Yang",
    booktitle = "Proceedings of EMNLP",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.473",
    year = "2020"
}
```
