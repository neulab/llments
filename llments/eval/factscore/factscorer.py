"""FactScore Scoring Module."""
import argparse
import string
import json
import numpy as np
import os
import logging
from typing import List, Optional, Dict, Any, Union, cast

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval

class FactScorer:
    """FactScorer Class.

    This class integrates various language models and retrieval mechanisms to evaluate the factual accuracy
    of generated text. It supports different configurations, including retrieval-based models with ChatGPT,
    LLaMA, and NPM. The class handles caching of retrieval results and generated atomic facts to optimize
    performance and reduce redundant computations. It also includes functionality for cost estimation related
    to OpenAI API usage.

    Attributes:
        model_name (str): Configuration of the language model to use.
        db (Dict[str, DocDB]): Dictionary mapping knowledge source names to DocDB instances.
        retrieval (Dict[str, Retrieval]): Dictionary mapping knowledge source names to Retrieval instances.
        npm (Dict[str, NPM]): Dictionary mapping knowledge source names to NPM instances.
        batch_size (int): Batch size for retrieval processes.
        openai_key (str): Path to the OpenAI API key file.
        abstain_detection_type (Optional[str]): Type of abstain detection to use.
        data_dir (str): Directory for storing data files.
        cache_dir (str): Directory for storing cache files.
        af_generator (Optional[AtomicFactGenerator]): Instance of AtomicFactGenerator for generating atomic facts.
        cost_estimate (str): Strategy for estimating API costs.
        lm (Optional[Union[CLM, OpenAIModel]]): Language model instance (CLM or OpenAIModel).
    """
    def __init__(
        self,
        model_name: str = "retrieval+ChatGPT",
        data_dir: str = "/factscore_data",
        model_dir: str = "/factscore_data",
        cache_dir: str = "/factscore_data",
        openai_key: str = "key.txt",
        cost_estimate: str = "consider_cache",
        abstain_detection_type: Optional[str] = None,
        batch_size: int = 256,
    ) -> None:
        """Initialize the FactScorer instance.

        Args:
            model_name (str, optional): Configuration of the language model to use.
                Defaults to "retrieval+ChatGPT".
            data_dir (str, optional): Directory for storing data files.
                Defaults to "/factscore_data".
            model_dir (str, optional): Directory for storing models.
                Defaults to "/factscore_data".
            cache_dir (str, optional): Directory for storing cache files.
                Defaults to "/factscore_data".
            openai_key (str, optional): Path to the OpenAI API key file.
                Defaults to "key.txt".
            cost_estimate (str, optional): Strategy for estimating API costs.
                Options: "consider_cache", "ignore_cache".
                Defaults to "consider_cache".
            abstain_detection_type (Optional[str], optional): Type of abstain detection to use.
                Options: "perplexity_ai", "generic", "none".
                Defaults to None.
            batch_size (int, optional): Batch size for retrieval processes.
                Defaults to 256.

        Raises:
            AssertionError: If `model_name` is not among the supported configurations.
        """
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name

        self.db: Dict[str, DocDB] = {}
        self.retrieval: Dict[str, Retrieval] = {}
        self.npm: Dict[str, NPM] = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        else:
            self.lm = None

    def save_cache(self) -> None:
        """Save caches for the language model, NPM instances, and retrieval instances.

        This method ensures that any new entries added to the caches are persisted to their respective
        cache files to optimize performance and avoid redundant computations.
        """
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(
        self,
        name: str = "enwiki-20230401",
        db_path: Optional[str] = None,
        data_path: Optional[str] = None,
    ) -> None:
        """Register a new knowledge source for retrieval.

        This method initializes a new `DocDB` and `Retrieval` instance for the specified knowledge source.
        If NPM is included in the model configuration, it also initializes an `NPM` instance for the knowledge source.

        Args:
            name (str, optional): Name of the knowledge source. Defaults to "enwiki-20230401".
            db_path (Optional[str], optional): Path to the SQLite database file.
                If not provided, defaults to `<data_dir>/<name>.db`. Defaults to None.
            data_path (Optional[str], optional): Path to the raw data file for building the database.
                If not provided, defaults to `<data_dir>/<name>.jsonl`. Defaults to None.

        Raises:
            AssertionError: If the knowledge source `name` is already registered.
        """
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(
        self,
        total_words: int,
        task: str,
        model: str,
    ) -> None:
        """Print the estimated cost of OpenAI API usage based on the number of tokens.

        Args:
            total_words (int): Total number of words to be processed.
            task (str): Description of the task for which cost is being estimated.
            model (str): OpenAI model used for the task.
        """
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use gpt-3.5-turbo-instruct, the cost is $0.002 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        rate = 0.0
        if model == "gpt-3.5-turbo-instruct":
            rate = 0.002
        elif model == "gpt-3.5-turbo":
            rate = 0.002
        else:
          rate = 0.0

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(
        self,
        topics: Union[str, List[str]],
        generations: Union[str, List[str]],
        gamma: int = 10,
        atomic_facts: Optional[List[List[str]]] = None,
        knowledge_source: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Compute the factual accuracy score for the provided generations based on topics.

        This method retrieves relevant passages for each topic, generates or uses provided atomic facts,
        evaluates whether the generated content is supported by the retrieved knowledge, and computes
        an overall factual accuracy score.

        Args:
            topics (Union[str, List[str]]): Single topic string or a list of topic strings.
            generations (Union[str, List[str]]): Single generation string or a list of generation strings.
            gamma (int, optional): Hyperparameter for length penalty. Defaults to 10.
            atomic_facts (Optional[List[List[str]]], optional): Precomputed atomic facts for each generation.
                If not provided, atomic facts will be generated. Defaults to None.
            knowledge_source (Optional[str], optional): Name of the knowledge source to use for retrieval.
                Defaults to "enwiki-20230401".
            verbose (bool, optional): If True, display progress bars. Defaults to False.

        Returns:
            Dict[str, Any]: Dictionary containing the overall score, respond ratio, decisions,
                and average number of atomic facts per valid response.
        """
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, "InstructGPT.pkl"))

            # estimate the total cost of atomic fact generation
            assert self.af_generator is not None, "AtomicFactGenerator should be initialized."
            
            total_words = 0
            for gen in generations:
                total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="atomic fact generation", model="gpt-3.5-turbo-instruct")

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    result = self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)
                    total_words += cast(int, result)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None:
                decisions.append(None)
            else:
                decision = cast(List[Dict[str, bool]], self._get_score(topic, generation, facts, knowledge_source))
                score = np.mean([d["is_supported"] for d in decision])
                
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out
    def _get_score(
        self,
        topic: str,
        generation: str,
        atomic_facts: List[str],
        knowledge_source: str,
        cost_estimate: Optional[str] = None,
    ) -> Union[List[Dict[str, bool]], int]:
        """Compute support scores for each atomic fact based on the knowledge source.

        This internal method evaluates whether each atomic fact is supported by the retrieved passages
        using the configured language model and NPM (if applicable).

        Args:
            topic (str): The topic associated with the generation.
            generation (str): The generated text to evaluate.
            atomic_facts (List[str]): List of atomic facts extracted from the generation.
            knowledge_source (str): Name of the knowledge source to use for retrieval.
            cost_estimate (Optional[str], optional): Strategy for estimating API costs.
                If set, the method only accumulates the number of words without generating scores.
                Defaults to None.

        Returns:
            Union[List[Dict[str, bool]], int]:
                - If `cost_estimate` is provided, returns the total number of words processed.
                - Otherwise, returns a list of decision dictionaries indicating support for each atomic fact.
        """
        decisions = []
        total_words = 0
        for atom in atomic_facts:
            atom = atom.strip()
            if self.lm:
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())

                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue

                output = self.lm.generate(prompt)

                if type(output[1])==np.ndarray:
                    # when logits are available
                    logits = np.array(output[1])
                    assert logits.shape[0] in [32000, 32001]
                    true_score = logits[5852]
                    false_score = logits[7700]
                    is_supported = true_score > false_score
                else:
                    # when logits are unavailable
                    generated_answer = output[0].lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            else:
                is_supported = True

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3

            decisions.append({"atom": atom, "is_supported": is_supported})

        if cost_estimate:
            return total_words
        else:
            return decisions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="key.txt")
    parser.add_argument('--data_dir',
                        type=str,
                        default="/factscore_data")
    parser.add_argument('--model_dir',
                        type=str,
                        default="/factscore_data")
    parser.add_argument('--cache_dir',
                        type=str,
                        default="/factscore_data")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)


    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")    
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.n_samples is not None and tot==args.n_samples:
                break
    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    # Save out as a json file
    with open(args.input_path.replace(".jsonl", f"_factscore_output.json"), 'w') as f:
        f.write(json.dumps(out) + "\n")

