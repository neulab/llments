"""Atomic Facts Module."""
import json
import numpy as np
import re
import string
import spacy
import nltk
from rank_bm25 import BM25Okapi
import os
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Optional, Any, Set, cast

from factscore.openai_lm import OpenAIModel

nltk.download("punkt")

class AtomicFactGenerator:
    """A generator class to convert AI-generated text into atomic facts.

    Attributes:
        nlp (spacy.lang.en.English): The spaCy language model for NLP tasks.
        is_bio (bool): Flag indicating if the generator is in bio mode.
        demon_path (str): Path to the demons JSON file.
        openai_lm (OpenAIModel): Instance of the OpenAI language model.
        demons (Dict[str, List[str]]): Dictionary of demon sentences and their corresponding facts.
        bm25 (BM25Okapi): BM25 instance for efficient retrieval of top matching demons.
    """
    def __init__(
        self,
        key_path: str = "key.txt",
        demon_dir: str = "/factscore_data",
        gpt3_cache_file: Optional[str] = None,
    ) -> None:
        """Initialize the AtomicFactGenerator.

        Args:
            key_path (str, optional): Path to the OpenAI API key file. Defaults to "key.txt".
            demon_dir (str, optional): Directory containing the demons JSON file. Defaults to "/factscore_data".
            gpt3_cache_file (Optional[str], optional): Path to the GPT-3 cache file. Defaults to None.
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = True
        self.demon_path = os.path.join(demon_dir, "demons.json" if self.is_bio else "demons_complex.json")

        self.openai_lm = OpenAIModel("InstructGPT", cache_file=gpt3_cache_file, key_path=key_path)

        # get the demos
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_cache(self) -> None:
        """Save the OpenAI language model cache."""
        self.openai_lm.save_cache()

    def run(
        self, generation: str, cost_estimate: Optional[Any] = None
    ) -> Tuple[List[Tuple[str, List[str]]], List[int]]:
        """Convert the generation into a set of atomic facts. Return a total words cost if cost_estimate != None.

        Args:
            generation (str): The AI-generated text to be processed.
            cost_estimate (Optional[Any], optional): If provided, returns the total word cost. Defaults to None.

        Returns:
            Tuple[List[Tuple[str, List[str]]], List[int]]:
                - List of tuples containing sentences and their corresponding atomic facts.
                - List of paragraph break indices.
        """
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        return self.get_atomic_facts_from_paragraph(paragraphs, cost_estimate=cost_estimate)

    def get_atomic_facts_from_paragraph(
        self, paragraphs: List[str], cost_estimate: Optional[Any] = None
    ) -> Tuple[List[Tuple[str, List[str]]], List[int]]:
        """Extract atomic facts from a list of paragraphs.

        Args:
            paragraphs (List[str]): List of paragraph texts.
            cost_estimate (Optional[Any], optional): If provided, returns the total word cost. Defaults to None.

        Returns:
            Tuple[List[Tuple[str, List[str]]], List[int]]:
                - List of tuples containing sentences and their corresponding atomic facts.
                - List of paragraph break indices.
        """
        sentences: List[str] = []
        para_breaks: List[int] =  []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0 :
                para_breaks.append(len(sentences))

            initials = detect_initials(paragraph)

            curr_sentences = sent_tokenize(paragraph)
            curr_sentences_2 = sent_tokenize(paragraph)

            curr_sentences = fix_sentence_splitter(curr_sentences, initials)
            curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

            # checking this, just to ensure the crediability of the sentence splitter fixing algorithm
            assert curr_sentences == curr_sentences_2, (paragraph, curr_sentences, curr_sentences_2)

            sentences += curr_sentences

        atoms_or_estimate = self.get_init_atomic_facts_from_sentence([sent for i, sent in enumerate(sentences) if not (not self.is_bio and ( \
                            (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                            (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))))], cost_estimate=cost_estimate)

        if cost_estimate:
            return cast(Tuple[List[Tuple[str, List[str]]], List[int]], atoms_or_estimate)
        else:
            atoms = atoms_or_estimate

        atomic_facts_pairs: List[Tuple[str, List[str]]] = []
        for i, sent in enumerate(sentences):
            if not self.is_bio and ( \
                (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))):
                atomic_facts_pairs.append((sent, []))
            elif self.is_bio and sent.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((sent, []))
            elif sent.startswith("Sure") or sent.startswith("Please") or (i==0 and sent.startswith("Here are")):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        # postprocess_atomic_facts will fix minor issues from InstructGPT
        # it is supposed to handle sentence splitter issue too, but since here
        # we fixed sentence splitter issue already,
        # the new para_breaks should be identical to the original para_breaks
        if self.is_bio:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)

        return atomic_facts_pairs, para_breaks

    def get_init_atomic_facts_from_sentence(
        self, sentences: List[str], cost_estimate: Optional[Any] = None
    ) -> Any:
        """Get the initial atomic facts from the sentences. Return a total words cost if cost_estimate != None.

        Args:
            sentences (List[str]): List of sentences to process.
            cost_estimate (Optional[Any], optional): If provided, returns the total word cost. Defaults to None.

        Returns:
            Any: Either a total word count estimate or a dictionary mapping sentences to their atomic facts.
        """
        is_bio = self.is_bio
        demons = self.demons

        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        prompts = []
        prompt_to_sent = {}
        atoms = {}
        for sentence in sentences:
            if sentence in atoms:
                continue
            top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            prompt = ""

            for i in range(n):
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(list(demons.keys())[i])
                for fact in demons[list(demons.keys())[i]]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"

            for match in top_machings:
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(match)
                for fact in demons[match]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"
            prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(sentence)
            prompts.append(prompt)
            prompt_to_sent[prompt] = sentence

        if cost_estimate:
            total_words_estimate = 0
            for prompt in prompts:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
                    continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            for prompt in prompts:
                output, _ = self.openai_lm.generate(prompt)
                atoms[prompt_to_sent[prompt]] = text_to_sentences(output)

            for key, value in demons.items():
                if key not in atoms:
                    atoms[key] = value

            return atoms


def best_demos(query: str, bm25: BM25Okapi, demons_sents: List[str], k: int) -> :
    """Retrieve the top matching demons for a given query using BM25.

    Args:
        query (str): The query sentence.
        bm25 (BM25Okapi): The BM25 instance for retrieval.
        demons_sents (List[str]): List of demon sentences.
        k (int): Number of top matches to retrieve.

    Returns:
        List[str]: List of top matching demon sentences.
    """
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return cast(List[str], top_machings) 

def text_to_sentences(text: str) -> List[str]:
    """Transform InstructGPT output into a list of sentences.

    Args:
        text (str): The raw output text from InstructGPT.

    Returns:
        List[str]: List of cleaned sentences extracted from the text.
    """
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if len(sentences) > 0: 
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.' 
    else:
        sentences = []
    return sentences

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Args:
        s (str): The input string to normalize.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text: str) -> str:
        return " ".join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text: str) -> str:
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text: str) -> bool:
    """Check if the given text represents an integer number.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text is an integer, False otherwise.
    """
    try:
        int(text)
        return True
    except Exception:
        return False

def is_date(text: str) -> bool:
    """Determine if the given text represents a date.

    Args:
        text (str): The text to evaluate.

    Returns:
        bool: True if the text is a date, False otherwise.
    """
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text: str) -> Set[str]:
    """Extract all unique numeric values from the text.

    Args:
        text (str): The input text.

    Returns:
        Set[str]: A set of unique numeric string values found in the text.
    """
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list

def detect_entities(text: str, nlp: spacy.lang.en.English) -> Set[str]:
    """Detect relevant entities in the text using spaCy's NLP model.

    Args:
        text (str): The input text to analyze.
        nlp (spacy.lang.en.English): The spaCy NLP model.

    Returns:
        Set[str]: A set of detected entity strings.
    """
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text: str) -> None:
        """Add text to the entities set, splitting by hyphens if present.

        If the input text contains hyphens, split it into separate entities.
        Otherwise, add the text directly to the entities set.

        Args:
            text (str): The text to add to entities.
        """
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)

    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)
        
    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities

def postprocess_atomic_facts(
    _atomic_facts: List[Tuple[str, List[str]]],
    para_breaks: List[int],
    nlp: spacy.lang.en.English,
) -> Tuple[List[Tuple[str, List[str]]], List[int]]:
    """Post-process atomic facts to fix minor issues and ensure consistency.

    Args:
        _atomic_facts (List[Tuple[str, List[str]]]): Initial list of atomic facts.
        para_breaks (List[int]): List of paragraph break indices.
        nlp (spacy.lang.en.English): The spaCy NLP model.

    Returns:
        Tuple[List[Tuple[str, List[str]]], List[int]]:
            - Updated list of atomic facts.
            - Updated list of paragraph break indices.
    """
    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts: List[List[Sequence[str]]] = []
    new_atomic_facts: List[Tuple[str, List[str]]] = []
    new_para_breaks: List[int] = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split())==1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        # print (entities)
        new_facts = []
        for i, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any([fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for j, other_fact in enumerate(facts) if j != i]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities==covered_entities
        except Exception:
            new_facts = facts # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s: str) -> bool:
    """Check if the given string represents an integer.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is an integer, False otherwise.
    """
    try:
        int(s)
        return True
    except Exception:
        return False

def detect_initials(text: str) -> List[str]:
    """Detect initials in the text.

    Args:
        text (str): The input text.

    Returns:
        List[str]: List of detected initials.
    """
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences: List[str], initials: List[str]) -> List[str]:
    """Fix sentence splitting issues caused by initials.

    Args:
        curr_sentences (List[str]): List of current sentences.
        initials (List[str]): List of detected initials.

    Returns:
        List[str]: Corrected list of sentences.
    """
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences

def main() -> None:
    """Main function to demonstrate the usage of AtomicFactGenerator."""
    generator = AtomicFactGenerator("api.key", "demos", gpt3_cache_file=None)
    atomic_facts, para_breaks = generator.run("Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. He has been named Arsenal F.C's greatest ever player.\n\nHenry made his professional debut with Monaco in 1994 before signing for defending Serie A champions Juventus. However, limited playing time, coupled with disagreements with the club's hierarchy, led to him signing for Premier League club Arsenal for £11 million in 1999.")

    print(atomic_facts)
    print(para_breaks)

if __name__ == "__main__":
    main()
