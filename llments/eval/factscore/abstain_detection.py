"""Abstain Detection Module."""
import numpy as np
import re
from typing import List

invalid_ppl_mentions: List[str] = [
    "I could not find any information",
    "The search results do not provide",
    "There is no information",
    "There are no search results",
    "there are no provided search results",
    "not provided in the search results",
    "is not mentioned in the provided search results",
    "There seems to be a mistake in the question",
    "Not sources found",
    "No sources found",
    "Try a more general question"
]

def remove_citation(text: str) -> str:
    """Remove citation references and fix specific starting phrases in the text.

    Args:
        text (str): The input text from which citations are to be removed.

    Returns:
        str: The text after removing citation references and making necessary replacements.
    """
    text = re.sub(r"\s*\[\d+\]\s*","", text)
    if text.startswith("According to , "):
        text = text.replace("According to , ", "According to the search results, ")
    return text

def is_invalid_ppl(text: str) -> bool:
    """Check if the text starts with any invalid phrases indicating insufficient information.

    Args:
        text (str): The input text to be checked.

    Returns:
        bool: True if the text starts with any invalid phrase, False otherwise.
    """
    return np.any([text.lower().startswith(mention.lower()) for mention in invalid_ppl_mentions])

def is_invalid_paragraph_ppl(text: str) -> bool:
    """Determine if a paragraph is invalid based on its content.

    A paragraph is considered invalid if it is empty or contains any invalid phrases.

    Args:
        text (str): The paragraph text to be evaluated.

    Returns:
        bool: True if the paragraph is invalid, False otherwise.
    """
    return len(text.strip())==0 or np.any([mention.lower() in text.lower() for mention in invalid_ppl_mentions])

def perplexity_ai_abstain_detect(generation: str) -> bool:
    """Detect if the AI generation should abstain based on perplexity analysis.

    This function removes citations from the generation, checks if it starts with any invalid phrases,
    and verifies that all paragraphs contain valid information.

    Args:
        generation (str): The generated text to be analyzed.

    Returns:
        bool: True if the generation should abstain, False otherwise.
    """
    output = remove_citation(generation)
    if is_invalid_ppl(output):
        return True
    valid_paras = []
    for para in output.split("\n\n"):
        if is_invalid_paragraph_ppl(para):
            break
        valid_paras.append(para.strip())

    if len(valid_paras) == 0:
        return True
    else:
        return False

def generic_abstain_detect(generation: str) -> bool:
    """Detect if the generation should abstain based on generic abstain phrases.

    Args:
        generation (str): The generated text to be analyzed.

    Returns:
        bool: True if the generation contains generic abstain phrases, False otherwise.
    """
    return generation.startswith("I'm sorry") or "provide more" in generation

def is_response_abstained(generation: str, fn_type: str) -> bool:
    """Determine if the response should be abstained based on the specified detection function type.

    Args:
        generation (str): The generated text to be analyzed.
        fn_type (str): The type of detection function to use ('perplexity_ai' or 'generic').

    Returns:
        bool: True if the response should abstain based on the detection function, False otherwise.
    """
    if fn_type == "perplexity_ai":
        return perplexity_ai_abstain_detect(generation)

    elif fn_type == "generic":
        return generic_abstain_detect(generation)

    else:
        return False

