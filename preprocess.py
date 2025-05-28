import re
import spacy  # Loaded here in case downstream processing needs linguistic features
from nltk.tokenize import sent_tokenize

def preprocess_text(text: str) -> str:
    """
    Clean and normalize a raw summary string by stripping out list markers,
    parenthetical asides, attachment references, and excessive whitespace,
    and by removing all-caps headings.

    This is a two-step pipeline:
      1. Strip out bullets, numbered/lettered lists, special markers, parentheses,
         single-letter abbreviations, and attachment notices.
      2. Remove lines or phrases that look like all-caps headings.

    Args:
        text (str): Raw input text to be preprocessed.

    Returns:
        str: The cleaned, single-space-normalized text.
    """
    # If input isn't a string, return empty string
    if not isinstance(text, str):
        return ""

    # -------------------------------------------------------------------------
    # STEP 1: Remove bullets, list markers, parentheses, and attachment references
    # -------------------------------------------------------------------------

    # 1a) Remove common bullet characters at the start of any line (e.g., "*", "•", "-", "-", "·")
    text = re.sub(
        r'^\s*[\*\•\-\u2022]',
        '',
        text,
        flags=re.MULTILINE
    )

    # 1b) Remove any stray bullet characters anywhere in the text
    text = re.sub(
        r'[\*\|\-]',
        '',
        text
    )

    # 1c) Remove numbered list prefixes, e.g., "1. ", "23) "
    text = re.sub(
        r'^\s*\d+[\.\)]\s*',
        '',
        text,
        flags=re.MULTILINE
    )

    # 1d) Remove lettered list prefixes, e.g., "a) ", "B] "
    text = re.sub(
        r'^\s*[A-Za-z][\.\)\]]\s*',
        '',
        text,
        flags=re.MULTILINE
    )

    # 1e) Strip out any text within parentheses, including the parentheses themselves
    text = re.sub(
        r'\([^)]*\)',
        '',
        text
    )

    # 1f) Remove any "See attachment" or "please refer to attachment" phrases (case-insensitive)
    text = re.sub(
        r'\b(See attachment|please refer to attachment)\b.*',
        '',
        text,
        flags=re.IGNORECASE
    )

    # 1g) Drop single-letter abbreviations followed by a period, e.g., "e.g.", but only single letters
    text = re.sub(
        r'\b\w\b\.',
        '',
        text
    )

    # 1h) Collapse any sequence of whitespace (spaces, tabs, newlines) into a single space
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    # -------------------------------------------------------------------------
    # STEP 2: Remove lines or segments that look like ALL-CAPS headings
    # -------------------------------------------------------------------------

    # 2a) Remove lines that are entirely uppercase words (with spaces), but only
    #     if the next character is Titlecase (to avoid removing genuine acronyms)
    text = re.sub(
        r'^[A-Z\s]+(?=[A-Z][a-z])\s*',
        '',
        text,
        flags=re.MULTILINE
    )

    # 2b) Remove any leftover line that has no lowercase or spaces (i.e., pure gibberish caps)
    text = re.sub(
        r'^[^A-Za-z\s]+$(?:\r?\n)?',
        '',
        text,
        flags=re.MULTILINE
    )

    # 2c) Strip out runs of two or more uppercase words immediately following punctuation
    text = re.sub(
        r'(?<=[:\.!?])\s*(?:[A-Z]+\s){2,}',
        '',
        text
    )

    return text
