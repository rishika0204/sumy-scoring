import string
from typing import Set


def clean_text(text: str) -> str:
    """
    Normalize input text by lowercasing and removing non-printable characters.

    Args:
        text (str): Raw text to clean.

    Returns:
        str: Cleaned text containing only printable characters.
    """
    # Early return for non-str inputs
    if not isinstance(text, str):
        return ""

    # Lowercase input
    text = text.lower()

    # Build printable character set
    printable: Set[str] = set(string.printable)

    # Filter characters by printable set
    return ''.join(ch for ch in text if ch in printable)
