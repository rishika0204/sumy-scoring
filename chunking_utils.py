from typing import List


def split_document_into_chunks(
    document: str,
    chunk_size: int = 100
) -> List[str]:
    """
    Split a document into word-based chunks of fixed size.

    Args:
        document (str): The text to split into chunks.
        chunk_size (int): Number of words per chunk.

    Returns:
        List[str]: List of document chunks.
    """
    # Early return for invalid input
    if not isinstance(document, str) or not document.strip():
        return []

    words = document.split()
    # Use list comprehension to build chunks
    return [
        ' '.join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
