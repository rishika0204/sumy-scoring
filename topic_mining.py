import re
import string
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from keyword_extraction import find_keywords


def clean_text(text: str) -> str:
    """
    Normalize input text by lowercasing and removing non-printable characters.

    Args:
        text (str): Raw text to clean.

    Returns:
        str: Cleaned text with only printable characters.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    printable = set(string.printable)
    # Filter out non-printable chars
    return ''.join(ch for ch in text if ch in printable)


def topic_modeling_with_keywords_lda_v2(
    train_docs: List[str],
    n_topics: int = 3,
    top_n_docs: int = 5,
    max_keywords: int = 10
) -> List[str]:
    """
    Fit an LDA model and extract keywords for each topic via PageRank.

    Steps:
      1. Vectorize documents to term-frequency.
      2. Fit LDA and compute topic distributions.
      3. For each topic:
         a. Select top-n documents by topic probability.
         b. Aggregate their text.
         c. Extract top keywords using find_keywords().

    Args:
        train_docs (List[str]): List of document strings.
        n_topics (int): Number of LDA topics.
        top_n_docs (int): Documents per topic to aggregate.
        max_keywords (int): Max keywords to extract per topic.

    Returns:
        List[str]: Keywords string for each topic.
    """
    # Early return if no documents
    if not train_docs:
        return []

    # Dynamic df thresholds
    min_df = 1
    max_df = 1.0 if len(train_docs) == 1 else 0.95

    # 1. Vectorize to term-frequency
    vectorizer = CountVectorizer(
        stop_words='english',
        min_df=min_df,
        max_df=max_df
    )
    doc_term_matrix = vectorizer.fit_transform(train_docs)

    # 2. Fit LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42
    )
    lda.fit(doc_term_matrix)

    # Topic distribution per document
    topic_probs = lda.transform(doc_term_matrix)

    topic_keywords_list: List[str] = []
    # 3. Extract keywords per topic
    for topic_idx in range(n_topics):
        # a. Top documents for this topic
        sorted_docs = np.argsort(topic_probs[:, topic_idx])[::-1]
        selected = sorted_docs[:top_n_docs]

        # b. Aggregate text
        agg_text = ' '.join(train_docs[i] for i in selected)
        clean_agg = clean_text(agg_text)
        if not clean_agg:
            topic_keywords_list.append('')
            continue

        # c. Keyword extraction via PageRank-based find_keywords()
        keywords = find_keywords(clean_agg, num_keywords=max_keywords)
        topic_keywords_list.append(keywords)

    return topic_keywords_list


def split_document_into_chunks_v2(
    document: str,
    chunk_size: int = 100
) -> List[str]:
    """
    Split a document into word-based chunks of fixed size.

    Args:
        document (str): Input text document.
        chunk_size (int): Number of words per chunk.

    Returns:
        List[str]: List of text chunks.
    """
    if not isinstance(document, str) or not document.strip():
        return []
    words = document.split()
    # Use list comprehension for chunking
    return [
        ' '.join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
