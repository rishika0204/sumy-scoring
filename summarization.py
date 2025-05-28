import logging
import re
from typing import List, Tuple

import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from scipy.sparse.linalg import svds
from concurrent.futures import ThreadPoolExecutor

from utils.preprocess_summary_v13 import preprocess_text


def summarize_paragraph_v3(
    paragraph: str,
    keywords: List[str] = [],
    sentence_count: int = 5
) -> Tuple[str, List[Tuple[int, float]]]:
    """
    Generate a concise summary of a paragraph using LSA and optional keyword boosting.

    Steps:
      1. Preprocess input text (clean, normalize).
      2. Parse and tokenize into sentences.
      3. Build term-document matrix and apply LSA.
      4. Optionally boost sentence scores if keywords are present.
      5. Return top `sentence_count` sentences and their boosted scores.

    Args:
        paragraph (str): Text to summarize.
        keywords (List[str]): Keywords to boost relevance.
        sentence_count (int): Number of sentences in output summary.

    Returns:
        Tuple[str, List[Tuple[int, float]]]: Summary string and list of (sentence_index, score).
    """
    # 1. Preprocess and validate input
    text = preprocess_text(paragraph)
    if not isinstance(text, str) or not text.strip():
        logging.warning("Empty or invalid paragraph input.")
        return "", []  # early return for invalid input

    # 2. Parse into sentences
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    document = parser.document
    if not document.sentences:
        return "", []  # no sentences to summarize

    # 3. Build dictionary and term-document matrix
    dictionary = summarizer._create_dictionary(document)
    if not dictionary:
        logging.info("No terms found in document for summarization.")
        return "", []

    term_matrix = summarizer._create_matrix(document, dictionary)
    freq_matrix = summarizer._compute_term_frequency(term_matrix)

    # Ensure we choose a valid rank k
    max_k = min(freq_matrix.shape)
    if max_k <= 1:
        logging.info("Term-document matrix too small for SVD.")
        return "", []
    k = min(5, max_k - 1)

    # 4. Apply SVD for LSA
    u, sigma, vt = svds(freq_matrix, k=k)
    sentence_scores = summarizer._compute_ranks(sigma, vt)

    # Prepare keyword set for boosting
    keyword_set = {kw.lower() for kw in keywords}

    # 5. Boost scores for sentences containing keywords
    def boost_score(idx_sent: Tuple[int, str]) -> Tuple[int, float]:
        idx, sent = idx_sent
        sent_lower = str(sent).lower()
        boost = sum(1 for kw in keyword_set if kw in sent_lower)
        return idx, sentence_scores[idx] + boost

    # Use multithreading to compute boosted scores
    boosted = list(
        ThreadPoolExecutor().map(
            boost_score,
            enumerate(document.sentences)
        )
    )

    # Sort by score descending and pick top sentences
    boosted.sort(key=lambda x: x[1], reverse=True)
    top_idxs = [idx for idx, _ in boosted[:sentence_count]]
    summary_sentences = [str(document.sentences[i]) for i in top_idxs]
    summary_text = " ".join(summary_sentences)

    return summary_text, boosted
