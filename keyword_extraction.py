import re
import string
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import lil_matrix, csr_matrix
from concurrent.futures import ThreadPoolExecutor


def clean_text(text: str) -> str:
    """
    Normalize and clean input text by lowercasing and removing non-printable characters.

    Args:
        text (str): The raw text to clean.

    Returns:
        str: The cleaned, lowercase text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    printable = set(string.printable)
    # Remove any characters not in printable set
    return text.translate(
        str.maketrans(
            {ch: None for ch in text if ch not in printable}
        )
    )


def build_partial_matrix(
    processed_text: List[str],
    word_index: Dict[str, int],
    window_size: int
) -> lil_matrix:
    """
    Construct a co-occurrence matrix for a slice of tokens using a sliding window.

    Args:
        processed_text (List[str]): List of tokens (filtered and lemmatized).
        word_index (Dict[str, int]): Mapping from token to index in matrix.
        window_size (int): Number of tokens in each context window.

    Returns:
        lil_matrix: Sparse co-occurrence counts.
    """
    vocab_len = len(word_index)
    partial = lil_matrix((vocab_len, vocab_len), dtype=np.float32)

    for i in range(len(processed_text) - window_size):
        window = processed_text[i : i + window_size]
        for j, w1 in enumerate(window):
            idx1 = word_index[w1]
            for k in range(j + 1, window_size):
                idx2 = word_index[window[k]]
                if idx1 == idx2:
                    continue
                dist = max(abs(j - k), 1)
                # weight inversely by distance
                partial[idx1, idx2] += 1.0 / dist
                partial[idx2, idx1] += 1.0 / dist
    return partial


def find_keywords(
    text: str,
    row_index: int,
    df: pd.DataFrame,
    num_keywords: int = 5
) -> str:
    """
    Extract the top `num_keywords` keywords from a text using a TextRank-like graph algorithm.

    Steps:
        1. Clean, tokenize, POS-tag, and lemmatize the text.
        2. Filter tokens by POS and remove stopwords/punctuation.
        3. Build a co-occurrence graph via sliding window.
        4. Normalize and apply PageRank.
        5. Extract and score candidate phrases.

    Args:
        text (str): The input document/text.
        row_index (int): Index in `df` to update results.
        df (pd.DataFrame): DataFrame where keywords will be stored.
        num_keywords (int): Number of top keywords to return.

    Returns:
        str: Comma-separated top keywords.
    """
    start_time = time.time()

    # 1. Clean & tokenize
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    if not tokens:
        return ""

    # 2. POS tag + lemmatize
    pos1 = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    adjective_tags = {"JJ", "JJR", "JJS"}
    lemmatized = [
        lemmatizer.lemmatize(w, 'a') if t in adjective_tags else lemmatizer.lemmatize(w)
        for w, t in pos1
    ]
    pos2 = nltk.pos_tag(lemmatized)

    # Keep nouns, adjectives, gerunds, foreign words
    valid_tags = {"NN","NNS","NNP","NNPS","JJ","JJR","JJS","VBG","FW"}
    filtered = [w for w, t in pos2 if t in valid_tags]
    if not filtered:
        return ""

    # 3. Remove stopwords + punctuation
    extra_sw = []
    try:
        with open('long_stopwords.txt') as f:
            extra_sw = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        pass
    stopwords_set = set(string.punctuation) | set(extra_sw)
    processed = [w for w in filtered if w not in stopwords_set]
    if not processed:
        return ""

    # Build vocabulary
    vocab = list(set(processed))
    idx_map = {w: i for i, w in enumerate(vocab)}
    window_size = 3

    # 4. Build graph in parallel
    chunk_len = max(1, len(processed) // 8)
    chunks = [processed[i:i+chunk_len] for i in range(0, len(processed), chunk_len)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        partials = list(executor.map(lambda c: build_partial_matrix(c, idx_map, window_size), chunks))
    graph = sum(partials).tocsr()

    # Normalize rows to probabilities
    sums = np.array(graph.sum(axis=1)).flatten()
    sums[sums == 0] = 1e-9
    graph_csc = graph.tocsc()
    for j in range(len(vocab)):
        sp, ep = graph_csc.indptr[j], graph_csc.indptr[j+1]
        graph_csc.data[sp:ep] /= sums[j]
    trans = graph_csc.tocsr()

    # 5. PageRank
    damping = 0.85
    score = np.ones(len(vocab), dtype=np.float32)
    for _ in range(50):
        prev = score.copy()
        score = (1 - damping) + damping * trans.dot(prev)
        if np.abs(prev - score).sum() <= 1e-4:
            break

    # 6. Extract candidate phrases
    phrases: List[List[str]] = []
    cur: List[str] = []
    for w in lemmatized:
        if w in stopwords_set:
            if cur:
                phrases.append(cur)
                cur = []
        else:
            cur.append(w)
    if cur:
        phrases.append(cur)

    # Dedupe and remove single-word inside multi-word
    unique_phrases = []
    seen = set()
    for ph in phrases:
        tup = tuple(ph)
        if tup not in seen:
            seen.add(tup)
            unique_phrases.append(ph)
    for w in vocab:
        if [w] in unique_phrases:
            for ph in unique_phrases:
                if len(ph) > 1 and w in ph:
                    unique_phrases.remove([w])
                    break

    # Score and select top
    scored = [(sum(score[idx_map[w]] for w in ph if w in idx_map), ' '.join(ph)) for ph in unique_phrases]
    scored.sort(reverse=True)
    top_keywords = [kw for _, kw in scored[:num_keywords]]

    # Write back to DataFrame
    df.at[row_index, 'Keywords using Custom TextRank Algorithm'] = ', '.join(top_keywords)
    print(f"Row {row_index+1} updated with top {num_keywords} keywords: {', '.join(top_keywords)}")
    print(f"Elapsed time: {time.time() - start_time:.2f}s")

    return ', '.join(top_keywords)
