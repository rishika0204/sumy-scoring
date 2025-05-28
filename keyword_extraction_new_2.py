from scipy.sparse import lil_matrix, csr_matrix
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import time

def find_keywords(text: str,
                  row_index: int,
                  top_n: int,
                  window_size: int = 3,
                  num_threads: int = 8,
                  damping_factor: float = 0.85,
                  max_iterations: int = 50) -> str:
    """
    Extracts the top_n keywords from `text` using a custom TextRank-like algorithm.

    Args:
        text (str): The raw text to process.
        row_index (int): An identifier for logging purposes (no longer used to update a DataFrame).
        top_n (int): Number of keywords to return.
        window_size (int): Size of the sliding window for co-occurrence.
        num_threads (int): How many threads to use when building the co-occurrence matrix.
        damping_factor (float): PageRank damping factor.
        max_iterations (int): Maximum PageRank iterations.

    Returns:
        str: Comma-separated top_n keywords.
    """
    start_time = time.time()

    # --- 1) Clean & tokenize ---
    cleaned_text = clean_text(text)
    tokens = word_tokenize(cleaned_text)
    if not tokens:
        return ""

    # --- 2) POS-tag & lemmatize ---
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    # keep only nouns/adjectives
    keep_tags = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}
    filtered = [
        (word.lower(), tag) 
        for (word, tag) in pos_tags 
        if tag in keep_tags
    ]
    # load extra stopwords if available
    punctuation = set(string.punctuation)
    additional = []
    try:
        with open("long_stopwords.txt") as f:
            additional = [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        pass
    stopwords_plus = set(w for w, _ in filtered) | punctuation | set(additional)
    processed = [
        lemmatizer.lemmatize(w) 
        for (w, _) in filtered 
        if w not in stopwords_plus
    ]
    if not processed:
        return ""

    # --- 3) Build vocabulary & word index ---
    vocabulary = list(dict.fromkeys(processed))
    vocab_len   = len(vocabulary)
    word_index  = {w: i for i, w in enumerate(vocabulary)}

    # --- 4) Split into chunks & build co-occurrence matrices in parallel ---
    chunk_size = max(1, len(processed) // num_threads)
    chunks = [
        processed[i:i + chunk_size] 
        for i in range(0, len(processed), chunk_size)
    ]
    with ThreadPoolExecutor(max_workers=num_threads) as exec:
        partials = list(exec.map(
            lambda c: build_partial_matrix(c, word_index, window_size),
            chunks
        ))
    weighted_edge = sum(partials).tocsr()

    # --- 5) Normalize to transition matrix ---
    col_sums = np.array(weighted_edge.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1e-9
    T = weighted_edge.copy().tocsc()
    for j in range(vocab_len):
        start, end = T.indptr[j], T.indptr[j+1]
        T.data[start:end] /= col_sums[j]
    T = T.tocsr()

    # --- 6) Run PageRank iterations ---
    scores = np.ones(vocab_len, dtype=float) / vocab_len
    for _ in range(max_iterations):
        new_scores = (1 - damping_factor) / vocab_len + damping_factor * (T @ scores)
        if np.allclose(new_scores, scores, atol=1e-6):
            break
        scores = new_scores

    # --- 7) Score all candidate phrases (unigrams & multi-word) ---
    # For simplicity, we only extract single words here; extend as needed.
    sorted_idx   = np.argsort(scores)[::-1]
    n_return     = min(top_n, len(sorted_idx))
    keywords     = [vocabulary[i] for i in sorted_idx[:n_return]]

    end_time = time.time()
    print(f"[Row {row_index}] Matrix build: {end_time - start_time:.2f}s, "
          f"PageRank & scoring: {end_time - start_time:.2f}s")

    return ", ".join(keywords)
