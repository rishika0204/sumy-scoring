import string
import time
from typing import List, Dict

import numpy as np
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import lil_matrix
from concurrent.futures import ThreadPoolExecutor

from utils.cleaning_utils import clean_text


def build_partial_matrix(
    tokens: List[str],
    word_index: Dict[str, int],
    window_size: int
) -> lil_matrix:
    """
    Build a sparse co-occurrence matrix for a list of tokens using a sliding window.

    Args:
        tokens (List[str]): Preprocessed and filtered tokens.
        word_index (Dict[str, int]): Map token to index in matrix.
        window_size (int): Context window size.

    Returns:
        lil_matrix: Weighted co-occurrence counts.
    """
    vocab_len = len(word_index)
    matrix = lil_matrix((vocab_len, vocab_len), dtype=np.float32)

    # Slide window over tokens
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i : i + window_size]
        for j, w1 in enumerate(window):
            idx1 = word_index[w1]
            for k in range(j + 1, window_size):
                idx2 = word_index[window[k]]
                if idx1 == idx2:
                    continue
                dist = max(abs(j - k), 1)
                # Update both directions
                matrix[idx1, idx2] += 1.0 / dist
                matrix[idx2, idx1] += 1.0 / dist
    return matrix


def find_keywords(
    text: str,
    num_keywords: int = 5
) -> List[str]:
    """
    Extract top keyword phrases from text using a TextRank-inspired algorithm.

    Steps:
      1. Clean and tokenize text.
      2. POS-tag and lemmatize.
      3. Filter tokens by POS and remove stopwords.
      4. Build co-occurrence graph in parallel.
      5. Normalize to probabilities and run PageRank.
      6. Assemble candidate phrases and score them.

    Args:
        text (str): Raw document text.
        num_keywords (int): Maximum number of keywords to return.

    Returns:
        List[str]: Selected keyword phrases.
    """
    start_time = time.time()

    # 1. Clean & tokenize
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    if not tokens:
        return []

    # 2. POS tagging and lemmatization
    pos_tags = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    adj_tags = {"JJ", "JJR", "JJS"}
    lemmas = [
        lemmatizer.lemmatize(w, 'a') if t in adj_tags else lemmatizer.lemmatize(w)
        for w, t in pos_tags
    ]
    pos2 = nltk.pos_tag(lemmas)

    # 3. Keep nouns, adjectives, gerunds, foreign words
    valid_pos = {"NN","NNS","NNP","NNPS","JJ","JJR","JJS","VBG","FW"}
    filtered = [w for w, t in pos2 if t in valid_pos]
    if not filtered:
        return []

    # Remove punctuation and extra stopwords
    extra_sw = []
    try:
        with open('long_stopwords.txt') as f:
            extra_sw = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        pass
    stopwords = set(string.punctuation) | set(extra_sw)
    processed = [w for w in filtered if w not in stopwords]
    if not processed:
        return []

    # Build graph nodes
    vocab = list(set(processed))
    idx_map = {w:i for i,w in enumerate(vocab)}
    window_size = 3

    # 4. Build edge weights in parallel
    chunk_size = max(1, len(processed)//8)
    chunks = [processed[i:i+chunk_size] for i in range(0, len(processed), chunk_size)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        partials = ex.map(lambda c: build_partial_matrix(c, idx_map, window_size), chunks)
    graph = sum(partials).tocsr()

    # 5. Normalize rows and PageRank
    row_sums = np.array(graph.sum(axis=1)).flatten()
    row_sums[row_sums==0] = 1e-9
    csc = graph.tocsc()
    for j in range(len(vocab)):
        sp, ep = csc.indptr[j], csc.indptr[j+1]
        csc.data[sp:ep] /= row_sums[j]
    trans = csc.tocsr()

    damping = 0.85
    score = np.ones(len(vocab), dtype=np.float32)
    for _ in range(50):
        prev = score.copy()
        score = (1-damping) + damping * trans.dot(prev)
        if np.abs(prev-score).sum() <= 1e-4:
            break

    # 6. Candidate phrase assembly
    phrases = []
    current = []
    for w in lemmas:
        if w in stopwords:
            if current:
                phrases.append(current)
                current = []
        else:
            current.append(w)
    if current:
        phrases.append(current)

    # Deduplicate single-word inside multi-word
    unique = []
    seen = set()
    for ph in phrases:
        tup = tuple(ph)
        if tup not in seen:
            seen.add(tup)
            unique.append(ph)
    for w in vocab:
        if [w] in unique:
            for ph in unique:
                if len(ph)>1 and w in ph:
                    unique.remove([w])
                    break

    # 7. Score and pick top
    scored = [(sum(score[idx_map[w]] for w in ph if w in idx_map), ' '.join(ph)) for ph in unique]
    scored.sort(reverse=True)
    keywords = [kw for _, kw in scored[:num_keywords]]

    print(f"Extracted {len(keywords)} keywords in {time.time()-start_time:.2f}s")
    return keywords
