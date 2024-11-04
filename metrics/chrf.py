from collections import Counter

def ngram_counts(text: str, n: int) -> Counter:
    """Generate n-gram counts for a given string and n-gram size."""
    return Counter([text[i:i+n] for i in range(len(text) - n + 1)])

def chrf(s1: str, s2: str, n: int = 6) -> float:
    """Calculate the CHR-F score for two strings."""
    ngrams_s1 = ngram_counts(s1, n)
    ngrams_s2 = ngram_counts(s2, n)

    overlap = sum((ngrams_s1 & ngrams_s2).values())

    precision = overlap / sum(ngrams_s1.values()) if ngrams_s1 else 0
    recall = overlap / sum(ngrams_s2.values()) if ngrams_s2 else 0

    if precision + recall == 0:
        return 0.0

    chrf_score = 2 * (precision * recall) / (precision + recall)
    return chrf_score
