from collections import Counter
import math

def n_gram_precision(reference, hypothesis, n):
    ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
    hyp_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)])

    matches = sum((hyp_ngrams & ref_ngrams).values())
    total_hyp_ngrams = sum(hyp_ngrams.values())

    return matches / total_hyp_ngrams if total_hyp_ngrams > 0 else 0

def brevity_penalty(reference, hypothesis):
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if hyp_len > ref_len:
        return 1
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - ref_len / hyp_len)

def bleu(reference, hypothesis, max_n=4):
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    precisions = [n_gram_precision(ref_words, hyp_words, n) for n in range(1, max_n + 1)]

    if all(precision == 0 for precision in precisions):
        precision_mean = 0
    else:
        precision_mean = math.exp(sum(math.log(p) for p in precisions if p > 0) / max_n)

    bp = brevity_penalty(ref_words, hyp_words)
    bleu_score = bp * precision_mean

    return bleu_score
