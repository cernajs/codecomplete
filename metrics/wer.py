import numpy as np

def wer(reference, hypothesis):
    """
    calculates levenstein distance on word level
    """

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)

    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + 1  # Substitution
                )

    wer_score = dp[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer_score
