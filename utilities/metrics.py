import numpy as np

def calculate_character_error_rate(gt_texts: list[str], pred_texts: list[str]) -> float:
    """Calculate Character Error Rate (CER) between ground truth and predicted texts."""

    assert len(gt_texts) == len(pred_texts), "Ground truth and predicted texts lists must have the same length."

    total_chars = 0
    total_errors = 0
    for gt, pred in zip(gt_texts, pred_texts):
        total_chars += len(gt)
        total_errors += calculate_levenshtein_distance(gt, pred)
    cer =  total_errors/ total_chars if total_chars > 0 else 0.0
    return cer

def calculate_levenshtein_distance(s1: str, s2: str) -> int:
    # ensure s1 is the longer string
    if len(s1) < len(s2):
        s1 , s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    # initialize matrix
    levenshtein_table = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    for i in range(len(s1) + 1):
        levenshtein_table[i][0] = i 
    for j in range(len(s2) + 1):
        levenshtein_table[0][j] = j
    # calculate distances
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            subst_cost = 0 if s1[i - 1] == s2[j - 1] else 1

            levenshtein_table[i][j] = min(
                levenshtein_table[i - 1][j] + 1,      # deletion
                levenshtein_table[i][j - 1] + 1,      # insertion
                levenshtein_table[i - 1][j - 1] + subst_cost # substitution
            )
    return levenshtein_table[len(s1)][len(s2)]