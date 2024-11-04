def iou(reference, hypothesis):
    ref_set = set(reference.split())
    hyp_set = set(hypothesis.split())
    intersection = len(ref_set & hyp_set)
    union = len(ref_set | hyp_set)
    return intersection / union if union != 0 else 0
