
def score_variance(scores):
    if len(scores) == 0:
        return 0.0

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)

    return variance
