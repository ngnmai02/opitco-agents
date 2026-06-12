
def persona_score_gap(results):
    persona_scores = {}

    for r in results:
        persona = r["persona"]
        score = r["score"]

        if persona not in persona_scores:
            persona_scores[persona] = []

        persona_scores[persona].append(score)

    means = [sum(v)/len(v) for v in persona_scores.values()]

    return max(means) - min(means)
