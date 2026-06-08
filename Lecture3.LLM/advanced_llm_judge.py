from ai_client import openai_chat_v2


def advanced_llm_judge(question, answer):
    """
    Advanced evaluation that decomposes the answer into individual claims.
    """

    # Step 1: break the answer into claims
    decompose_prompt = f"""
Break this answer into individual verifiable claims:
{answer}

Return a list of claims, one per line.
"""

    claims_text = openai_chat_v2(decompose_prompt)
    claims = [claim.strip()
              for claim in claims_text.split('\n') if claim.strip()]

    # Step 2: evaluate each claim
    scores = []
    verdict_values = {"fully": 1.0, "mostly": 0.9,
                      "partial": 0.6, "minor": 0.3, "none": 0.0}

    for claim in claims:
        evaluate_prompt = f"""
Evaluate this claim with respect to the question "{question}":
Claim: {claim}

Pick one option:
- fully: fully answers the question
- mostly: mostly answers it
- partial: partially relevant
- minor: weakly related
- none: not relevant

Return a single word: fully/mostly/partial/minor/none
"""

        verdict = openai_chat_v2(evaluate_prompt).strip().lower()
        score = verdict_values.get(verdict, 0.6)
        scores.append(score)

    # Step 3: compute the final score
    final_score = sum(scores) / len(scores) if scores else 0
    final_score_10 = round(final_score * 10, 1)

    return {
        "score": final_score_10,
        "claims": claims,
        "individual_scores": scores,
        "verdict_distribution": {v: scores.count(s) for v, s in verdict_values.items()}
    }


# Example usage
question = "How does photosynthesis work?"
answer = "Photosynthesis is the process by which plants use light to produce glucose from CO2 and water. It happens in chloroplasts. Oxygen is released."

result = advanced_llm_judge(question, answer)
print(f"Advanced score: {result['score']}/10")
print(f"Number of claims: {len(result['claims'])}")
print(f"Distribution: {result['verdict_distribution']}")
