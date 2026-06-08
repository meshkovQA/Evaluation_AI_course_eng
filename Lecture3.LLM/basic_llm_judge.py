from ai_client import google_chat_v3


def basic_llm_judge(question, answer):
    """
    Simple LLM-as-a-judge evaluation of an answer.
    """
    prompt = f"""
Score the quality of the answer to the question from 1 to 10:

Question: {question}
Answer: {answer}

Criteria: accuracy, completeness, readability
Return only a number from 1 to 10 and a brief explanation.
"""

    response = google_chat_v3(
        prompt, model="gemini-2.0-flash", temperature=0.1)
    return response


# Example usage
question = "How does photosynthesis work?"
answer = "Photosynthesis is the process by which plants use light to produce glucose from CO2 and water."

result = basic_llm_judge(question, answer)
print(f"Basic score: {result}")
