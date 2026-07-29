# ai_client.py
from openai import OpenAI
from google import genai
from google.genai import types

from config import open_ai_api_key, google_api_key

client = OpenAI(api_key=open_ai_api_key)

client_google = genai.Client(api_key=google_api_key)


def google_chat(prompt: str) -> str:
    response = client_google.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    answer = response.text
    print(answer)
    return answer


def google_chat_v2(prompt: str) -> str:
    response = client_google.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )

    answer = response.text
    print(answer)
    return answer


def google_chat_v3(prompt: str, model: str, temperature: float) -> str:
    response = client_google.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )

    answer = response.text
    print(answer)
    return answer


def openai_chat(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-5.4-mini-2026-03-17",
        input=prompt,
    )

    answer = response.output_text
    print(answer)
    return answer


def openai_chat_v2(prompt: str, model: str = "gpt-5.4-mini-2026-03-17") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    print(answer)
    return answer

# Examples of how to use the functions


# translation = openai_chat_v2("Translate to French: Hello world")
# translation = google_chat("Translate to French: Hello world")
# # Result: "Bonjour le monde"

# # Summarization
# summary = openai_chat(
#     "Summarize: [long text about climate change...]")
# # Result: a short summary

# # Code generation
# code = openai_chat("Write a Python function to sort a list")
# # Result: working Python code

# # Creative writing
# story = openai_chat("Write a short story about a robot")
# # Result: an original short story
