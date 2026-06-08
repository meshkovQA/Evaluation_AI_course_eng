import requests


def call_api(prompt, options=None, context=None):
    response = requests.post(
        "http://5.11.83.110:8002/api/v1/chat/",
        json={"message": prompt},
        headers={'X-API-Key': 'rag-api-key'}
    )
    data = response.json()

    contexts = [source.get('content', '')
                for source in data.get('sources', [])]

    # IMPORTANT: Return an object, not a string
    return {
        "output": {
            "answer": data.get('content', ''),
            "contexts": contexts  # Contexts inside output
        }
        # "output": data.get('content', '')
    }
