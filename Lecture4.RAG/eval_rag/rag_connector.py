# rag_connector.py
import requests
import time
from typing import List, Dict, Any


class RAGConnector:
    """Simple connector to the RAG system"""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str = None,
        timeout: int = 30
    ):

        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout

    def query(self, question: str) -> Dict[str, Any]:

        try:
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['X-API-Key'] = self.api_key

            response = requests.post(
                self.endpoint_url,
                json={"message": question},
                headers=headers,
                timeout=self.timeout
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            print(f"❌ Error: {e}")
            return {'error': str(e)}

    def batch_query(
        self,
        questions: List[str],
        expected_answers: List[str] = None
    ) -> List[Dict[str, Any]]:

        results = []

        for i, question in enumerate(questions):
            print(f"📝 {i+1}/{len(questions)}: {question[:50]}...")

            rag_response = self.query(question)

            result = {
                'question': question,
                'answer': rag_response.get('content', ''),
                'contexts': [s.get('content', '') for s in rag_response.get('sources', [])],
                'ground_truth': expected_answers[i] if expected_answers else ''
            }

            results.append(result)
            time.sleep(0.1)  # Pause between queries

        print(f"✅ Processed {len(results)} questions")
        return results


# # ==================== USAGE EXAMPLE ====================

# if __name__ == "__main__":

#     connector = RAGConnector(
#         endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
#         api_key="rag-api-key"
#     )

#     questions = [
#         'What methods are used in product_service?',
#         'How does the authorization process work?'
#     ]

#     expected = [
#         'Business representatives, developers and testers participate in the testing process.',
#         'The developer provides assistance in passing test scenarios and advises on data preparation.'
#     ]

#     results = connector.batch_query(questions, expected)

#     print("\n=== Results ===")
#     for res in results:
#         print(f"\nQuestion: {res['question']}")
#         print(f"RAG Answer: {res['answer']}")
#         print(f"Expected answer: {res['ground_truth']}")
#         print(f"Contexts: {res['contexts']}")
