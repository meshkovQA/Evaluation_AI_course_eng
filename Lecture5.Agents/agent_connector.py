# agent_connector.py
import requests
import time
import uuid
from typing import List, Dict, Any, Optional


class AgentConnector:
    """Connector to AI agent with tool support"""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        user_id: str = "AleksM",
        session_id: Optional[str] = None,
        timeout: int = 120
    ):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.user_id = user_id
        self.session_id = session_id or self._generate_session_id()
        self.timeout = timeout

    def _generate_session_id(self) -> str:
        """Generates UUID for session"""
        return str(uuid.uuid4())

    def _build_headers(self) -> Dict[str, str]:
        """Builds request headers"""
        return {
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'X-Session-Id': self.session_id,
            'X-User-Id': self.user_id,
            'X-API-Key': self.api_key
        }

    def query(
        self,
        question: str,
        urls: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Sends query to agent

        Returns:
            Dict with fields:
            - answer: agent response text
            - tools_used: list of tools used
            - raw_response: full API response
        """
        try:
            # Building body
            body = {"question": question}
            if urls:
                body["urls"] = urls

            # Sending query
            response = requests.post(
                self.endpoint_url,
                json=body,
                headers=self._build_headers(),
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            # Extracting relevant data
            result = {
                'output': data.get('output', ''),
                'tools_used': data.get('tools_used', []),
                'raw_response': data
            }

            return result

        except requests.exceptions.Timeout:
            return {
                'error': 'Request timeout',
                'output': '',
                'tools_used': []
            }
        except requests.exceptions.RequestException as e:
            return {
                'error': f'Request failed: {str(e)}',
                'output': '',
                'tools_used': []
            }
        except Exception as e:
            return {
                'error': f'Unexpected error: {str(e)}',
                'output': '',
                'tools_used': []
            }

    def batch_query(
        self,
        questions: List[str],
        urls_list: Optional[List[List[str]]] = None,
        expected_answers: Optional[List[str]] = None,
        delay: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Batch processing of questions

        Args:
            questions: list of questions
            urls_list: list of URL lists for each question (optional)
            expected_answers: expected answers for evaluation (optional)
            delay: delay between requests in seconds
        """
        results = []

        for i, question in enumerate(questions):
            print(f"📝 {i+1}/{len(questions)}: {question[:60]}...")

            # Getting URLs for current question
            urls = urls_list[i] if urls_list and i < len(urls_list) else None

            # Query to agent
            agent_response = self.query(question, urls)

            # Building result
            result = {
                'question': question,
                'output': agent_response.get('output', ''),
                'tools_used': agent_response.get('tools_used', []),
                'urls': urls,
                'error': agent_response.get('error')
            }

            # Print information about tools used
            if result['tools_used']:
                tools_str = ', '.join(result['tools_used'])
                print(f"   🔧 Tools: {tools_str}")

            results.append(result)

            # Pause between requests
            if i < len(questions) - 1:
                time.sleep(delay)

        # Statistics
        successful = sum(1 for r in results if not r.get('error'))
        print(f"\n✅ Successfully: {successful}/{len(results)}")

        # Tool usage statistics
        all_tools = []
        for r in results:
            all_tools.extend(r.get('tools_used', []))
        if all_tools:
            unique_tools = set(all_tools)
            print(
                f"🔧 Tools used: {len(all_tools)} (unique: {len(unique_tools)})")
            print(f"   List: {', '.join(unique_tools)}")

        return results

    def new_session(self):
        """Creates a new session"""
        self.session_id = self._generate_session_id()
        print(f"🔄 New session: {self.session_id}")
