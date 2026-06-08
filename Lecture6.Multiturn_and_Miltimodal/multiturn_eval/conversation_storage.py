"""
Storing and managing conversations in JSON format
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class ConversationStorage:
    """Class for saving and loading conversations"""

    def __init__(self, file_path: str = "data/conversations.json"):
        # Get absolute path relative to current file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(base_dir, file_path)

        # Creating directory if does not exist
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Initializing file if does not exist
        if not os.path.exists(self.file_path):
            self._save_data({"conversations": []})

    def _load_data(self) -> Dict:
        """Loads data from JSON file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"conversations": []}

    def _save_data(self, data: Dict) -> None:
        """Saves data to JSON file"""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _generate_id(self) -> str:
        """Generates unique ID for conversation"""
        data = self._load_data()
        existing_ids = [c.get('id', '') for c in data.get('conversations', [])]

        # Find maximum number
        max_num = 0
        for conv_id in existing_ids:
            if conv_id.startswith('conv_'):
                try:
                    num = int(conv_id.split('_')[1])
                    max_num = max(max_num, num)
                except (IndexError, ValueError):
                    pass

        return f"conv_{max_num + 1:03d}"

    def save_conversation(self, turns: List[Dict]) -> str:
        """
        Saves conversation
        """
        if not turns:
            return ""

        conv_id = self._generate_id()
        conversation = {
            "id": conv_id,
            "timestamp": datetime.now().isoformat(),
            "turns_count": len(turns),
            "turns": turns
        }

        data = self._load_data()
        data["conversations"].append(conversation)
        self._save_data(data)

        return conv_id

    def load_all(self) -> List[Dict]:
        """Loads all saved conversations"""
        data = self._load_data()
        return data.get("conversations", [])

    def load_by_id(self, conv_id: str) -> Optional[Dict]:
        """Loads conversation by ID"""
        data = self._load_data()
        for conv in data.get("conversations", []):
            if conv.get("id") == conv_id:
                return conv
        return None

    def count(self) -> int:
        """Returns count of saved conversations"""
        data = self._load_data()
        return len(data.get("conversations", []))

    def list_conversations(self) -> None:
        """Prints list of all conversations to console"""
        conversations = self.load_all()

        if not conversations:
            print("No saved conversations")
            return

        print(f"\n{'='*60}")
        print(f"Saved conversations ({len(conversations)}):")
        print(f"{'='*60}")

        for conv in conversations:
            conv_id = conv.get('id', 'N/A')
            timestamp = conv.get('timestamp', 'N/A')
            turns_count = conv.get('turns_count', len(conv.get('turns', [])))

            # Get first user message for preview
            first_user_msg = ""
            for turn in conv.get('turns', []):
                if turn.get('role') == 'user':
                    first_user_msg = turn.get('content', '')[:50]
                    if len(turn.get('content', '')) > 50:
                        first_user_msg += "..."
                    break

            print(f"\n[{conv_id}] {timestamp}")
            print(f"   Turns: {turns_count}")
            print(f"   Preview: \"{first_user_msg}\"")

        print(f"\n{'='*60}")

    def clear(self) -> None:
        """Deletes all conversations"""
        self._save_data({"conversations": []})

    def delete(self, conv_id: str) -> bool:
        """
        Deletes conversation by ID
        """
        data = self._load_data()
        conversations = data.get("conversations", [])

        for i, conv in enumerate(conversations):
            if conv.get("id") == conv_id:
                conversations.pop(i)
                data["conversations"] = conversations
                self._save_data(data)
                return True

        return False

    def get_conversation_preview(self, conv: Dict, max_length: int = 100) -> str:
        """Returns conversation preview"""
        turns = conv.get('turns', [])
        if not turns:
            return "Empty conversation"

        preview_parts = []
        for turn in turns[:4]:  # First 4 turns
            role = "U" if turn.get('role') == 'user' else "A"
            content = turn.get('content', '')[:30]
            preview_parts.append(f"{role}: {content}")

        preview = " | ".join(preview_parts)
        if len(preview) > max_length:
            preview = preview[:max_length] + "..."

        return preview
