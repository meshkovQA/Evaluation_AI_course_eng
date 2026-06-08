# dataset_parser.py
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class DatasetParser:
    """Simple dataset parser for evaluation"""

    def __init__(self):
        self.required_columns = ['question']  # Only question is required
        self.optional_columns = ['expected_response', 'expected_tools']

    def load_dataset(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load dataset from Excel or CSV file"""
        try:
            # Determine file extension
            file_extension = Path(file_path).suffix.lower()

            # Loading file
            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
            else:
                print(f"Unsupported file format: {file_extension}")
                return None

            # Check for required columns
            missing_columns = [
                col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                print(f"Found columns: {list(df.columns)}")
                return None

            # Remove rows with empty values in required columns
            df_clean = df.dropna(subset=self.required_columns)

            print(f"Dataset loaded with {len(df_clean)} questions")

            # Show information about found optional columns
            found_optional = [
                col for col in self.optional_columns if col in df_clean.columns]
            if found_optional:
                print(f"Found optional columns: {found_optional}")

            return df_clean

        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def _parse_tools_field(self, tools_value) -> List[str]:
        """
        Parses tools field into a list

        Supported formats:
        - "tool1, tool2, tool3" (comma-separated)
        - "tool1; tool2; tool3" (semicolon-separated)
        - ["tool1", "tool2", "tool3"] (already a list)
        - NaN / None -> empty list
        """
        # If empty value
        if pd.isna(tools_value) or tools_value is None or tools_value == "":
            return []

        # If already a list
        if isinstance(tools_value, list):
            return [str(tool).strip() for tool in tools_value if str(tool).strip()]

        # If string
        tools_str = str(tools_value).strip()
        if not tools_str:
            return []

        # Try splitting by comma or semicolon
        if ',' in tools_str:
            tools = [t.strip() for t in tools_str.split(',')]
        elif ';' in tools_str:
            tools = [t.strip() for t in tools_str.split(';')]
        else:
            # Single tool
            tools = [tools_str]

        # Remove empty strings
        return [t for t in tools if t]

    def get_questions(self, df: pd.DataFrame) -> List[str]:
        """Get list of all questions"""
        return df['question'].tolist()

    def get_expected_responses(self, df: pd.DataFrame) -> List[str]:
        """Get list of expected answers, replacing nan with empty string"""
        if 'expected_response' not in df.columns:
            return [""] * len(df)
        return df['expected_response'].fillna("").astype(str).tolist()

    def get_expected_tools(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Get list of expected tools for each question

        Returns:
            List of tool lists (e.g., [["tool1", "tool2"], ["tool3"], []])
        """
        if 'expected_tools' not in df.columns:
            return [[] for _ in range(len(df))]

        return [self._parse_tools_field(tools) for tools in df['expected_tools']]

    def get_question_response_pairs(self, df: pd.DataFrame) -> List[Dict[str, any]]:
        """
        Get list of question-answer pairs with expected tools

        Returns:
            List of dicts with keys 'question', 'expected_response', 'expected_tools'
        """
        pairs = []
        expected_tools_list = self.get_expected_tools(df)

        for i, row in df.iterrows():
            expected_response = row.get('expected_response', '')
            if pd.isna(expected_response) or str(expected_response).strip().lower() in ['nan', 'none', '']:
                expected_response = None
            else:
                expected_response = str(expected_response).strip()
            pair = {
                'question': row['question'],
                'expected_response': expected_response,
                'expected_tools': expected_tools_list[i] if i < len(expected_tools_list) else []
            }
            pairs.append(pair)

        return pairs

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate dataset

        Returns:
            Dict with dataset information
        """
        info = {
            'total_rows': len(df),
            'valid_pairs': 0,
            'empty_questions': 0,
            'empty_responses': 0,
            'empty_tools': 0,
            'avg_question_length': 0,
            'avg_response_length': 0,
            'avg_tools_count': 0,
            'unique_tools': set(),
            'has_expected_response_column': 'expected_response' in df.columns,
            'has_expected_tools_column': 'expected_tools' in df.columns
        }

        valid_pairs = 0
        question_lengths = []
        response_lengths = []
        tools_counts = []

        expected_tools_list = self.get_expected_tools(df)

        for i, row in df.iterrows():
            question = str(row['question']).strip()

            if not question or question == 'nan':
                info['empty_questions'] += 1
                continue

            # Check expected_response if column exists
            if 'expected_response' in df.columns:
                response = str(row['expected_response']).strip()
                if not response or response == 'nan':
                    info['empty_responses'] += 1
                else:
                    response_lengths.append(len(response))

            # Check expected_tools
            tools = expected_tools_list[i] if i < len(
                expected_tools_list) else []
            if not tools:
                info['empty_tools'] += 1
            else:
                tools_counts.append(len(tools))
                info['unique_tools'].update(tools)

            valid_pairs += 1
            question_lengths.append(len(question))

        info['valid_pairs'] = valid_pairs
        if question_lengths:
            info['avg_question_length'] = sum(
                question_lengths) / len(question_lengths)
        if response_lengths:
            info['avg_response_length'] = sum(
                response_lengths) / len(response_lengths)
        if tools_counts:
            info['avg_tools_count'] = sum(tools_counts) / len(tools_counts)

        return info

    def preview_dataset(self, df: pd.DataFrame, n: int = 3):
        """Show first n examples from dataset"""
        print("\n=== Dataset Preview ===")
        expected_tools_list = self.get_expected_tools(df)

        for i, row in df.head(n).iterrows():
            print(f"\nExample {i+1}:")
            print(f"Question: {row['question']}")

            # Expected answer
            if 'expected_response' in df.columns:
                expected_response = row.get('expected_response')
                if expected_response is not None and not pd.isna(expected_response):
                    response_preview = str(expected_response)[:100]
                    if len(str(expected_response)) > 100:
                        response_preview += "..."
                    print(f"Expected answer: {response_preview}")
                else:
                    print("Expected answer: Not set")

            # Expected tools
            if 'expected_tools' in df.columns:
                tools = expected_tools_list[i] if i < len(
                    expected_tools_list) else []
                if tools:
                    print(f"Expected tools: {', '.join(tools)}")
                else:
                    print("Expected tools: Not set")

        print("=" * 50)
