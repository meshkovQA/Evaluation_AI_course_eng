# dataset_parser.py
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class DatasetParser:
    """Simple dataset parser for RAG evaluation"""

    def __init__(self):
        self.required_columns = ['question']  # Only question is required
        self.optional_columns = ['expected_response']

    def load_dataset(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load dataset from Excel or CSV file"""
        try:
            file_extension = Path(file_path).suffix.lower()

            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
            else:
                print(f"Unsupported file format: {file_extension}")
                return None

            missing_columns = [
                col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                print(f"Found columns: {list(df.columns)}")
                return None

            df_clean = df.dropna(subset=self.required_columns)

            print(f"Loaded dataset with {len(df_clean)} questions")
            return df_clean

        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def get_questions(self, df: pd.DataFrame) -> List[str]:
        """Get list of all questions"""
        return df['question'].tolist()

    def get_expected_responses(self, df: pd.DataFrame) -> List[str]:
        """Get list of expected answers, replacing NaN with empty string"""
        return df['expected_response'].fillna("").astype(str).tolist()

    def get_question_response_pairs(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Get list of question-answer pairs

        Returns:
            List of dicts with keys 'question' and 'expected_response'
        """
        pairs = []
        for _, row in df.iterrows():
            pairs.append({
                'question': row['question'],
                'expected_response': row['expected_response']
            })
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
            'avg_question_length': 0,
            'avg_response_length': 0
        }

        valid_pairs = 0
        question_lengths = []
        response_lengths = []

        for _, row in df.iterrows():
            question = str(row['question']).strip()
            response = str(row['expected_response']).strip()

            if not question or question == 'nan':
                info['empty_questions'] += 1
                continue

            if not response or response == 'nan':
                info['empty_responses'] += 1
                continue

            valid_pairs += 1
            question_lengths.append(len(question))
            response_lengths.append(len(response))

        info['valid_pairs'] = valid_pairs
        if question_lengths:
            info['avg_question_length'] = sum(
                question_lengths) / len(question_lengths)
        if response_lengths:
            info['avg_response_length'] = sum(
                response_lengths) / len(response_lengths)

        return info

    def preview_dataset(self, df: pd.DataFrame, n: int = 3):
        """Show first n examples from dataset"""
        print("\n=== Dataset Preview ===")
        for i, row in df.head(n).iterrows():
            print(f"\nExample {i+1}:")
            print(f"Question: {row['question']}")

            expected_response = row.get('expected_response')
            if expected_response is not None and not pd.isna(expected_response):
                print(f"Expected answer: {expected_response}")
            else:
                print("Expected answer: Not set")
        print("=" * 50)


# Example usage
if __name__ == "__main__":
    parser = DatasetParser()

    file_path = "data/evaluation_dataset.xlsx"  # replace with your path
    df = parser.load_dataset(file_path)

    if df is not None:
        info = parser.validate_dataset(df)
        print("\n=== Dataset info ===")
        print(f"Total rows: {info['total_rows']}")
        print(f"Valid pairs: {info['valid_pairs']}")
        print(f"Empty questions: {info['empty_questions']}")
        print(f"Empty answers: {info['empty_responses']}")
        print(
            f"Average question length: {info['avg_question_length']:.1f} characters")
        print(
            f"Average answer length: {info['avg_response_length']:.1f} characters")

        parser.preview_dataset(df)

        pairs = parser.get_question_response_pairs(df)
        print(f"\nRetrieved {len(pairs)} question-answer pairs")
