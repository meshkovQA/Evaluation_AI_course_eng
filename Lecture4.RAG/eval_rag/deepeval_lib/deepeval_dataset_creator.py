"""
Dataset creation in DeepEval from Excel and RAG answers.
Uses the existing DatasetParser and RAGConnector.
"""
import sys
from pathlib import Path
import time

# Add path first
current_dir = Path(__file__).resolve().parent  # .../deepeval_lib
eval_rag_dir = current_dir.parent  # .../eval_rag
sys.path.insert(0, str(eval_rag_dir))

from deepeval.dataset import Golden  # noqa: E402
import time  # noqa: E402
from rag_connector import RAGConnector  # noqa: E402
from dataset_parser import DatasetParser  # noqa: E402
from deepeval.dataset import Golden, EvaluationDataset  # noqa: E402


def create_deepeval_dataset_from_excel(
    excel_path: str,
    dataset_alias: str,
    rag_connector: RAGConnector,
    sleep_time: float = 0.1
):

    # ============= STEP 1: Parse Excel =============

    parser = DatasetParser()
    df = parser.load_dataset(excel_path)

    if df is None:
        print("❌ Error loading the dataset")
        return

    info = parser.validate_dataset(df)
    print(f"\n📊 Dataset info:")
    print(f"   Total rows: {info['total_rows']}")
    print(f"   Valid pairs: {info['valid_pairs']}")

    # ============= STEP 2: Extract questions and expected answers =============
    questions = parser.get_questions(df)
    expected_responses = parser.get_expected_responses(df)

    print(f"✅ Extracted {len(questions)} questions")

    # ============= STEP 3: Query the RAG system =============
    print(f"\n🤖 Step 3: Getting answers from RAG system...")

    goldens = []

    for i, question in enumerate(questions, 1):
        print(f"\n  [{i}/{len(questions)}] {question[:60]}...")

        rag_response = rag_connector.query(question)

        if 'error' in rag_response:
            print(f"      ⚠️  RAG error: {rag_response['error']}")
            continue

        actual_output = rag_response.get('content', '')
        sources = rag_response.get('sources', [])
        retrieval_context = [s.get('content', '')
                             for s in sources if s.get('content')]

        print(f"      ✅ Answer: {actual_output[:60]}...")
        print(f"      📚 Contexts: {len(retrieval_context)}")

        golden = Golden(
            input=question,
            actual_output=actual_output,
            expected_output=expected_responses[i-1] if i -
            1 < len(expected_responses) else "",
            retrieval_context=retrieval_context
        )

        goldens.append(golden)

        time.sleep(sleep_time)

    # ============= STEP 4: Save to DeepEval =============
    print(f"\n💾 Step 4: Saving dataset to DeepEval...")

    try:
        from deepeval.dataset import EvaluationDataset

        evaluation_dataset = EvaluationDataset(goldens)

        evaluation_dataset.push(alias=dataset_alias)

        print(f"✅ Dataset '{dataset_alias}' successfully created in DeepEval!")
        print(f"📊 Saved {len(goldens)} records")
        print(f"🌐 View the dataset at https://app.confident-ai.com")

    except Exception as e:
        print(f"❌ Error saving to DeepEval: {e}")
        print("💡 Make sure:")
        print("   1. You ran: deepeval login")
        print("   2. You have an internet connection")
        print("   3. Your DeepEval API key is valid")


if __name__ == "__main__":

    # ============= CONFIGURATION =============

    # 1. Initialize RAG connector
    rag_connector = RAGConnector(
        endpoint_url="http://5.11.83.110:8002/api/v1/chat/",
        api_key="rag-api-key",
        timeout=30
    )

    # 2. Path to Excel file
    excel_path = "data/evaluation_dataset.xlsx"  # ← Your path

    # 3. Dataset alias in DeepEval UI
    dataset_alias = "Second dataset"  # ← Name in UI

    # ============= RUN =============

    create_deepeval_dataset_from_excel(
        excel_path=excel_path,
        dataset_alias=dataset_alias,
        rag_connector=rag_connector,
        sleep_time=0.1  # Pause between RAG queries
    )
