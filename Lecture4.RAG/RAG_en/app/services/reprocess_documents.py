#!/usr/bin/env python3
"""
Script for forced re-processing of documents.
Run inside the container or with API access.
"""

import sys
import os
import time

sys.path.insert(0, '/home/app')

def force_reprocess_documents():
    """Forcefully re-processes all documents"""
    print("=" * 70)
    print("🔄 FORCED DOCUMENT REPROCESSING")
    print("=" * 70)

    try:
        from app.services.document_service import DocumentService
        from app.models.document import DocumentStatus
        import asyncio

        service = DocumentService()

        print(f"\n📚 Found documents: {len(service.documents)}")

        if not service.documents:
            print("⚠️  No documents to process")
            return

        for doc_id, document in service.documents.items():
            print(f"\n{'='*70}")
            print(f"📄 Document: {document.title}")
            print(f"   ID: {doc_id}")
            print(f"   Current status: {document.status.value}")
            print(f"   File: {document.file_path}")
            print(f"   Chunks: {len(document.chunks)}")

            if document.status == DocumentStatus.READY and len(document.chunks) > 0:
                print("   ✅ Document already processed, skipping")
                continue

            print(f"   🔄 Starting processing...")

            document.status = DocumentStatus.PROCESSING
            document.chunks = []
            document.content = ""

            try:
                asyncio.run(service._process_document(doc_id))

                time.sleep(1)

                if document.status == DocumentStatus.READY:
                    print(f"   ✅ Successfully processed!")
                    print(f"   📊 Created chunks: {len(document.chunks)}")
                elif document.status == DocumentStatus.ERROR:
                    print(f"   ❌ Error: {document.error_message}")
                else:
                    print(f"   ⚠️  Status: {document.status.value}")

            except Exception as e:
                print(f"   ❌ Exception during processing: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*70}")
        print("📊 FINAL STATISTICS")
        print(f"{'='*70}")

        total = len(service.documents)
        ready = sum(1 for d in service.documents.values() if d.status == DocumentStatus.READY)
        error = sum(1 for d in service.documents.values() if d.status == DocumentStatus.ERROR)
        processing = sum(1 for d in service.documents.values() if d.status == DocumentStatus.PROCESSING)

        print(f"Total documents: {total}")
        print(f"✅ Ready: {ready}")
        print(f"❌ With errors: {error}")
        print(f"🔄 Processing: {processing}")

        if error > 0:
            print(f"\n❌ Documents with errors:")
            for doc in service.documents.values():
                if doc.status == DocumentStatus.ERROR:
                    print(f"   - {doc.title}: {doc.error_message}")

    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()


def check_environment():
    """Checks the environment before processing"""
    print("\n🔍 ENVIRONMENT CHECK")
    print("=" * 70)

    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"✅ OPENAI_API_KEY is set: {openai_key[:10]}...{openai_key[-4:]}")
    else:
        print(f"❌ OPENAI_API_KEY is not set!")
        print("   Set it in the .env file or environment variables")
        return False

    storage_path = os.getenv('DOCUMENTS_STORAGE_PATH', '/home/app/storage/documents')
    vector_path = os.getenv('CHROMA_PERSIST_DIRECTORY', '/home/app/storage/vector_db/chroma')

    print(f"📂 Document storage: {storage_path}")
    print(f"   Exists: {os.path.exists(storage_path)}")

    print(f"📂 Vector DB: {vector_path}")
    print(f"   Exists: {os.path.exists(vector_path)}")

    return True


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "DOCUMENT REPROCESSING" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")

    if not check_environment():
        print("\n❌ Environment is not ready. Fix the issues and try again.")
        sys.exit(1)

    force_reprocess_documents()

    print("\n✅ Done! Check documents via API:")
    print("   GET http://localhost:8002/api/v1/documents/")
    print()
