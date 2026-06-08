# RAG System API

A simple and powerful RAG (Retrieval-Augmented Generation) system for uploading documents and getting AI answers based on their content.

## 🎯 What is this?

The RAG system allows you to:
- Upload documents in various formats
- Automatically create vector indexes
- Ask questions and get answers based on document content
- See information sources in answers

## 🚀 Quick Start

### 1. Clone and configure
```bash
git clone <your-repo>
cd rag-system
cp .env.example .env
```

### 2. Run
```bash
docker-compose up --build
```

### 3. Check it works
```bash
curl "http://localhost:8002/health"
```

### 4. Open documentation
http://localhost:8002/docs

## 📁 Supported Formats

| Format | Extensions | Examples |
|--------|------------|---------|
| PDF | `.pdf` | Reports, presentations |
| Word | `.docx` | Documents, guides |
| Text | `.txt`, `.md` | Notes, README |
| Tables | `.csv`, `.xlsx` | Data, lists |
| Web | `.html` | Pages, articles |
| Data | `.json` | Configs, API responses |

## 🔧 Configuration

### Minimal configuration (.env)
```env
# Vector database (works without API keys)
VECTOR_DB_TYPE=chroma

# Embeddings - free HuggingFace model
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# For full functionality add OpenAI key
# OPENAI_API_KEY=sk-your-key-here
# DEFAULT_LLM_PROVIDER=openai
# EMBEDDING_PROVIDER=openai
```

### Advanced configuration
```env
# Chunk sizes
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200

# RAG parameters
MAX_RELEVANT_CHUNKS=5
SIMILARITY_THRESHOLD=0.7

# Limits
MAX_FILE_SIZE=52428800  # 50MB
```

## 📚 API Usage

### Upload a document
```bash
curl -X POST "http://localhost:8002/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "title=My Document"
```

### Ask a question
```bash
curl -X POST "http://localhost:8002/api/v1/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is this document about?",
    "max_relevant_chunks": 3
  }'
```

### List documents
```bash
curl "http://localhost:8000/api/v1/documents/"
```

### Search documents
```bash
curl "http://localhost:8002/api/v1/documents/search/?query=important+information"
```

## 🐍 Python Client

```python
import requests

class RAGClient:
    def __init__(self, base_url="http://localhost:8002"):
        self.api_url = f"{base_url}/api/v1"
    
    def upload_document(self, file_path, title=None):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'title': title or 'Document'}
            return requests.post(
                f"{self.api_url}/documents/upload",
                files=files, data=data
            ).json()
    
    def ask_question(self, question, max_chunks=3):
        payload = {
            "message": question,
            "max_relevant_chunks": max_chunks
        }
        return requests.post(
            f"{self.api_url}/chat/",
            json=payload
        ).json()
    
    def get_documents(self):
        return requests.get(
            f"{self.api_url}/documents/"
        ).json()

# Usage
client = RAGClient()

# Upload
result = client.upload_document('report.pdf', 'Q3 Report')
print(f"Uploaded: {result['document_id']}")

# Ask a question
answer = client.ask_question('What are the main results?')
print(f"Answer: {answer['content']}")
print(f"Sources: {len(answer['sources'])}")
```

## 🔍 Main Endpoints

### System
- `GET /` - System information
- `GET /health` - Health check
- `GET /info` - Configuration details

### Documents
- `POST /api/v1/documents/upload` - Upload a document
- `GET /api/v1/documents/` - List documents
- `GET /api/v1/documents/{id}` - Document info
- `DELETE /api/v1/documents/{id}` - Delete a document
- `GET /api/v1/documents/search/` - Search documents

### Chat
- `POST /api/v1/chat/` - Ask a question about documents

## 🔧 Local Development

### Requirements
- Python 3.11+
- Docker (optional)

### Setup
```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencies
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload
```

## 📊 Monitoring

### Health check
```bash
curl "http://localhost:8002/health"
```

### Docker logs
```bash
docker-compose logs -f rag-api
```

### Testing
```bash
python test_rag_system.py
```

## 🏗️ Architecture

```
📁 Upload document
    ↓
🔍 Parse content (PDF, DOCX, TXT...)
    ↓
✂️ Split into chunks
    ↓
🧠 Create embeddings
    ↓
💾 Save to vector DB
    ↓
❓ User question
    ↓
🔍 Search similar chunks
    ↓
🤖 Generate LLM response
    ↓
📋 Answer with sources
```

## ⚙️ Components

- **FastAPI** - REST API
- **ChromaDB** - vector database
- **HuggingFace/OpenAI** - embeddings and LLM
- **PyPDF2, python-docx** - document parsing
- **Docker** - containerization

## 🚨 Troubleshooting

### System won't start
```bash
# Check logs
docker-compose logs rag-api

# Rebuild
docker-compose down
docker-compose up --build
```

### API key errors
- API keys are not required for basic operation
- Use HuggingFace model (free)
- OpenAI key is only needed for full functionality

### File upload issues
- Maximum size: 50MB
- Supported formats: PDF, DOCX, TXT, CSV, JSON, HTML, MD, XLSX
- Check access permissions for the `storage/` folder

## 📋 Response Examples

### Successful document upload
```json
{
  "document_id": "doc-123",
  "filename": "report.pdf",
  "status": "processing",
  "message": "Document successfully uploaded and is being processed"
}
```

### Chat response with sources
```json
{
  "content": "According to the document, sales grew by 15%...",
  "sources": [
    {
      "document_title": "Q3 Report",
      "similarity": 0.89,
      "content": "Sales in Q3 amounted to $150M, a 15% increase..."
    }
  ],
  "metadata": {
    "sources_count": 1,
    "context_length": 850
  }
}
```

## 🔒 Security

- File type validation
- Upload size limits
- Safe filenames
- Isolation in Docker container

## 🌟 Extension Opportunities

- [ ] User authentication
- [ ] Image support in PDF
- [ ] Database integration
- [ ] Response caching
- [ ] Metrics and analytics

## 📞 Support

- **Documentation**: http://localhost:8002/docs
- **Health check**: http://localhost:8002/health
- **Testing**: `python test_rag_system.py`

## 📄 License

MIT License — use freely in your projects.
