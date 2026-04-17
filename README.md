# RAG Production Pipeline

End-to-end Retrieval-Augmented Generation (RAG) system 
built from scratch and deployed to Azure.

## 🔗 Live API
https://rag-production-app.proudsea-e316e07e.eastus.azurecontainerapps.io/docs

## 🏗️ Architecture
PDF → Loader → Chunker → Embeddings → ChromaDB
↓
Question → Retriever → Reranker → LLM → Answer
↓
FastAPI (Azure)
## 🔧 Tech Stack
- **API**: FastAPI + Uvicorn
- **LLM**: OpenAI GPT-4o
- **Embeddings**: text-embedding-3-small
- **Vector Store**: ChromaDB
- **Reranker**: Cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Framework**: LangChain
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Cloud**: Azure Container Apps

## 📊 RAGAS Evaluation Scores
| Metric | Score |
|---|---|
| Faithfulness | 1.00 |
| Answer Relevancy | 0.98 |
| Context Precision | 1.00 |
| Context Recall | 1.00 |

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/sasifraza/rag-production.git
cd rag-production
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
```

### 2. Ingest Documents
```bash
# Add your PDF to data/raw/
python scripts/ingest.py
```

### 3. Run Locally
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Query the API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

## 🐳 Docker
```bash
docker build -t rag-production .
docker run -p 8000:8000 --env-file .env rag-production
```

## 📁 Project Structure

app/
├── ingestion/     # Load, chunk, embed documents
├── retrieval/     # Vector store + reranker
├── generation/    # LLM + prompt templates
├── pipeline/      # RAG orchestrator
└── utils/         # Shared helpers
config/            # Settings
eval/              # RAGAS evaluation
tests/             # Unit tests
scripts/           # CLI tools
docker/            # Docker config

## ✍️ Author
Syed Asif Raza, PhD
- LinkedIn: linkedin.com/in/syed-asif-raza-phd-873aab3
- GitHub: github.com/sasifraza