# 🧠 RAG Knowledge Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system that lets you upload documents — PDFs, CSVs, or web URLs — and chat with an LLM about their contents, with accurate source citations.

🔗 **Live Demo**: [https://rag-knowledge-assistant-uttc.onrender.com](https://rag-knowledge-assistant-uttc.onrender.com)
📦 **Repo**: [github.com/rahirwar440/rag-knowledge-assistant](https://github.com/rahirwar440/rag-knowledge-assistant)

## 🎯 What It Does

Upload a document — or point it at a webpage — and ask questions about it in natural language. Answers are grounded in the source content and come with the exact page or snippet they were derived from, through a simple chat interface (no API tooling required to try it).

## 🏗️ Architecture

```
                          ┌── PDF  (PyPDFLoader)
User → FastAPI (async) ───┼── CSV  (CSVLoader)      → Chunking → Embeddings (HF Inference API)
                          └── URL  (WebBaseLoader)                        ↓
                                                                   ChromaDB (Vector Store)
                                                                            ↓
User Query → Retriever (top-k similarity search) → Groq LLM (Llama 3.1) → Answer + Sources
```

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (async endpoints) |
| Frontend | Vanilla HTML/CSS/JS chat UI, served as a static file |
| Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Hugging Face Inference API (`all-MiniLM-L6-v2`) |
| LLM | Groq (Llama 3.1 8B Instant) |
| Containerization | Docker |
| Deployment | Render |

## ✨ Key Features

- **Multi-format ingestion** — upload PDFs and CSVs directly, or fetch and index content straight from a URL
- **Chat-style UI** — a lightweight, ChatGPT-like interface for uploading documents and asking questions, no API client needed
- **Async FastAPI backend** for high-throughput document processing and querying
- **Source-cited answers** — every response includes the page number and exact text snippet it was derived from, expandable inline in the UI
- **Memory-optimized architecture** — uses lazy loading and API-based embeddings instead of loading heavy ML models in-process, keeping the app lightweight enough to run on free-tier cloud infrastructure (512MB RAM)
- **Fully containerized** with Docker for consistent, reproducible deployments

## 🚀 Getting Started (Local Setup)

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com)
- A free [Hugging Face API token](https://huggingface.co/settings/tokens)

### Installation

```bash
# Clone the repo
git clone https://github.com/rahirwar440/rag-knowledge-assistant.git
cd rag-knowledge-assistant

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

### Run Locally

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/` for the chat interface, or `http://127.0.0.1:8000/docs` for the interactive API documentation (Swagger UI).

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload a PDF or CSV; it gets chunked, embedded, and stored in ChromaDB |
| `/upload-url` | POST | Fetch a web page by URL, extract and index its text content |
| `/chat` | POST | Ask a question; returns an answer with source citations |

### Example Request

```json
POST /chat
{
  "question": "What was the company's revenue in 2025?"
}
```

### Example Response

```json
{
  "answer": "The company reported total revenue of $18.4 million in fiscal year 2025.",
  "sources": [
    {
      "page": 2,
      "source": "uploaded_files/report.pdf",
      "snippet": "In the fiscal year 2025, the company reported total revenue of 18.4 million dollars..."
    }
  ]
}
```

## 🐳 Docker

Build and run the container locally:

```bash
docker build -t rag-assistant .
docker run -p 7860:7860 --env-file .env rag-assistant
```

## 🧩 Design Decisions

- **Lazy loading**: Heavy dependencies (LangChain integrations, embedding clients) are imported only when a request actually needs them, rather than at startup. This keeps cold-start times low and memory usage well within free-tier cloud limits — a real constraint encountered and solved during deployment.
- **API-based embeddings over local models**: Instead of loading a sentence-transformer model in-process (which requires PyTorch and significant RAM), embeddings are generated via the Hugging Face Inference API — keeping the deployed footprint small enough to run on a 512MB instance.
- **Groq for inference**: Chosen for its free tier and very low-latency inference, avoiding the need for paid API credits or local GPU compute.
- **Static frontend, no framework**: The chat UI is plain HTML/CSS/JS served directly by FastAPI's `StaticFiles`, keeping the deployment single-service and avoiding a separate frontend build/deploy step.

## 🔮 Roadmap

- [ ] RAG evaluation pipeline using Ragas (faithfulness, answer relevancy, context precision)
- [ ] Streaming responses
- [ ] Multi-document conversation memory
- [ ] Persistent vector storage (currently in-memory per deploy)

## 📄 License

MIT
