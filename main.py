import re
from langchain_core.documents import Document
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

app = FastAPI(title="RAG Knowledge Assistant")

# These stay None until first needed - keeps startup fast
embeddings = None
llm = None
prompt = None
vectorstore = None
retriever = None
all_chunks = []  # keeps a plain list of every chunk for keyword fallback search

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_embeddings():
    global embeddings
    if embeddings is None:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=hf_token
        )
    return embeddings


def get_llm():
    global llm
    if llm is None:
        from langchain_groq import ChatGroq
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")
    return llm


def get_prompt():
    global prompt
    if prompt is None:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. Be precise — if the context contains information about multiple people, make sure you match the exact person named in the question and do not mix their details with anyone else's. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
""")
    return prompt


class ChatRequest(BaseModel):
    question: str

def smart_split_pdf_documents(documents, splitter):
    """
    If the document looks like a list of structured records (e.g. an
    employee directory with repeated "Employee ID:" markers), split on
    those exact record boundaries so each chunk is one complete record —
    this avoids salary/name mismatches caused by character-count-based
    splitting cutting a record in half. Falls back to normal splitting
    for regular prose documents.
    """
    full_text = "\n".join(doc.page_content for doc in documents)
    source = documents[0].metadata.get("source", "") if documents else ""

    record_starts = list(re.finditer(r'(?=Employee ID\s*:)', full_text))
    if len(record_starts) >= 3:
        chunks = []
        positions = [m.start() for m in record_starts] + [len(full_text)]
        for i in range(len(positions) - 1):
            record_text = full_text[positions[i]:positions[i + 1]].strip()
            if record_text:
                chunks.append(Document(page_content=record_text, metadata={"source": source, "page": 0}))
        return chunks
    else:
        return splitter.split_documents(documents)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore, retriever

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ext = file.filename.lower().split(".")[-1]

    if ext == "pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # PDF text extraction sometimes drops line breaks between records
        # (e.g. "...8,80,000Employee 005..."). Insert a newline wherever a
        # lowercase letter/digit/comma is immediately followed by an
        # uppercase letter with no space — a common fix for this artifact.
        for doc in documents:
            cleaned = re.sub(r'(?<=[a-z0-9,])(?=[A-Z])', '\n', doc.page_content)
            doc.page_content = cleaned
    elif ext == "csv":
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(file_path)
        documents = loader.load()
    else:
        return {"error": f"Unsupported file type: .{ext}. Please upload a PDF or CSV."}

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    if ext == "pdf":
        chunks = smart_split_pdf_documents(documents, splitter)
    else:
        chunks = splitter.split_documents(documents)

    emb = get_embeddings()

    global all_chunks
    if vectorstore is None:
        vectorstore = Chroma.from_documents(chunks, emb)
    else:
        vectorstore.add_documents(chunks)

    all_chunks.extend(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return {
        "filename": file.filename,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "status": "success"
    }


class URLRequest(BaseModel):
    url: str


@app.post("/upload-url")
async def upload_url(request: URLRequest):
    global vectorstore, retriever

    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma

    try:
        loader = WebBaseLoader(request.url)
        documents = loader.load()
    except Exception as e:
        return {"error": f"Could not load URL: {str(e)}"}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    emb = get_embeddings()

    if vectorstore is None:
        vectorstore = Chroma.from_documents(chunks, emb)
    else:
        vectorstore.add_documents(chunks)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return {
        "filename": request.url,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "status": "success"
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    if retriever is None:
        return {"error": "No documents uploaded yet. Please upload a file first using /upload."}

    # Semantic search (embedding-based)
    vector_docs = retriever.invoke(request.question)

    # Keyword fallback: if the question contains a likely proper noun
    # (e.g. a person's name), also do an exact text search across all chunks.
    # This catches cases where semantic search alone confuses near-identical
    # structured records (like employee lists) that only differ by name/number.
    name_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', request.question)

    keyword_docs = []
    for name in name_matches:
        for chunk in all_chunks:
            if name.lower() in chunk.page_content.lower():
                keyword_docs.append(chunk)

    # If we found an exact name match, trust it completely and use ONLY
    # those chunks — mixing in generic semantic-search chunks for other
    # employees is what causes the LLM to cross-contaminate values between
    # different people's records.
    if keyword_docs:
        seen = set()
        docs = []
        for doc in keyword_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)
        docs = docs[:3]
    else:
        docs = vector_docs
    context = "\n\n".join(doc.page_content for doc in docs)

    p = get_prompt()
    model = get_llm()

    messages = p.format_messages(context=context, question=request.question)
    response = model.invoke(messages)

    sources = [
        {
            "page": doc.metadata.get("page", "unknown"),
            "source": doc.metadata.get("source", "unknown"),
            "snippet": doc.page_content[:150]
        }
        for doc in docs
    ]

    return {
        "answer": response.content,
        "sources": sources
    }


# @app.get("/")
# async def root():
#     return {"message": "RAG Knowledge Assistant is running. Go to /docs to try it out."}

app.mount("/", StaticFiles(directory="static", html=True), name="static")