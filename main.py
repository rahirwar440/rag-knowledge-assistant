import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI(title="RAG Knowledge Assistant")

# These get set up once when the app starts
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
""")

# Global vectorstore - starts empty, gets filled when a file is uploaded
vectorstore = None
retriever = None

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore, retriever

    # Save uploaded file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load and process the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Create (or add to) the vector store
    if vectorstore is None:
        vectorstore = Chroma.from_documents(chunks, embeddings)
    else:
        vectorstore.add_documents(chunks)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return {
        "filename": file.filename,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks),
        "status": "success"
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    if retriever is None:
        return {"error": "No documents uploaded yet. Please upload a file first using /upload."}

    docs = retriever.invoke(request.question)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = prompt.format_messages(context=context, question=request.question)
    response = llm.invoke(messages)

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


@app.get("/")
async def root():
    return {"message": "RAG Knowledge Assistant is running. Go to /docs to try it out."}