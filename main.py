import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

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
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    return llm


def get_prompt():
    global prompt
    if prompt is None:
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
""")
    return prompt


class ChatRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore, retriever

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    emb = get_embeddings()

    if vectorstore is None:
        vectorstore = Chroma.from_documents(chunks, emb)
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


@app.get("/")
async def root():
    return {"message": "RAG Knowledge Assistant is running. Go to /docs to try it out."}